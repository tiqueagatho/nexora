# Nexora — FASE 1: Informe Arquitectónico Completo

## Marco Genérico de Agentes LLM en Rust — Diseño Modular, Desacoplado y de Coste Cero

---

## 0. Resumen Ejecutivo

**Nexora** es un framework genérico para agentes LLM en Rust, inspirado en la elegancia de LangChain/LangGraph pero construido sobre los cimientos de la seguridad de memoria, concurrencia nativa y abstracciones de coste cero de Rust. El diseño se basa en **cuatro capas de abstracción** desacopladas que se combinan sin penalización en tiempo de ejecución:

| Capa | Abstracción | Patrón Clave | Coste |
|------|-------------|-------------|-------|
| **1. Backend** | Inferencia LLM desacoplada | `trait ChatModel` + Tower `Service` | Zero (monomorphizado) |
| **2. Herramientas** | Tipado estricto + JSON Schema | `#[derive(JsonSchema)]` + `schemars` | Zero (compile-time) |
| **3. Grafos** | Estado genérico + DAG/Pregel | `trait GraphState` + `Reducer<S>` | Zero (generics) |
| **4. Concurrencia** | Aislamiento por actores | Tokio `mpsc` + `broadcast` | ~50ns/message |

La clave arquitectónica: **cada capa es un crate independiente** que el usuario puede importar por separado, y las abstracciones se resuelven completamente en compile-time mediante monomorphización.

---

## 1. Sistema de Inferencia Desacoplado (Abstracción de LLM)

### 1.1 El Problema

Un framework genérico no puede asumir un proveedor. El usuario debe poder cambiar de `OpenAI` a `ollama` local a `candle` (CPU con AVX2) sin modificar una sola línea de código de su agente. La abstracción debe ser tan barata que un backend local con SIMD genere exactamente las mismas instrucciones que si el usuario hubiera escrito el código directamente.

### 1.2 Patrón Elegido: `trait ChatModel` + Tower `Service`

**Decisión de diseño:** Adoptar el patrón `trait ChatModel` como abstracción primaria, **no** Tower `Service` directamente. Tower se usa como capa de middleware (retry, rate-limit, timeout) por encima.

**Por qué no Tower `Service` como abstracción principal:**
- `Service<Request>` requiere `poll_ready` + `call` con `&mut self` — demasiado bajo nivel para la API de usuario
- La interfaz de LLM es naturalmente `async fn`, no `poll`-based
- Tower añade complejidad innecesaria en la capa de usuario

**Tower se usa como middleware interno:**

```
[Usuario] → trait ChatModel::invoke()
                    │
                    ▼
            ┌───────────────┐
            │ Tower Stack   │
            │ Retry         │
            │ RateLimit     │
            │ Timeout       │
            │ Tracing       │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │ ChatModel     │
            │ OpenAIBackend │
            │ OllamaBackend │
            │ CandleBackend │
            └───────────────┘
```

### 1.3 Definición del Trait `ChatModel`

```rust
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

// ── Mensajes ──────────────────────────────────────────────
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Content,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPart {
    Text { text: String },
    Image { url: String, mime: String },
}

// ── Respuesta ─────────────────────────────────────────────
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

// ── Parámetros de generación ──────────────────────────────
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
}

// ── El Trait Principal ────────────────────────────────────
#[async_trait]
pub trait ChatModel: Send + Sync {
    /// Nombre del backend (para logs y debugging)
    fn name(&self) -> &str;

    /// Invocación síncrona (respuesta completa)
    async fn invoke(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        params: &GenerateParams,
    ) -> Result<ChatResponse, ModelError>;

    /// Streaming token a token
    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        params: &GenerateParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, ModelError>> + Send>>, ModelError>;

    /// Capacidad del backend
    fn capabilities(&self) -> ModelCapabilities;
}

#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_images: bool,
    pub max_context_length: u32,
    pub backend_type: BackendType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendType {
    CloudApi,    // OpenAI, Anthropic, etc.
    LocalHttp,   // Ollama, LM Studio
    NativeLocal, // candle, llama.cpp bindings
}
```

### 1.4 El Trait `ToolSchema` — Puente entre Backend y Herramientas

```rust
/// Schema de herramienta que se envía al LLM (compatible con OpenAI/Anthropic format)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    #[schemars(rename = "parameters")]
    pub input_schema: serde_json::Value, // JSON Schema del input
}

/// Trait que cualquier herramienta debe implementar
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Nombre único de la herramienta
    fn name(&self) -> &str;

    /// Descripción para el LLM
    fn description(&self) -> &str;

    /// JSON Schema del input (generado compile-time via schemars)
    fn input_schema(&self) -> serde_json::Value;

    /// Ejecutar la herramienta con argumentos validados
    async fn execute(&self, args: serde_json::Value) -> Result<ToolOutput, ToolError>;

    /// Convertir a ToolSchema para enviar al backend
    fn as_schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}
```

### 1.5 Ejemplo: Backend Ollama

```rust
pub struct OllamaBackend {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaBackend {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.to_string(),
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl ChatModel for OllamaBackend {
    fn name(&self) -> &str {
        &self.model
    }

    async fn invoke(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        params: &GenerateParams,
    ) -> Result<ChatResponse, ModelError> {
        let body = OllamaRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            tools: if tools.is_empty() { None } else { Some(tools.to_vec()) },
            stream: Some(false),
            options: OllamaOptions {
                temperature: params.temperature,
                num_predict: params.max_tokens,
                ..Default::default()
            },
        };

        let resp = self.client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await?
            .json::<OllamaResponse>()
            .await?;

        Ok(resp.into())
    }

    // ... stream() implementation using SSE parsing
}
```

### 1.6 Backend Nativo CPU (candle/llama.cpp) — Dónde Brillan los Features

```rust
// Solo compilado cuando el feature "backend-native" está activo
#[cfg(feature = "backend-native")]
pub struct NativeBackend {
    model: llm::Model,
    session: llm::Session,
    device: Device,
}

#[cfg(feature = "backend-native")]
impl NativeBackend {
    pub fn load(path: &str, device: Device) -> Result<Self, ModelError> {
        let model = llm::load(path, &LoadOverrides {
            device: Some(device.clone()),
            #[cfg(feature = "avx2")]
            num_threads: num_cpus::get(),
            #[cfg(not(feature = "avx2"))]
            num_threads: 1,
            ..Default::default()
        })?;

        Ok(Self {
            model,
            session: llm::Session::new(&model),
            device,
        })
    }
}

// El mismo trait ChatModel, misma interfaz, diferente implementación
#[cfg(feature = "backend-native")]
#[async_trait]
impl ChatModel for NativeBackend {
    fn name(&self) -> &str { "native-candle" }

    async fn invoke(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        params: &GenerateParams,
    ) -> Result<ChatResponse, ModelError> {
        // Ejecución SYNCHRONOUS pero spawned en tokio::spawn_blocking
        // para no bloquear el runtime async
        let session = self.session.clone();
        let model = self.model.clone();
        let prompt = messages_to_prompt(messages);

        tokio::task::spawn_blocking(move || {
            let mut session = session;
            let response = model.generate(&mut session, &prompt, params, |token| {
                // Callback for streaming — forward to channel
                print!("{}", token);
                llm::CallbackResult::Continue
            })?;
            Ok(ChatResponse::from(response))
        }).await?
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_tools: false,  // Los backends nativos no soportan tool calling nativo
            supports_streaming: true,
            supports_images: false,
            max_context_length: self.model.context_length() as u32,
            backend_type: BackendType::NativeLocal,
        }
    }
}
```

### 1.7 El Middleware Stack (Tower) — Transparente al Usuario

```rust
use tower::{ServiceBuilder, Service};
use tower_http::{timeout::TimeoutLayer, trace::TraceLayer};

/// Construir un backend con middleware de forma transparente
pub fn build_backend(
    config: &BackendConfig,
) -> Box<dyn ChatModel> {
    let core: Box<dyn ChatModel> = match config.provider {
        Provider::OpenAI { api_key, model } => {
            Box::new(OpenAI::new(&api_key, &model))
        }
        Provider::Ollama { url, model } => {
            Box::new(OllamaBackend::new(&url, &model))
        }
        Provider::Native { path, device } => {
            Box::new(NativeBackend::load(&path, device)?)
        }
    };

    // Tower middleware se aplica como wrapper, no como parte del trait
    // El usuario nunca ve esto
    Box::new(MiddlewareBackend {
        inner: core,
        retry: RetryPolicy::new(3, Duration::from_millis(500)),
        timeout: Duration::from_secs(60),
    })
}
```

---

## 2. Tipado Estricto de Herramientas (Type-Safe Tools)

### 2.1 El Problema

En Python/Pydantic, las herramientas se definen como clases `BaseModel` y el schema JSON se genera por introspección en runtime. En Rust, queremos que **la compilación misma garantice** que la herramienta es válida, y que el schema JSON se genere sin coste en runtime.

### 2.2 El Macro `#[derive(Tool)]` — El Corazón del Sistema

El patrón clave: un **derive macro** que inspecta la estructura del input type y genera automáticamente:
1. La implementación del trait `Tool`
2. El JSON Schema (via `schemars`)
3. La validación de argumentos (via `serde`)

```rust
// ── Definición de la herramienta (SOLO esto) ──────────────
#[derive(Debug, Tool)]
#[tool(
    name = "get_weather",
    description = "Obtiene el clima actual de una ciudad"
)]
pub struct GetWeather {
    #[tool(param(
        description = "Nombre de la ciudad",
        example = "Madrid"
    ))]
    city: String,

    #[tool(param(
        description = "Unidades de temperatura",
        default = "celsius"
    ))]
    units: TemperatureUnit,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TemperatureUnit {
    Celsius,
    Fahrenheit,
}

// ── El derive macro genera AUTOMÁTICAMENTE: ───────────────
//
// impl Tool for GetWeather {
//     fn name(&self) -> &str { "get_weather" }
//     fn description(&self) -> &str { "Obtiene el clima actual de una ciudad" }
//     fn input_schema(&self) -> serde_json::Value {
//         // Generado por schemars::schema_for::<GetWeatherInput>()
//         // en compile-time, embebido como static
//         schema_for!(GetWeatherInput)
//     }
//     async fn execute(&self, args: serde_json::Value) -> Result<ToolOutput, ToolError> {
//         let input: GetWeatherInput = serde_json::from_value(args)
//             .map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
//         self.run_inner(input).await
//     }
// }
```

### 2.3 Anatomía del Macro Derive

```rust
// En crates/nexora-tools-macros/src/lib.rs

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

#[proc_macro_derive(Tool, attributes(tool))]
pub fn derive_tool(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Parse #[tool(name = "...", description = "...")] on the struct
    let tool_attrs = parse_tool_attrs(&input.attrs);

    // Parse #[tool(param(...))] on each field
    let field_params = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                fields.named.iter().map(|f| parse_param_attrs(f)).collect::<Vec<_>>()
            }
            _ => panic!("Tool structs must have named fields"),
        },
        _ => panic!("Tool can only be derived on structs"),
    };

    // Generate the input schema struct name
    let input_type = syn::Ident::new(&format!("{}Input", name), name.span());

    let tool_name = &tool_attrs.name;
    let tool_desc = &tool_attrs.description;

    let expanded = quote! {
        // Generate a Serde-compatible input type with JsonSchema
        #[derive(Debug, Clone, ::serde::Serialize, ::serde::Deserialize, ::schemars::JsonSchema)]
        pub struct #input_type {
            #(#field_params),*
        }

        impl ::nexora::tools::Tool for #name {
            fn name(&self) -> &str {
                #tool_name
            }

            fn description(&self) -> &str {
                #tool_desc
            }

            fn input_schema(&self) -> ::serde_json::Value {
                ::serde_json::to_value(
                    ::schemars::schema_for!(#input_type)
                ).expect("Tool schema must be valid JSON")
            }

            async fn execute(
                &self,
                args: ::serde_json::Value,
            ) -> Result<::nexora::tools::ToolOutput, ::nexora::tools::ToolError> {
                let input: #input_type = ::serde_json::from_value(args)
                    .map_err(|e| ::nexora::tools::ToolError::InvalidArgs(e.to_string()))?;
                self.run_impl(input).await
            }
        }
    };

    TokenStream::from(expanded)
}
```

### 2.4 Ventajas sobre Python/Pydantic

| Aspecto | Python (Pydantic) | Rust (nexora) |
|---------|-------------------|---------------------|
| Validación | Runtime (introspección) | Compile-time (serde derive) |
| Schema JSON | `model_json_schema()` en runtime | `schema_for!()` en compile-time |
| Errores | `ValidationError` en runtime | `Result<T, E>` tipado en compile-time |
| Coste | ~10-50μs por validación | ~0 (el compilador genera código óptimo) |
| Extensibilidad | Herencia / mixins | Traits + generics |

---

## 3. Grafos de Estados Genéricos (State Graphs)

### 3.1 El Problema

LangGraph permite definir flujos de trabajo como grafos de estados donde cada nodo es una función `State -> PartialState>. En Rust, queremos:
1. Que el usuario defina su propio tipo de estado (cualquier struct)
2. Que el grafo sea genérico sobre ese estado
3. Que los reducers se definan como traits, no como mágia de runtime
4. Soporte para ciclos (loops de reflexión) y ejecución paralela

### 3.2 Decisión Arquitectónica: Híbrido DAG + Pregel

**Descubrimiento clave de la investigación:** LangGraph NO es un executor DAG — es un modelo **Pregel de superpasos** (Bulk Synchronous Parallel). Los nodos se activan cuando sus canales de entrada se actualizan, no por estructura del grafo.

**Nuestro diseño híbrido:**
- **Porción acíclica** → Topological sort → ejecución paralela por niveles (más eficiente)
- **Ciclos** → Detección + conversión a ejecución iterativa con convergencia
- **Barrera de sincronización** → Todos los nodos de un nivel completan antes del siguiente

### 3.3 Trait `GraphState` — El Contrato

```rust
use serde::{Serialize, Deserialize};

/// Requisitos mínimos para un estado de grafo.
/// Cualquier struct que implemente estos traits puede usarse como estado.
pub trait GraphState: Clone + Send + Sync + 'static + Serialize + DeserializeOwned + std::fmt::Debug {
    /// Aplicar un patch parcial al estado (para reducers)
    fn apply_patch(&mut self, patch: Self::Patch);

    /// Crear un patch vacío (identity para el reducer)
    fn empty_patch() -> Self::Patch;
}

/// Tipo del patch (actualización parcial)
/// Cada implementador define qué campos pueden actualizarse
pub trait StatePatch: Clone + Send + Sync + 'static {
    /// Merge de dos patches (para cuando múltiples nodos escriben en el mismo step)
    fn merge(self, other: Self) -> Self;

    /// Verificar si el patch es vacío (no-op)
    fn is_empty(&self) -> bool;
}
```

### 3.4 Trait `Reducer` — Fusión de Escrituras Concurrentes

```rust
use async_trait::async_trait;

/// Los reducers determinan cómo se fusinan actualizaciones de múltiples nodos
/// que escriben al mismo campo en el mismo paso.
pub trait Reducer<S: GraphState>: Send + Sync {
    /// Fusionar un patch en el estado actual
    fn reduce(&self, state: &mut S, patch: S::Patch) -> Result<(), GraphError>;
}

/// Reducer de append (para listas de mensajes, logs, etc.)
pub struct AppendReducer;

impl<S: GraphState> Reducer<S> for AppendReducer {
    fn reduce(&self, _state: &mut S, _patch: S::Patch) -> Result<(), GraphError> {
        // El usuario implementa esto en su GraphState::apply_patch
        // con lógica de append para campos de tipo Vec
        Ok(())
    }
}

/// Reducer de último valor (default para escalares)
pub struct LastValueReducer;

impl<S: GraphState> Reducer<S> for LastValueReducer {
    fn reduce(&self, state: &mut S, patch: S::Patch) -> Result<(), GraphError> {
        state.apply_patch(patch);
        Ok(())
    }
}

/// Reducer de máximo (para scores, prioridades)
pub struct MaxReducer;

/// Reducer personalizado vía closure (zero-cost, monomorphizado)
pub struct FnReducer<F> {
    f: F,
}

impl<S, F> Reducer<S> for FnReducer<F>
where
    S: GraphState,
    F: Fn(&mut S, S::Patch) -> Result<(), GraphError> + Send + Sync,
{
    fn reduce(&self, state: &mut S, patch: S::Patch) -> Result<(), GraphError> {
        (self.f)(state, patch)
    }
}
```

### 3.5 El Builder — API de Construcción del Grafo

```rust
/// Nodo del grafo: una función async que transforma el estado
pub type NodeFn<S> = Arc<
    dyn Fn(Arc<GraphContext>, S) -> Pin<Box<dyn Future<Output = Result<S>> + Send>>
        + Send
        + Sync,
>;

/// Contexto compartido por todos los nodos
pub struct GraphContext {
    pub config: Arc<AgentConfig>,
    pub tools: Arc<Vec<Box<dyn Tool>>>,
    pub event_tx: broadcast::Sender<GraphEvent>,
}

/// Construilder de grafos — tipo-safe, builder pattern
pub struct GraphBuilder<S: GraphState> {
    nodes: HashMap<NodeId, NodeFn<S>>,
    edges: Vec<(EdgeSource, NodeId)>,
    conditional_edges: Vec<(NodeId, Box<dyn Fn(&S) -> NodeId + Send + Sync>)>,
    reducers: HashMap<String, Arc<dyn Reducer<S>>>,
    max_parallel: usize,
    max_steps: usize,
}

impl<S: GraphState> GraphBuilder<S> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            conditional_edges: Vec::new(),
            reducers: HashMap::new(),
            max_parallel: 4,
            max_steps: 100,
        }
    }

    /// Añadir un nodo al grafo
    pub fn add_node<F, Fut>(mut self, id: &str, func: F) -> Self
    where
        F: Fn(Arc<GraphContext>, S) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<S>> + Send + 'static,
    {
        let node_fn: NodeFn<S> = Arc::new(move |ctx, state| {
            Box::pin(func(ctx, state))
        });
        self.nodes.insert(id.into(), node_fn);
        self
    }

    /// Añadir arista estática (A → B)
    pub fn add_edge(mut self, from: &str, to: &str) -> Self {
        self.edges.push((EdgeSource::Node(from.into()), to.into()));
        self
    }

    /// Añadir arista desde múltiples fuentes (barrier/AND-join)
    pub fn add_edge_from_many(mut self, from: &[&str], to: &str) -> Self {
        for f in from {
            self.edges.push((EdgeSource::Node(f.into()), to.into()));
        }
        self
    }

    /// Añadir arista condicional
    pub fn add_conditional_edge<F>(mut self, from: &str, router: F) -> Self
    where
        F: Fn(&S) -> &str + Send + Sync + 'static,
    {
        self.conditional_edges.push((
            from.into(),
            Box::new(move |state| router(state).into()),
        ));
        self
    }

    /// Configurar reducer para un campo del estado
    pub fn with_reducer(mut self, field: &str, reducer: impl Reducer<S> + 'static) -> Self {
        self.reducers.insert(field.to_string(), Arc::new(reducer));
        self
    }

    /// Compilar el grafo (valida estructura, detecta ciclos)
    pub fn compile(self) -> Result<CompiledGraph<S>, GraphError> {
        // 1. Validar que todos los nodos referenciados existen
        // 2. Detectar ciclos y marcar nodos iterativos
        // 3. Calcular topological levels para la porción acíclica
        // 4. Construir el execution engine
        CompiledGraph::new(self)
    }
}
```

### 3.6 Ejemplo: Agente de Reflexión

```rust
// ── Estado del agente ─────────────────────────────────────
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub messages: Vec<Message>,
    pub iteration: u32,
    pub score: f32,
    pub done: bool,
}

impl GraphState for AgentState {
    fn apply_patch(&mut self, patch: Self::Patch) {
        if let Some(msgs) = patch.messages {
            self.messages.extend(msgs);
        }
        if let Some(i) = patch.iteration {
            self.iteration = i;
        }
        if let Some(s) = patch.score {
            self.score = s;
        }
        if let Some(d) = patch.done {
            self.done = d;
        }
    }

    fn empty_patch() -> Self::Patch {
        AgentStatePatch::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct AgentStatePatch {
    pub messages: Option<Vec<Message>>,
    pub iteration: Option<u32>,
    pub score: Option<f32>,
    pub done: Option<bool>,
}

impl StatePatch for AgentStatePatch {
    fn merge(self, other: Self) -> Self {
        Self {
            messages: self.messages.map(|mut m| {
                if let Some(o) = other.messages { m.extend(o); }
                m
            }).or(other.messages),
            iteration: self.iteration.or(other.iteration),
            score: self.score.or(other.score),
            done: self.done.or(other.done),
        }
    }

    fn is_empty(&self) -> bool {
        self.messages.is_none() && self.iteration.is_none()
            && self.score.is_none() && self.done.is_none()
    }
}

// ── Definición del grafo ──────────────────────────────────
let graph = GraphBuilder::<AgentState>::new()
    .add_node("analyze", |ctx, state| async move {
        let llm = ctx.config.llm();
        let response = llm.invoke(&state.messages, &[], &GenerateParams::default()).await?;
        Ok(AgentStatePatch {
            messages: Some(vec![response.message]),
            ..Default::default()
        })
    })
    .add_node("judge", |ctx, state| async move {
        let score = evaluate_quality(&state.messages);
        Ok(AgentStatePatch {
            score: Some(score),
            done: Some(score > 0.8),
            iteration: Some(state.iteration + 1),
            ..Default::default()
        })
    })
    .add_edge(START, "analyze")
    .add_edge("analyze", "judge")
    .add_conditional_edge("judge", |state| {
        if state.done { END } else { "analyze" }  // Ciclo de reflexión
    })
    .with_max_parallel(2)
    .with_max_steps(20)
    .compile()?;

let result = graph.run(initial_state).await?;
```

---

## 4. Concurrencia y Aislamiento por Actores

### 4.1 El Problema

Los agentes LLM ejecutan múltiples operaciones concurrentes: LLM calls, tool executions, streaming, checkpointing. Sin un patrón de concurrencia claro, el framework puede:
- Provocar deadlocks (bloqueos mutuos)
- Saturar la caché del hardware (thrashing entre cores)
- Crear condiciones de carrera en el estado

### 4.2 Solución: Actor System + Channel Map

El patrón actor aísla cada componente en su propia tarea Tokio, comunicándose exclusivamente por canales tipados. El estado mutable vive **dentro** del actor, nunca se comparte.

```
┌──────────────────────────────────────────────────────────┐
│                    Actor System                           │
│                                                          │
│  ┌──────────────┐   mpsc    ┌──────────────┐            │
│  │ InferenceActor│◄─────────│ GraphExecutor │            │
│  │  (model,     │  oneshot  │  (state,     │            │
│  │   kv_cache)  │──────────►│   channels)  │            │
│  └──────────────┘           └──────┬───────┘            │
│                                    │                     │
│                        ┌───────────┼───────────┐         │
│                        │           │           │         │
│                   mpsc │      mpsc │      mpsc │         │
│                        ▼           ▼           ▼         │
│              ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│              │ToolActor │  │ToolActor │  │ToolActor │   │
│              │ (http,   │  │ (db,     │  │ (calc,   │   │
│              │  search) │  │  cache)  │  │  eval)   │   │
│              └──────────┘  └──────────┘  └──────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────┐        │
│  │ StreamActor                                  │        │
│  │  broadcast::Receiver → WebSocket/SSE/Stdout  │        │
│  └──────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
```

### 4.3 Definición de Actores

```rust
// ── Mensajes entre actores (tagged union = dispatch zero-cost) ──
pub enum InferenceMsg {
    Chat {
        messages: Vec<Message>,
        tools: Vec<ToolSchema>,
        params: GenerateParams,
        respond: oneshot::Sender<Result<ChatResponse, ModelError>>,
    },
    Stream {
        messages: Vec<Message>,
        tools: Vec<ToolSchema>,
        params: GenerateParams,
        respond: mpsc::Sender<Result<StreamChunk, ModelError>>,
        done: oneshot::Sender<()>,
    },
    Shutdown,
}

// ── El Actor ──────────────────────────────────────────────
pub struct InferenceActor {
    receiver: mpsc::Receiver<InferenceMsg>,
    backend: Box<dyn ChatModel>,
    semaphore: Arc<Semaphore>,  // Limita concurrencia al backend
}

impl InferenceActor {
    pub fn new(
        receiver: mpsc::Receiver<InferenceMsg>,
        backend: Box<dyn ChatModel>,
        max_concurrent: usize,
    ) -> Self {
        Self {
            receiver,
            backend,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    pub async fn run(&mut self) {
        while let Some(msg) = self.receiver.recv().await {
            match msg {
                InferenceMsg::Chat { messages, tools, params, respond } => {
                    let _permit = self.semaphore.acquire().await.unwrap();
                    let result = self.backend.invoke(&messages, &tools, &params).await;
                    let _ = respond.send(result);
                }
                InferenceMsg::Stream { messages, tools, params, respond, done } => {
                    let _permit = self.semaphore.acquire().await.unwrap();
                    match self.backend.stream(&messages, &tools, &params).await {
                        Ok(mut stream) => {
                            while let Some(chunk) = stream.next().await {
                                if respond.send(chunk).await.is_err() { break; }
                            }
                        }
                        Err(e) => { let _ = respond.send(Err(e)).await; }
                    }
                    let _ = done.send(());
                }
                InferenceMsg::Shutdown => break,
            }
        }
    }
}

// ── Handle público (cheap to clone) ──────────────────────
#[derive(Clone)]
pub struct InferenceHandle {
    sender: mpsc::Sender<InferenceMsg>,
}

impl InferenceHandle {
    pub async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        params: &GenerateParams,
    ) -> Result<ChatResponse, ModelError> {
        let (tx, rx) = oneshot::channel();
        self.sender.send(InferenceMsg::Chat {
            messages: messages.to_vec(),
            tools: tools.to_vec(),
            params: params.clone(),
            respond: tx,
        }).await.map_err(|_| ModelError::ActorShutdown)?;
        rx.await.map_err(|_| ModelError::ActorShutdown)?
    }
}
```

### 4.4 Por Qué Esto Evita Deadlocks y Cache Thrashing

| Problema | Solución del Actor |
|----------|-------------------|
| **Deadlock** | No hay locks compartidos. Cada actor posee su estado exclusivamente. La comunicación es `channel.send().await` que nunca bloquea un hilo. |
| **Cache thrashing** | El estado del actor vive en una sola tarea Tokio. El work-stealing scheduler mantiene los datos calientes en L1/L2 del core donde ejecuta. No hay bouncing de líneas de caché entre cores. |
| **Data races** | `Send + Sync` enforced at compile time. Los tipos en canales cruzan boundaries de task → Rust exige `Send`. |
| **Backpressure** | `mpsc::channel(capacity)` — el sender hace `.await` cuando el buffer está lleno. Flujo controlado sin rate limiting explícito. |
| **Cancellation safety** | Si el caller dropea el `oneshot::Sender`, el actor detecta el error en `send()` y puede skippear trabajo. |

### 4.5 Escalado del Hardware del Usuario

```rust
/// Configuración de concurrencia adaptable al hardware
pub struct ConcurrencyConfig {
    /// Máximo de llamadas LLM concurrentes
    pub max_inference: usize,
    /// Máximo de tool executions concurrentes
    pub max_tools: usize,
    /// Máximo de nodos del grafo ejecutándose en paralelo
    pub max_graph_parallel: usize,
    /// Tamaño del buffer de streaming
    pub stream_buffer: usize,
}

impl ConcurrencyConfig {
    /// Auto-detectar hardware y configurar óptimamente
    pub fn auto_detect() -> Self {
        let cpus = num_cpus::get();
        let has_gpu = cfg!(feature = "cuda") && cuda::is_available();

        Self {
            max_inference: if has_gpu { 4 } else { 1 },
            max_tools: cpus.min(8),
            max_graph_parallel: cpus,
            stream_buffer: 256,
        }
    }

    /// Configuración para portátil viejo (CPU pura, AVX2)
    pub fn laptop_conservative() -> Self {
        Self {
            max_inference: 1,
            max_tools: 2,
            max_graph_parallel: 2,
            stream_buffer: 64,
        }
    }

    /// Configuración para servidor de producción (múltiples GPUs)
    pub fn server_production(gpu_count: usize) -> Self {
        Self {
            max_inference: gpu_count,
            max_tools: 32,
            max_graph_parallel: 64,
            stream_buffer: 1024,
        }
    }
}
```

---

## 5. Estrategia de Abstracciones de Coste Cero

### 5.1 Principios Fundamentales

| Abstracción | Mecanismo Rust | Coste Real |
|-------------|---------------|------------|
| `trait ChatModel` genérico | Monomorphización (generics) | **Zero** — el compilador genera código especializado por backend |
| `#[derive(Tool)]` | Procedural macro → código estático | **Zero** — se resuelve en compile-time |
| `trait GraphState` | Generic bounds `S: Clone + Send + Sync` | **Zero** — el compilador genera código por tipo de estado |
| `Reducer<S>` | Generics + `#[inline]` | **Zero** — monomorphizado, inlinable |
| Actor channels | `tokio::mpsc` | **~50ns** por mensaje (acceptable, es I/O async) |
| JSON Schema | `schemars::schema_for!()` | **Zero** en runtime (generado en compile-time) |
| Newtypes (`TokenId`, `NodeId`) | Struct wrappers | **Zero** — elided en assembly |
| SIMD kernels | `#[target_feature(enable = "avx2,fma")]` | **Zero** — emisión directa de instrucciones |

### 5.2 Estrategia de Monomorphización vs Dynamic Dispatch

```rust
// ── CASO 1: Backend conocido en compile-time (monomorphizado) ──
fn process<S: ChatModel>(model: &S, messages: &[Message]) -> ChatResponse {
    model.invoke(messages, &[], &GenerateParams::default())
}
// Compilador genera: process<OllamaBackend>, process<OpenAIBackend>, etc.
// Coste: ZERO. Es como haber escrito el código directamente.

// ── CASO 2: Backend elegido en runtime (dynamic dispatch) ──
fn process_dyn(model: &dyn ChatModel, messages: &[Message]) -> ChatResponse {
    model.invoke(messages, &[], &GenerateParams::default())
}
// Vtable lookup: ~10-20ns. Aceptable para I/O bound (LLM calls son >1ms).

// ── DECISIÓN: Usar generics en hot paths, dyn en boundaries ──
// - Inferencia del grafo: generics (el estado S se monomorphiza)
// - Selección de backend: dyn (elegido en runtime, I/O bound)
// - Tool execution: dyn (colección heterogénea de tools)
```

### 5.3 Feature Flags para Hardware

```toml
# Cargo.toml del workspace
[workspace]
resolver = "2"
members = [
    "crates/nexora-core",
    "crates/nexora-tools",
    "crates/nexora-graph",
    "crates/nexora-tools-macros",
    "crates/nexora-backend-api",      # OpenAI, Anthropic
    "crates/nexora-backend-ollama",    # Ollama local
    "crates/nexora-backend-native",    # candle/llama.cpp
    "crates/nexora-allocator",         # mimalloc/jemalloc
]

# crates/nexora-backend-native/Cargo.toml
[features]
default = []
avx2 = []
avx512 = []
neon = []      # ARM (Apple Silicon, Graviton)
fma = []
cuda = ["dep:cuda-sys"]
metal = ["dep:metal-rs"]

[target.'cfg(target_arch = "x86_64")'.dependencies]
# Solo en x86_64: autodetectar SIMD en runtime
[target.'cfg(target_arch = "aarch64")'.dependencies]
# Solo en ARM: NEON instructions
```

### 5.4 Asignador de Memoria Condicional

```rust
// crates/nexora-allocator/src/lib.rs

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[cfg(not(any(feature = "mimalloc", "jemalloc")))]
#[global_allocator]
static GLOBAL: std::alloc::System = std::alloc::System;

/// Arena allocator para KV-cache (bump allocation, O(1) free)
pub struct KVCacheArena {
    buffer: Vec<u8>,
    offset: usize,
}

impl KVCacheArena {
    pub fn new(capacity_mb: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity_mb * 1024 * 1024],
            offset: 0,
        }
    }

    #[inline(always)]
    pub fn alloc_f32_slice(&mut self, len: usize) -> &mut [f32] {
        let size = len * std::mem::size_of::<f32>();
        let align = std::mem::align_of::<f32>();
        let aligned = (self.offset + align - 1) & !(align - 1);
        let start = aligned;
        self.offset = aligned + size;
        unsafe { std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr().add(start) as *mut f32, len) }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}
```

---

## 6. Arquitectura del Workspace — Estructura Modular

### 6.1 Mapa de Dependencias

```
nexora/
├── Cargo.toml                  # workspace root
├── crates/
│   ├── nexora-core/       # Traits fundamentales: Message, ChatModel, Tool, Error
│   │   └── Cargo.toml          # Deps: serde, serde_json, async-trait, thiserror, schemars
│   │
│   ├── nexora-tools/      # Implementación de Tool trait + registry
│   │   └── Cargo.toml          # Deps: nexora-core, tokio
│   │
│   ├── nexora-tools-macros/ # proc-macro crate: #[derive(Tool)]
│   │   └── Cargo.toml          # Deps: syn, quote, proc-macro2, schemars
│   │
│   ├── nexora-graph/      # GraphBuilder, CompiledGraph, Reducer, GraphState
│   │   └── Cargo.toml          # Deps: nexora-core, tokio, futures, serde
│   │
│   ├── nexora-backend-api/ # Backends cloud: OpenAI, Anthropic
│   │   └── Cargo.toml          # Deps: nexora-core, reqwest, tokio
│   │
│   ├── nexora-backend-ollama/ # Backend Ollama
│   │   └── Cargo.toml          # Deps: nexora-core, reqwest, tokio
│   │
│   ├── nexora-backend-native/ # Backend candle/llama.cpp (opcional)
│   │   └── Cargo.toml          # Deps: nexora-core, candle (optional)
│   │
│   └── nexora-allocator/  # mimalloc/jemalloc + KV-cache arena
│       └── Cargo.toml          # Deps: mimalloc (optional), jemallocator (optional)
│
├── examples/
│   ├── simple_chat.rs           # Chat básico con Ollama
│   ├── tool_calling.rs          # Definición de tools + ejecución
│   ├── reflection_agent.rs      # Grafo con ciclo de reflexión
│   ├── parallel_analysis.rs     # Fan-out/fan-in paralelo
│   └── custom_backend.rs        # Implementar un backend propio
│
└── benchmarks/
    ├── inference_latency.rs
    ├── graph_execution.rs
    └── tool_validation.rs
```

### 6.2 Dependencias del Workspace

```toml
# nexora/Cargo.toml
[workspace]
resolver = "2"
members = ["crates/*"]
default-members = [
    "crates/nexora-core",
    "crates/nexora-tools",
    "crates/nexora-tools-macros",
    "crates/nexora-graph",
    "crates/nexora-backend-api",
    "crates/nexora-backend-ollama",
]

[workspace.dependencies]
# Fundamentales
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
async-trait = "0.1"
thiserror = "2.0"
schemars = "1.0"

# Async runtime
tokio = { version = "1", features = ["full"] }
futures = "0.3"

# HTTP (para backends API)
reqwest = { version = "0.12", features = ["json", "stream"] }

# Utilidades
tracing = "0.1"
uuid = { version = "1", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }

# Macros (proc-macro crates no pueden ser workspace deps)
# syn, quote, proc-macro2 se definen en nexora-tools-macros/Cargo.toml
```

### 6.3 Perfil de Compilación para Producción

```toml
# nexora/Cargo.toml (continuación)

[profile.release]
opt-level = 3                    # Optimización máxima
lto = "fat"                      # Link-Time Optimization completo (intra-crate)
codegen-units = 1                # Un solo codegen unit (mejor optimización)
panic = "abort"                  # Sin unwinding (más pequeño, más rápido)
strip = "symbols"                # Strip symbols (binario más pequeño)
overflow-checks = false          # Sin overflow checks en release
debug = false                    # Sin debug info

[profile.release.build-override]
opt-level = 3

# Perfil para benchmarks (más agresivo)
[profile.bench]
inherits = "release"
opt-level = 3
lto = "fat"
codegen-units = 1

# Perfil para desarrollo (compilación rápida)
[profile.dev]
opt-level = 0
debug = true
incremental = true

[profile.dev.package."*"]
opt-level = 2  # Dependencias externas sí optimizadas
```

### 6.4 Feature Flags del Usuario Final

```toml
# Ejemplo: cómo un usuario consume nexora

[dependencies]
nexora = { version = "0.1", features = [
    "backend-ollama",     # Backend Ollama
    "backend-api",        # Backends cloud (OpenAI, Anthropic)
    "tools-macros",       # #[derive(Tool)]
    "graph",              # State graph engine
    "allocator-mimalloc", # mimalloc (recomendado)
] }

# Para portátil viejo (CPU pura, sin GPU):
# nexora = { version = "0.1", features = ["backend-ollama", "graph"] }

# Para servidor con GPU:
# nexora = { version = "0.1", features = ["backend-api", "backend-native", "graph", "allocator-jemalloc"] }
```

---

## 7. Flujo de Datos Completo — Cómo Todo se Conecta

```
┌─────────────────────────────────────────────────────────────────┐
│                         USUARIO                                 │
│  define: AgentState, tools, graph structure                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  GraphBuilder::<AgentState>::new()                              │
│    .add_node("llm", llm_fn)                                    │
│    .add_node("tools", tool_fn)                                  │
│    .add_conditional_edge("tools", router)                       │
│    .compile() → CompiledGraph<AgentState>                       │
│                                                                 │
│  [compile-time]                                                 │
│    ✓ Validación de tipos (AgentState: GraphState)               │
│    ✓ Detección de ciclos                                       │
│    ✓ Topological sort → execution levels                        │
│    ✓ Monomorphización de nodos genéricos                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  graph.run(initial_state).await                                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Level 0: [fetch_data]          ← 1 nodo, secuencial    │    │
│  │ Level 1: [llm_analyze,         ← 2 nodos, PARALELO     │    │
│  │           sentiment_analyze]      via tokio::join!       │    │
│  │ Level 2: [integrate]           ← fan-in, waits for all  │    │
│  │ Level 3: [judge]               ← evalúa score           │    │
│  │                                                   ↓      │    │
│  │ Ciclo detectado: judge → llm_analyze (iteración)         │    │
│  │ Repetir hasta convergence o max_steps                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Cada nodo:                                                     │
│    1. Lee estado actual (Arc clone, no lock)                    │
│    2. Ejecuta función async                                     │
│    3. Retorna patch parcial                                     │
│    4. Reducer aplica patch al estado compartido                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Actor System (bajo el capó)                                    │
│                                                                 │
│  InferenceActor ←── mpsc ── GraphExecutor                       │
│       │                    │                                    │
│       ├── Backend Ollama   │ ToolActors (parallel)              │
│       ├── Backend OpenAI   │     ├── HTTP Tool                  │
│       └── Backend Candle   │     ├── DB Tool                    │
│                             │     └── Calc Tool                  │
│                             │                                    │
│  StreamActor ←── broadcast ── (tokens → WebSocket/SSE)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Resumen de Decisiones Arquitectónicas

| Decisión | Alternativa Descartada | Razón |
|----------|----------------------|-------|
| `trait ChatModel` como abstracción | Tower `Service` directo | Tower es demasiado bajo nivel para API de usuario; se usa como middleware interno |
| Monomorphización (generics) | `dyn Trait` en todo | Monomorphización = zero-cost; `dyn` solo en boundaries I/O-bound |
| DAG + Pregel hybrid | Solo DAG (topo sort) | Pregel permite ciclos nativos; híbrido cubre ambos casos |
| `Reducer<S>` trait | `Annotated` mágico de runtime | Traits son explícitos, comprobables en compile-time, zero-cost |
| Actor system | `Arc<RwLock<State>>` global | Actores eliminan deadlocks, cache thrashing, y data races |
| `#[derive(Tool)]` macro | Definición manual de Tool | La macro elimina boilerplate y garantiza consistencia compile-time |
| Feature flags por hardware | Runtime detection only | `#[cfg(feature)]` genera código solo para el hardware objetivo |
| `serde` para checkpoints | Formato custom | Serde es el estándar de Rust; backends pluggables (memory, SQLite, Redis) |
| `tokio` runtime | `async-std` / `smol` | Tokio es el estándar de facto; mejor ecosistema, work-stealing scheduler |

---

## 9. Próximos Pasos (Fase 2)

Con este análisis arquitectónico validado, la Fase 2 implementará:

1. **Workspace `Cargo.toml`** con todos los crates y dependencias
2. **`nexora-core`**: Traits `ChatModel`, `Tool`, `GraphState`, tipos base
3. **`nexora-tools-macros`**: El `#[derive(Tool)]` procedural macro
4. **`nexora-graph`**: `GraphBuilder`, `CompiledGraph`, `Reducer`, ejecución
5. **`nexora-backend-ollama`**: Backend funcional como referencia
6. **`nexora-allocator`**: mimalloc/jemalloc + KV-cache arena
7. **Ejemplos ejecutables**: Chat, tool calling, reflection agent, parallel analysis
8. **Benchmark harness**: latencia de inferencia, ejecución de grafo, validación

Cada componente será **real, compilable, testeable** — sin pseudocódigo.

---

## 10. Estado de Implementación (Fase 2 — 10/07/2026)

### Crates implementados

| Crate | Estado | Tests | Clippy | Descripción |
|-------|--------|-------|--------|-------------|
| `nexora-core` | ✅ Completo | 17 | ✓ limpio | `Message`, `Content`, `Role`, `Tool` trait, `ChatModel` trait, `CoreError`, `ModelError` |
| `nexora-tools-macros` | ✅ Completo | (compile) | ✓ limpio | `#[derive(Tool)]` proc macro con `OnceLock` estático |
| `nexora-tools` | ✅ Completo | 13 | ✓ limpio | `ToolRegistry`, `ToolRegistryBuilder`, ejecución con timeout |
| `nexora-graph` | ✅ Completo | 23 | ✓ limpio | `GraphBuilder`, `CompiledGraph`, `StatePatch`, `Reducer`, topological sort, parallel barrier, iterative convergence |
| `nexora-backend-ollama` | ✅ HTTP + SSE | — | ✓ limpio | `OllamaBackend` con streaming via `reqwest` + `futures::Stream` |
| `nexora-backend-api` | ✅ HTTP + SSE | — | ✓ limpio | `OpenAIBackend` con streaming, compatible OpenAI/Azure/Groq/Together |
| `nexora-allocator` | ✅ Arena | 3 | ✓ limpio | `KVCacheArena` arena allocator + conditional `#[global_allocator]` (mimalloc) |
| `nexora-examples` | ✅ 4 ejemplos | — | ✓ limpio | `simple_chat`, `tool_calling`, `reflection_agent`, `parallel_nodes` |
| `nexora` (bin) | ✅ CLI | — | ✓ limpio | `clap` CLI: `run` (inference) + `info` |

**Total: 56 tests, 0 failures, 0 clippy warnings**

### Traits y tipos clave implementados

```rust
// ── nexora-core ─────────────────────────────────────
pub trait ChatModel: Send + Sync {
    fn name(&self) -> &str;
    async fn invoke(&self, messages: &[Message], tools: &[ToolSchema], params: &GenerateParams)
        -> Result<ChatResponse, ModelError>;
    async fn stream(...) -> Result<Pin<Box<dyn Stream<Item = ...>>>, ModelError>;
    fn capabilities(&self) -> ModelCapabilities;
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    async fn run(&self, arguments: serde_json::Value) -> Result<ToolOutput, ToolError>;
}

// ── nexora-graph ────────────────────────────────────
pub trait GraphState: Clone + Send + Sync + 'static {
    type Patch: StatePatch;
    fn empty_patch() -> Self::Patch;
    fn apply_patch(&mut self, patch: Self::Patch);
    fn snapshot(&self) -> Self { self.clone() }  // default impl
}

pub trait StatePatch: Send + 'static {
    fn merge(self, other: Self) -> Self;
    fn is_empty(&self) -> bool;
}
```

### Arquitectura de grafo ejecutada

```
GraphBuilder::new("name")
    .add_node("fetch", fetch_fn)
    .add_node("analyze", analyze_fn)
    .add_edge("START", "fetch")
    .add_edge("fetch", "analyze")
    .add_conditional_edge("analyze", router_fn, &["refine", "__end__"])
    .compile()?
    → CompiledGraph<S>

compiled.run(&mut state).await?
    → ExecutionResult { state, snapshots, steps, duration_ms }
```

### Limitaciones conocidas

- `ToolRegistry` no tiene `Arc<dyn Tool>` genérico — requiere tipos concretos registrados
- No hay integración `ChatModel ↔ GraphContext` todavía (tools no se inyectan automáticamente al modelo)
- Actor system (InferenceActor, StreamActor) no implementado — pendiente Fase 2+
- No hay persistencia de checkpoints (serde checkpointer diseñado pero no implementado)
- `#[derive(Tool)]` no soporta variants enum — solo structs planos
