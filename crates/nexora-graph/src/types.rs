use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::broadcast;

use crate::state::GraphState;

/// Identificador de nodo (Arc<str> para cheap cloning).
pub type NodeId = Arc<str>;

/// Router function: evalúa el estado y retorna el nombre del siguiente nodo.
pub type RouterFn<S> = Box<dyn Fn(&S) -> &'static str + Send + Sync>;

/// Tipo de función de nodo: async fn(GraphContext, S) -> S::Patch.
/// Recibe el contexto y un clone del estado actual, retorna un patch.
pub type NodeFn<S> = Arc<
    dyn Fn(
            Arc<GraphContext>,
            S,
        )
            -> Pin<Box<dyn Future<Output = Result<<S as GraphState>::Patch, String>> + Send>>
        + Send
        + Sync,
>;

/// Contexto compartido por todos los nodos del grafo.
pub struct GraphContext {
    /// Herramientas disponibles.
    pub tools: Arc<Vec<Box<dyn nexora_core::tool::Tool>>>,
    /// Sender para eventos de streaming.
    pub event_tx: broadcast::Sender<GraphEvent>,
    /// Metadatos de la ejecución.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GraphContext {
    pub fn new(tools: Vec<Box<dyn nexora_core::tool::Tool>>) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            tools: Arc::new(tools),
            event_tx,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Eventos emitidos durante la ejecución del grafo.
#[derive(Debug, Clone)]
pub enum GraphEvent {
    NodeStarted { node_id: NodeId },
    NodeCompleted { node_id: NodeId, duration_ms: u64 },
    NodeFailed { node_id: NodeId, error: String },
    StepStarted { step: usize },
    StepCompleted { step: usize },
    Token(String),
    Custom(String, serde_json::Value),
}

/// Configuración de ejecución del grafo.
pub struct RunConfig {
    pub thread_id: String,
    pub max_steps: usize,
    pub max_parallel: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            thread_id: uuid::Uuid::new_v4().to_string(),
            max_steps: 100,
            max_parallel: 4,
        }
    }
}

/// Resultado de la ejecución del grafo.
pub struct ExecutionResult<S: GraphState> {
    pub state: S,
    pub snapshots: Vec<S>,
    pub steps: usize,
    pub duration_ms: u64,
}

impl<S: GraphState> fmt::Debug for ExecutionResult<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionResult")
            .field("steps", &self.steps)
            .field("duration_ms", &self.duration_ms)
            .finish_non_exhaustive()
    }
}
