use std::pin::Pin;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::message::Message;
use crate::tool::ToolSchema;

/// Chunk de streaming emitido por un backend LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Texto del token parcial.
    pub text: String,
    /// Tool calls acumulados hasta este punto (solo en el último chunk).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// La respuesta está completa.
    pub done: bool,
}

/// Respuesta completa de un LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
    pub finish_reason: FinishReason,
}

/// Llamada a herramienta devuelta por el LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Uso de tokens.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Razón de finalización de la generación.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

/// Parámetros de generación.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerateParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Vec<String>,
}

/// Capacidades de un backend LLM.
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_images: bool,
    pub max_context_length: u32,
    pub backend_type: BackendType,
}

/// Tipo de backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendType {
    CloudApi,
    LocalHttp,
    NativeLocal,
}

/// Trait principal para backends LLM.
///
/// Cualquier proveedor de inferencia (OpenAI, Ollama, candle local, etc.)
/// implementa este trait. El framework es agnóstico al backend.
///
/// # Ejemplo
///
/// ```ignore
/// let response = ollama.invoke(&messages, &tools, &GenerateParams::default()).await?;
/// println!("{}", response.message.content.as_text().unwrap());
/// ```
#[async_trait]
pub trait ChatModel: Send + Sync {
    /// Nombre identificativo del backend (para logs).
    fn name(&self) -> &str;

    /// Invocación con respuesta completa.
    async fn invoke(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        params: &GenerateParams,
    ) -> Result<ChatResponse, ModelError>;

    /// Streaming token a token.
    ///
    /// Devuelve un `Stream` que emite `StreamChunk` parciales.
    /// El último chunk tiene `done: true`.
    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        params: &GenerateParams,
    ) -> Result<
        Pin<Box<dyn futures::Stream<Item = Result<StreamChunk, ModelError>> + Send>>,
        ModelError,
    >;

    /// Capacidades declaradas del backend.
    fn capabilities(&self) -> ModelCapabilities;
}
