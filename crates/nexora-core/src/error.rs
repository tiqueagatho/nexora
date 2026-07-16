use thiserror::Error;

/// Errores del core del framework.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("backend error: {0}")]
    Model(#[from] ModelError),

    #[error("tool error: {0}")]
    Tool(#[from] crate::tool::ToolError),

    #[error("serialización JSON: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("graph error: {0}")]
    Graph(String),

    #[error("configuración inválida: {0}")]
    Config(String),
}

/// Errores de backends LLM.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("modelo no disponible: {0}")]
    Unavailable(String),

    #[error("rate limit excedido, retry en {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },

    #[error("contexto excedido: {0}")]
    ContextExceeded(String),

    #[error("content filter: {0}")]
    ContentFiltered(String),

    #[error("serialización: {0}")]
    Serialization(String),

    #[error("backend apagado")]
    ActorShutdown,

    #[error("timeout: {0}")]
    Timeout(String),

    #[error("error interno del backend: {0}")]
    Internal(String),
}
