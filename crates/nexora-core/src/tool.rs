use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Schema de herramienta en formato compatible con la API OpenAI.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// JSON Schema del input de la herramienta.
    #[schemars(rename = "parameters")]
    pub input_schema: serde_json::Value,
}

/// Output de una herramienta ejecutada.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub result: String,
    pub success: bool,
}

impl ToolOutput {
    pub fn success(result: impl Into<String>) -> Self {
        Self {
            result: result.into(),
            success: true,
        }
    }

    pub fn error(result: impl Into<String>) -> Self {
        Self {
            result: result.into(),
            success: false,
        }
    }
}

/// Errores de ejecución de herramientas.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("argumentos inválidos: {0}")]
    InvalidArgs(String),

    #[error("ejecución fallida: {0}")]
    ExecutionFailed(String),

    #[error("timeout ejecutando herramienta: {0}")]
    Timeout(String),

    #[error("herramienta no encontrada: {0}")]
    NotFound(String),

    #[error("error interno: {0}")]
    Internal(String),
}

impl Serialize for ToolError {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

/// Trait que toda herramienta debe implementar.
///
/// Las herramientas son funciones que el LLM puede invocar durante la generación.
/// Cada herramienta declara su schema JSON (generado compile-time via `schemars`)
/// y ejecuta su lógica de negocio de forma asíncrona.
///
/// # Ejemplo con derive macro
///
/// ```ignore
/// #[derive(Debug, Tool)]
/// #[tool(name = "get_weather", description = "Obtiene el clima de una ciudad")]
/// pub struct GetWeather {
///     #[tool(param(description = "Ciudad", example = "Madrid"))]
///     city: String,
/// }
///
/// impl GetWeather {
///     async fn run_impl(&self, input: GetWeatherInput) -> Result<ToolOutput, ToolError> {
///         Ok(ToolOutput::success(format!("Soleado en {}", input.city)))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Nombre único de la herramienta (snake_case).
    fn name(&self) -> &str;

    /// Descripción legible por humanos y LLM.
    fn description(&self) -> &str;

    /// JSON Schema del input (generado en compile-time).
    fn input_schema(&self) -> serde_json::Value;

    /// Ejecutar la herramienta con argumentos validados.
    async fn execute(&self, args: serde_json::Value) -> Result<ToolOutput, ToolError>;

    /// Convertir a `ToolSchema` para enviar al backend.
    fn as_schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_output_success() {
        let out = ToolOutput::success("42");
        assert!(out.success);
        assert_eq!(out.result, "42");
    }

    #[test]
    fn tool_output_error() {
        let out = ToolOutput::error("bad input");
        assert!(!out.success);
        assert_eq!(out.result, "bad input");
    }

    #[test]
    fn tool_error_display() {
        let e = ToolError::NotFound("calc".into());
        assert!(e.to_string().contains("calc"));

        let e = ToolError::InvalidArgs("bad".into());
        assert!(e.to_string().contains("bad"));

        let e = ToolError::ExecutionFailed("crash".into());
        assert!(e.to_string().contains("crash"));
    }

    #[test]
    fn tool_error_serialize() {
        let e = ToolError::NotFound("x".into());
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("x"));
    }

    #[test]
    fn tool_schema_roundtrip() {
        let schema = ToolSchema {
            name: "test_tool".into(),
            description: "A test tool".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let json = serde_json::to_string(&schema).unwrap();
        let deserialized: ToolSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "test_tool");
        assert_eq!(deserialized.description, "A test tool");
    }
}
