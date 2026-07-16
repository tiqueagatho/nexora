use std::collections::HashMap;
use std::sync::Arc;

pub use nexora_core::tool::{Tool, ToolError, ToolOutput, ToolSchema};

/// Registro de herramientas disponibles.
///
/// Almacena herramientas por nombre y permite lookup rápido.
/// Thread-safe via `Arc`.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registrar una herramienta.
    pub fn register<T: Tool>(&mut self, tool: T) {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(tool));
    }

    /// Registrar una herramienta desde un `Arc`.
    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    /// Obtener una herramienta por nombre.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Listar todos los schemas de herramientas registradas.
    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|t| t.as_schema()).collect()
    }

    /// Número de herramientas registradas.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Verificar si el registro está vacío.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Ejecutar una herramienta por nombre.
    pub async fn execute(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolOutput, ToolError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;
        tool.execute(args).await
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder pattern para construir un ToolRegistry.
pub struct ToolRegistryBuilder {
    registry: ToolRegistry,
}

impl ToolRegistryBuilder {
    pub fn new() -> Self {
        Self {
            registry: ToolRegistry::new(),
        }
    }

    pub fn register<T: Tool>(mut self, tool: T) -> Self {
        self.registry.register(tool);
        self
    }

    pub fn build(self) -> ToolRegistry {
        self.registry
    }
}

impl Default for ToolRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct EchoTool;

    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes input back"
        }

        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {"msg": {"type": "string"}}})
        }

        async fn execute(&self, args: serde_json::Value) -> Result<ToolOutput, ToolError> {
            let msg = args
                .get("msg")
                .and_then(|v| v.as_str())
                .unwrap_or("empty");
            Ok(ToolOutput::success(msg))
        }
    }

    struct FailTool;

    #[async_trait::async_trait]
    impl Tool for FailTool {
        fn name(&self) -> &str {
            "fail"
        }
        fn description(&self) -> &str {
            "Always fails"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: serde_json::Value) -> Result<ToolOutput, ToolError> {
            Err(ToolError::ExecutionFailed("intentional".into()))
        }
    }

    struct CountingTool {
        count: Arc<AtomicU32>,
    }

    #[async_trait::async_trait]
    impl Tool for CountingTool {
        fn name(&self) -> &str {
            "count"
        }
        fn description(&self) -> &str {
            "Counts calls"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: serde_json::Value) -> Result<ToolOutput, ToolError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutput::success("ok"))
        }
    }

    #[tokio::test]
    async fn registry_new_empty() {
        let reg = ToolRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[tokio::test]
    async fn registry_register_and_get() {
        let mut reg = ToolRegistry::new();
        reg.register(EchoTool);
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 1);
        let tool = reg.get("echo").unwrap();
        assert_eq!(tool.name(), "echo");
    }

    #[tokio::test]
    async fn registry_get_missing() {
        let reg = ToolRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[tokio::test]
    async fn registry_execute() {
        let mut reg = ToolRegistry::new();
        reg.register(EchoTool);
        let output = reg
            .execute("echo", serde_json::json!({"msg": "hello"}))
            .await
            .unwrap();
        assert!(output.success);
        assert_eq!(output.result, "hello");
    }

    #[tokio::test]
    async fn registry_execute_not_found() {
        let reg = ToolRegistry::new();
        let err = reg
            .execute("nope", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::NotFound(_)));
    }

    #[tokio::test]
    async fn registry_execute_fail_tool() {
        let mut reg = ToolRegistry::new();
        reg.register(FailTool);
        let err = reg
            .execute("fail", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
    }

    #[tokio::test]
    async fn registry_schemas() {
        let mut reg = ToolRegistry::new();
        reg.register(EchoTool);
        let schemas = reg.schemas();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, "echo");
    }

    #[tokio::test]
    async fn registry_execute_shared_state() {
        let count = Arc::new(AtomicU32::new(0));
        let mut reg = ToolRegistry::new();
        reg.register(CountingTool {
            count: count.clone(),
        });

        reg.execute("count", serde_json::json!({})).await.unwrap();
        reg.execute("count", serde_json::json!({})).await.unwrap();
        reg.execute("count", serde_json::json!({})).await.unwrap();

        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn registry_multiple_tools() {
        let mut reg = ToolRegistry::new();
        reg.register(EchoTool);
        reg.register(FailTool);
        assert_eq!(reg.len(), 2);

        let o1 = reg
            .execute("echo", serde_json::json!({"msg": "ok"}))
            .await
            .unwrap();
        assert!(o1.success);

        let err = reg
            .execute("fail", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
    }

    #[test]
    fn builder_pattern() {
        let reg = ToolRegistryBuilder::new()
            .register(EchoTool)
            .register(FailTool)
            .build();
        assert_eq!(reg.len(), 2);
        assert!(reg.get("echo").is_some());
        assert!(reg.get("fail").is_some());
    }

    #[test]
    fn builder_default() {
        let reg = ToolRegistryBuilder::default().build();
        assert!(reg.is_empty());
    }

    #[test]
    fn registry_arc_register() {
        let mut reg = ToolRegistry::new();
        let tool: Arc<dyn Tool> = Arc::new(EchoTool);
        reg.register_arc(tool);
        assert_eq!(reg.len(), 1);
        assert!(reg.get("echo").is_some());
    }

    #[test]
    fn registry_overwrite_same_name() {
        let mut reg = ToolRegistry::new();
        reg.register(EchoTool);
        reg.register(EchoTool);
        assert_eq!(reg.len(), 1);
    }
}
