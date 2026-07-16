use nexora_backend_ollama::OllamaBackend;
use nexora_core::chat_model::*;
use nexora_core::message::Message;
use nexora_core::tool::*;
use nexora_tools::ToolRegistryBuilder;

// ── Tool implementations ──────────────────────────────────

struct Calculator;

#[async_trait::async_trait]
impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Realiza operaciones matemáticas básicas (suma, resta, multiplicación, división)"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expresión matemática a evaluar (ej: '2 + 3 * 4')"
                }
            },
            "required": ["expression"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let expr = args
            .get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs("missing 'expression'".into()))?;
        let result = simple_eval(expr);
        Ok(ToolOutput::success(format!("Resultado: {}", result)))
    }
}

struct Weather;

#[async_trait::async_trait]
impl Tool for Weather {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Obtiene el clima actual de una ciudad"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Nombre de la ciudad"
                }
            },
            "required": ["city"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let city = args
            .get("city")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs("missing 'city'".into()))?;
        Ok(ToolOutput::success(format!(
            "Clima en {}: 22°C, soleado",
            city
        )))
    }
}

/// Ejemplo de tool calling con loop de agent.
///
/// Uso: cargo run -p nexora-examples --example tool_calling
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "llama3.2".into());
    let url = std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into());

    let backend = OllamaBackend::new(&url, &model);

    // Crear registry con herramientas
    let registry = ToolRegistryBuilder::new()
        .register(Calculator)
        .register(Weather)
        .build();

    let schemas = registry.schemas();

    // Estado del chat
    let mut messages = vec![
        Message::system(
            "Eres un asistente con acceso a herramientas. Usa las herramientas cuando sea necesario.",
        ),
        Message::user("¿Cuánto es 15 * 7 + 23? Y dime el clima en Madrid."),
    ];

    let params = GenerateParams {
        temperature: Some(0.3),
        max_tokens: Some(1024),
        ..Default::default()
    };

    println!("=== Tool Calling Agent ===\n");
    println!(
        "Usuario: {}",
        messages.last().unwrap().content.as_text().unwrap_or("")
    );
    println!();

    // Agent loop: máximo 5 iteraciones
    for iteration in 0..5 {
        println!("--- Iteración {} ---", iteration + 1);

        let response = backend.invoke(&messages, &schemas, &params).await?;
        println!(
            "Asistente: {}",
            response.message.content.as_text().unwrap_or("")
        );

        // Si no hay tool calls, terminar
        if response.tool_calls.is_empty() {
            println!("\n[FIN]");
            break;
        }

        // Ejecutar cada tool call
        messages.push(response.message.clone());
        let mut results = Vec::new();

        for tc in &response.tool_calls {
            println!("  → Tool call: {}({})", tc.name, tc.arguments);

            let result = match registry.execute(&tc.name, tc.arguments.clone()).await {
                Ok(output) => output.result,
                Err(e) => format!("Error: {}", e),
            };

            println!("  ← Resultado: {}", result);
            results.push(result);
        }

        // Agregar tool results como mensajes
        for (tc, result) in response.tool_calls.iter().zip(results.iter()) {
            messages.push(Message::tool(result.as_str(), tc.id.clone()));
        }

        println!();
    }

    Ok(())
}

fn simple_eval(expr: &str) -> f64 {
    let tokens: Vec<&str> = expr.split_whitespace().collect();
    if tokens.len() == 1 {
        return tokens[0].parse().unwrap_or(0.0);
    }

    let mut result: f64 = tokens[0].parse().unwrap_or(0.0);
    let mut i = 1;
    while i < tokens.len() - 1 {
        let op = tokens[i];
        let val: f64 = tokens[i + 1].parse().unwrap_or(0.0);
        match op {
            "+" => result += val,
            "-" => result -= val,
            "*" => result *= val,
            "/" => {
                if val != 0.0 {
                    result /= val;
                }
            }
            _ => {}
        }
        i += 2;
    }
    result
}
