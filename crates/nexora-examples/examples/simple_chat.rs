use nexora_backend_api::OpenAIBackend;
use nexora_backend_ollama::OllamaBackend;
use nexora_core::chat_model::*;
use nexora_core::message::Message;

/// Chat simple con streaming.
///
/// Uso:
///   cargo run -p nexora-examples --example simple_chat --features backend-ollama
///   cargo run -p nexora-examples --example simple_chat --features backend-api
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "llama3.2".into());
    let backend_url = std::env::var("LLM_BACKEND").unwrap_or_else(|_| "ollama".into());

    let messages = vec![
        Message::system("Eres un asistente útil y conciso."),
        Message::user("¿Qué es Rust y por qué es especial?"),
    ];

    let params = GenerateParams {
        temperature: Some(0.7),
        max_tokens: Some(512),
        ..Default::default()
    };

    match backend_url.as_str() {
        "ollama" => {
            let url =
                std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into());
            let backend = OllamaBackend::new(&url, &model);

            println!("=== Streaming con Ollama ({}) ===\n", model);
            let mut stream = backend.stream(&messages, &[], &params).await?;

            use futures::StreamExt;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                print!("{}", chunk.text);
                if chunk.done {
                    println!();
                }
            }
        }
        "openai" => {
            let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
            let base_url = std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".into());
            let backend = OpenAIBackend::with_base_url(&api_key, &model, &base_url);

            println!("=== Streaming con OpenAI ({}) ===\n", model);
            let mut stream = backend.stream(&messages, &[], &params).await?;

            use futures::StreamExt;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                print!("{}", chunk.text);
                if chunk.done {
                    println!();
                }
            }
        }
        other => {
            eprintln!("Backend desconocido: {}. Usa 'ollama' o 'openai'.", other);
            std::process::exit(1);
        }
    }

    Ok(())
}
