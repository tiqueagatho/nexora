use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use nexora_backend_api::OpenAIBackend;
use nexora_backend_ollama::OllamaBackend;
use nexora_core::chat_model::{ChatModel, GenerateParams};
use nexora_core::message::Message;

/// Open Sauces — framework open-source para agentes LLM en Rust.
#[derive(Parser)]
#[command(name = "nexora", version, about = "Open-source LLM agent framework in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ejecuta un agente con un mensaje del usuario.
    Run {
        /// Mensaje del usuario.
        #[arg(short, long)]
        message: String,

        /// Backend: ollama | openai
        #[arg(short, long, default_value = "ollama")]
        backend: String,

        /// Nombre del modelo (e.g. llama3.2, gpt-4o)
        #[arg(short = 'M', long, default_value = "llama3.2")]
        model: String,

        /// URL base del backend.
        #[arg(long)]
        url: Option<String>,

        /// API key (requerido para openai).
        #[arg(long)]
        api_key: Option<String>,

        /// System prompt.
        #[arg(short = 'p', long, default_value = "You are a helpful assistant.")]
        system_prompt: String,

        /// Temperatura de generación.
        #[arg(short = 't', long, default_value_t = 0.7)]
        temperature: f32,

        /// Máximo de tokens.
        #[arg(short = 'n', long)]
        max_tokens: Option<u32>,
    },

    /// Muestra información del sistema y backends disponibles.
    Info,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            message,
            backend,
            model,
            url,
            api_key,
            system_prompt,
            temperature,
            max_tokens,
        } => {
            run_agent(
                &message,
                &backend,
                &model,
                url.as_deref(),
                api_key.as_deref(),
                &system_prompt,
                temperature,
                max_tokens,
            )
            .await
        }
        Commands::Info => {
            println!("Open Sauces v{}", env!("CARGO_PKG_VERSION"));
            println!("Backend: ollama | openai");
            println!("Graph engine: hybrid DAG + Pregel");
            println!("Allocator: mimalloc (conditional)");
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_agent(
    message: &str,
    backend: &str,
    model: &str,
    url: Option<&str>,
    api_key: Option<&str>,
    system_prompt: &str,
    temperature: f32,
    max_tokens: Option<u32>,
) -> Result<()> {
    let model_box: Box<dyn ChatModel> = match backend {
        "ollama" => {
            let base_url = url.unwrap_or("http://localhost:11434");
            println!("[backend] Conectando a Ollama en {} con modelo {}", base_url, model);
            Box::new(OllamaBackend::new(base_url, model))
        }
        "openai" => {
            let key = api_key.context("Se requiere --api-key para el backend openai")?;
            let base_url = url.unwrap_or("https://api.openai.com/v1");
            println!("[backend] Conectando a OpenAI en {} con modelo {}", base_url, model);
            Box::new(OpenAIBackend::with_base_url(key, model, base_url))
        }
        other => {
            anyhow::bail!("Backend desconocido: {}. Soportados: ollama, openai", other);
        }
    };

    let params = GenerateParams {
        temperature: Some(temperature),
        max_tokens,
        ..Default::default()
    };

    let messages = vec![
        Message::system(system_prompt),
        Message::user(message),
    ];

    println!("[agent] Enviando mensaje...");
    let response = model_box
        .invoke(&messages, &[], &params)
        .await
        .context("Error invocando al modelo")?;

    match response.message.content.as_text() {
        Some(text) => println!("\n{}", text),
        None => println!("\n[respuesta vacía]"),
    }

    println!("\n[tokens] prompt={} completion={} total={}",
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        response.usage.total_tokens,
    );

    Ok(())
}
