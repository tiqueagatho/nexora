use nexora_backend_ollama::OllamaBackend;
use nexora_core::chat_model::*;
use nexora_core::message::Message;
use nexora_graph::builder::GraphBuilder;
use nexora_graph::state::*;
use nexora_graph::types::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ── State ─────────────────────────────────────────────────

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct AgentState {
    messages: Vec<Message>,
    iteration: usize,
    critique: Option<String>,
}

impl GraphState for AgentState {
    type Patch = AgentPatch;

    fn empty_patch() -> Self::Patch {
        AgentPatch::default()
    }

    fn apply_patch(&mut self, patch: Self::Patch) {
        if let Some(msgs) = patch.messages {
            self.messages.extend(msgs);
        }
        if let Some(i) = patch.iteration {
            self.iteration = i;
        }
        if patch.critique.is_some() {
            self.critique = patch.critique;
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct AgentPatch {
    messages: Option<Vec<Message>>,
    iteration: Option<usize>,
    critique: Option<String>,
}

impl StatePatch for AgentPatch {
    fn merge(self, other: Self) -> Self {
        Self {
            messages: self.messages.or(other.messages),
            iteration: other.iteration.or(self.iteration),
            critique: other.critique.or(self.critique),
        }
    }

    fn is_empty(&self) -> bool {
        self.messages.is_none() && self.iteration.is_none() && self.critique.is_none()
    }
}

// ── Node functions ────────────────────────────────────────

async fn generate(_ctx: Arc<GraphContext>, state: AgentState) -> Result<AgentPatch, String> {
    let backend = OllamaBackend::new("http://localhost:11434", "llama3.2");

    let mut messages = vec![Message::system(
        "Eres un escritor creativo. Genera contenido breve y original.",
    )];
    messages.extend(state.messages.clone());

    let params = GenerateParams {
        temperature: Some(0.9),
        ..Default::default()
    };
    let response = backend
        .invoke(&messages, &[], &params)
        .await
        .map_err(|e| e.to_string())?;

    Ok(AgentPatch {
        messages: Some(vec![response.message]),
        ..Default::default()
    })
}

async fn critique(_ctx: Arc<GraphContext>, state: AgentState) -> Result<AgentPatch, String> {
    let backend = OllamaBackend::new("http://localhost:11434", "llama3.2");

    let last_content = state
        .messages
        .last()
        .and_then(|m| m.content.as_text())
        .unwrap_or("");

    let messages = vec![
        Message::system(
            "Eres un editor crítico. Analiza el texto y sugiere mejoras concisas. Si el texto es bueno, di 'APROBADO'.",
        ),
        Message::user(last_content),
    ];

    let params = GenerateParams {
        temperature: Some(0.3),
        ..Default::default()
    };
    let response = backend
        .invoke(&messages, &[], &params)
        .await
        .map_err(|e| e.to_string())?;

    let critique_text = response.message.content.as_text().unwrap_or("").to_string();

    Ok(AgentPatch {
        critique: Some(critique_text),
        ..Default::default()
    })
}

fn should_continue(state: &AgentState) -> &'static str {
    if state.iteration >= 3 {
        println!("  [Máximo de iteraciones alcanzado]");
        return "__end__";
    }

    match &state.critique {
        Some(c) if c.contains("APROBADO") => {
            println!("  [Texto aprobado por el crítico]");
            "__end__"
        }
        Some(_) => {
            println!(
                "  [Iteración {}: crítico sugiere cambios]",
                state.iteration + 1
            );
            "regenerate"
        }
        None => "regenerate",
    }
}

async fn reflect(_ctx: Arc<GraphContext>, state: AgentState) -> Result<AgentPatch, String> {
    Ok(AgentPatch {
        iteration: Some(state.iteration + 1),
        ..Default::default()
    })
}

async fn merge_critique(_ctx: Arc<GraphContext>, state: AgentState) -> Result<AgentPatch, String> {
    if let Some(ref critique) = state.critique {
        Ok(AgentPatch {
            messages: Some(vec![Message::user(format!(
                "Feedback del crítico: {}. Por favor, revisa tu texto.",
                critique
            ))]),
            iteration: Some(state.iteration),
            critique: None,
        })
    } else {
        Ok(AgentPatch::default())
    }
}

/// Reflection Agent: genera texto y lo refina iterativamente.
///
/// Uso: cargo run -p nexora-examples --example reflection_agent
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Reflection Agent ===\n");

    let compiled = GraphBuilder::new("reflection_agent")
        .add_node("generate", generate)
        .add_node("critique", critique)
        .add_node("reflect", reflect)
        .add_node("merge", merge_critique)
        .add_edge("START", "generate")
        .add_edge("generate", "critique")
        .add_conditional_edge(
            "critique",
            should_continue,
            &[("regenerate", "reflect"), ("__end__", "__end__")],
        )
        .add_edge("reflect", "merge")
        .add_edge("merge", "generate")
        .compile()?;

    let mut state = AgentState {
        messages: vec![Message::user(
            "Escribe un haiku sobre programación en Rust.",
        )],
        iteration: 0,
        critique: None,
    };

    println!("Prompt: Escribe un haiku sobre programación en Rust.\n");

    let _result = compiled.run(&mut state).await?;

    println!("\n=== Resultado Final ===");
    println!("Iteraciones: {}", state.iteration);
    if let Some(last) = state.messages.last() {
        println!("Texto final: {}", last.content.as_text().unwrap_or(""));
    }

    Ok(())
}
