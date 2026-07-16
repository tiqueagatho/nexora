use nexora_graph::builder::GraphBuilder;
use nexora_graph::state::*;
use nexora_graph::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

// ── State ─────────────────────────────────────────────────

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PipelineState {
    data: HashMap<String, String>,
}

impl GraphState for PipelineState {
    type Patch = PipelinePatch;

    fn empty_patch() -> Self::Patch {
        PipelinePatch::default()
    }

    fn apply_patch(&mut self, patch: Self::Patch) {
        if let Some(new_data) = patch.data {
            self.data.extend(new_data);
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PipelinePatch {
    data: Option<HashMap<String, String>>,
}

impl StatePatch for PipelinePatch {
    fn merge(self, other: Self) -> Self {
        match (self.data, other.data) {
            (Some(mut a), Some(b)) => {
                a.extend(b);
                Self { data: Some(a) }
            }
            (Some(a), None) => Self { data: Some(a) },
            (None, Some(b)) => Self { data: Some(b) },
            (None, None) => Self { data: None },
        }
    }

    fn is_empty(&self) -> bool {
        self.data.is_none()
    }
}

// ── Parallel node functions ───────────────────────────────

async fn fetch_data_a(
    _ctx: Arc<GraphContext>,
    _state: PipelineState,
) -> Result<PipelinePatch, String> {
    println!("  [Nodo A] Descargando datos del API externo...");
    tokio::time::sleep(Duration::from_millis(500)).await;
    let mut data = HashMap::new();
    data.insert("source_a".into(), "datos del API A".into());
    Ok(PipelinePatch { data: Some(data) })
}

async fn fetch_data_b(
    _ctx: Arc<GraphContext>,
    _state: PipelineState,
) -> Result<PipelinePatch, String> {
    println!("  [Nodo B] Procesando base de datos local...");
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut data = HashMap::new();
    data.insert("source_b".into(), "datos de la DB B".into());
    Ok(PipelinePatch { data: Some(data) })
}

async fn fetch_data_c(
    _ctx: Arc<GraphContext>,
    _state: PipelineState,
) -> Result<PipelinePatch, String> {
    println!("  [Nodo C] Scraping web externa...");
    tokio::time::sleep(Duration::from_millis(700)).await;
    let mut data = HashMap::new();
    data.insert("source_c".into(), "datos scraping C".into());
    Ok(PipelinePatch { data: Some(data) })
}

async fn merge_results(
    _ctx: Arc<GraphContext>,
    state: PipelineState,
) -> Result<PipelinePatch, String> {
    let count = state.data.len();
    println!("  [Merge] {} fuentes de datos consolidadas", count);

    let mut data = HashMap::new();
    data.insert("merged".into(), format!("{} fuentes consolidadas", count));
    Ok(PipelinePatch { data: Some(data) })
}

/// Parallel Pipeline: ejecuta nodos A, B, C en paralelo usando un dispatcher.
///
/// Grafo: START -> dispatch -> [A, B, C] -> merge -> __end__
///
/// Uso: cargo run -p nexora-examples --example parallel_nodes
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Parallel Pipeline Example ===\n");

    async fn dispatch(
        _ctx: Arc<GraphContext>,
        state: PipelineState,
    ) -> Result<PipelinePatch, String> {
        println!("  [Dispatch] Iniciando pipeline paralelo...");
        Ok(PipelinePatch {
            data: Some(state.data.clone()),
        })
    }

    let compiled = GraphBuilder::new("parallel_pipeline")
        .add_node("dispatch", dispatch)
        .add_node("fetch_a", fetch_data_a)
        .add_node("fetch_b", fetch_data_b)
        .add_node("fetch_c", fetch_data_c)
        .add_node("merge", merge_results)
        .add_edge("START", "dispatch")
        .add_edge("dispatch", "fetch_a")
        .add_edge("dispatch", "fetch_b")
        .add_edge("dispatch", "fetch_c")
        .add_edge("fetch_a", "merge")
        .add_edge("fetch_b", "merge")
        .add_edge("fetch_c", "merge")
        .compile()?;

    let mut state = PipelineState {
        data: HashMap::new(),
    };

    let start = std::time::Instant::now();
    let _result = compiled.run(&mut state).await?;
    let elapsed = start.elapsed();

    println!("\n=== Resultado ===");
    println!("Tiempo total: {:?}", elapsed);
    println!("Fuentes: {:?}", state.data.keys().collect::<Vec<_>>());
    println!("Merge: {:?}", state.data.get("merged"));

    if elapsed < Duration::from_millis(1000) {
        println!("\n✓ Los nodos se ejecutaron en paralelo (tiempo < suma individual)");
    } else {
        println!("\n✗ Los nodos parecen haber corrido en serie");
    }

    Ok(())
}
