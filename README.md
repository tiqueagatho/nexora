# Nexora

Open-source LLM agent framework in Rust. Modular, fast, memory-safe.

Nexora lets you build AI agents by composing **models**, **tools**, and **graphs** — with zero-cost abstractions and native concurrency. Swap backends (Ollama, OpenAI, local candle) without changing a line of agent code.

## Quick Start

```bash
# Clone
git clone https://github.com/user/nexora.git
cd nexora

# Build
cargo build --release

# Run with Ollama (default)
cargo run -- run -m "What is the capital of France?" --model llama3.2

# Run with OpenAI
cargo run -- run -m "Explain Rust ownership" --backend openai --model gpt-4o --api-key sk-...

# Run an example
cargo run -p nexora-examples --example tool_calling
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  User / CLI                                         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Graph Engine (DAG + Pregel hybrid)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Node A  │→ │  Node B  │→ │  Node C  │          │
│  │ (tools)  │  │  (LLM)   │  │ (decide) │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│       ↑ parallel execution via tokio                 │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  trait ChatModel (backend-agnostic)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Ollama  │  │  OpenAI  │  │  Candle  │          │
│  │  (HTTP)  │  │  (HTTP)  │  │  (CPU)   │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────┘
```

## Crates

| Crate | Description |
|-------|-------------|
| `nexora-core` | Core traits (`ChatModel`, `Tool`, `GraphState`) and types |
| `nexora-graph` | Graph builder, executor, state management, parallel execution |
| `nexora-tools` | Tool registry with timeout support |
| `nexora-tools-macros` | `#[derive(Tool)]` procedural macro |
| `nexora-backend-ollama` | Ollama backend (HTTP + SSE streaming) |
| `nexora-backend-api` | OpenAI-compatible backend (works with Azure, Groq, Together) |
| `nexora-allocator` | Conditional global allocator (mimalloc) + KV-cache arena |

## Core Concepts

### Define a tool

```rust
use nexora_core::tool::{Tool, ToolOutput, ToolError, ToolSchema};
use async_trait::async_trait;
use serde_json::{json, Value};

struct Calculator;

#[async_trait]
impl Tool for Calculator {
    fn name(&self) -> &str { "calculator" }
    fn description(&self) -> &str { "Evaluate math expressions" }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string", "description": "Math expression" }
            },
            "required": ["expression"]
        })
    }
    async fn run(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let expr = args["expression"].as_str().unwrap_or("0");
        let result = eval(expr); // your eval logic
        Ok(ToolOutput::success(json!({ "result": result })))
    }
}
```

### Build a graph

```rust
use nexora_graph::builder::GraphBuilder;
use nexora_graph::state::*;
use nexora_graph::types::*;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct MyState { score: f64 }

impl GraphState for MyState {
    type Patch = MyPatch;
    fn empty_patch() -> Self::Patch { MyPatch::default() }
    fn apply_patch(&mut self, patch: Self::Patch) {
        if let Some(s) = patch.score { self.score = s; }
    }
}

let graph = GraphBuilder::new("my_agent")
    .add_node("fetch", fetch_data)
    .add_node("analyze", analyze_data)
    .add_edge("START", "fetch")
    .add_edge("fetch", "analyze")
    .compile()?;

let mut state = MyState::default();
let result = graph.run(&mut state).await?;
```

### Swap backends

```rust
// Ollama (local, free)
let model = OllamaBackend::new("http://localhost:11434", "llama3.2");

// OpenAI (or any compatible API)
let model = OpenAIBackend::with_base_url("sk-...", "gpt-4o", "https://api.openai.com/v1");

// Both implement ChatModel — your agent code doesn't change
let response = model.invoke(&messages, &tools, &params).await?;
```

## Key Design Decisions

| Decision | Why |
|----------|-----|
| `trait ChatModel` (not Tower `Service`) | `async fn` is natural for LLM calls; Tower is middleware-only |
| Monomorphization (generics) | Zero-cost at runtime; `dyn Trait` only at I/O boundaries |
| Hybrid DAG + Pregel | Supports both acyclic pipelines and iterative loops |
| `Reducer<S>` as explicit trait | Compile-time safety, no magic runtime dispatch |
| `#[derive(Tool)]` macro | Eliminates boilerplate, ensures schema consistency |

## Running Tests

```bash
# All tests (56 tests, ~3s)
cargo test --workspace

# With clippy
cargo clippy --workspace

# Specific crate
cargo test -p nexora-graph
```

## Examples

```bash
cargo run -p nexora-examples --example simple_chat
cargo run -p nexora-examples --example tool_calling
cargo run -p nexora-examples --example reflection_agent
cargo run -p nexora-examples --example parallel_nodes
```

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make changes + add tests
4. `cargo test --workspace && cargo clippy --workspace`
5. Submit a PR

### Project Structure

```
nexora/
├── src/main.rs                    # CLI entry point
├── crates/
│   ├── nexora-core/              # Core traits + types
│   ├── nexora-graph/             # Graph engine
│   ├── nexora-tools/             # Tool registry
│   ├── nexora-tools-macros/      # #[derive(Tool)]
│   ├── nexora-backend-ollama/    # Ollama backend
│   ├── nexora-backend-api/       # OpenAI backend
│   ├── nexora-allocator/         # Memory allocator
│   └── nexora-examples/          # Runnable examples
└── PHASE1-ARCHITECTURE.md        # Full architecture document
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Nexora by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
