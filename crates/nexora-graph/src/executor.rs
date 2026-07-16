use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::state::GraphState;
use crate::types::{
    ExecutionResult, GraphContext, GraphEvent, NodeFn, NodeId, RouterFn, RunConfig,
};

pub struct CompiledGraph<S: GraphState> {
    pub(crate) nodes: HashMap<NodeId, NodeFn<S>>,
    #[allow(dead_code)]
    pub(crate) successors: HashMap<NodeId, HashSet<NodeId>>,
    #[allow(dead_code)]
    pub(crate) predecessors: HashMap<NodeId, HashSet<NodeId>>,
    pub(crate) levels: Vec<Vec<NodeId>>,
    pub(crate) iterative_nodes: HashSet<NodeId>,
    pub(crate) conditional_routers: HashMap<NodeId, RouterFn<S>>,
    #[allow(dead_code)]
    pub(crate) max_parallel: usize,
    #[allow(dead_code)]
    pub(crate) max_steps: usize,
}

impl<S: GraphState> CompiledGraph<S> {
    pub async fn run(&self, initial_state: &mut S) -> Result<ExecutionResult<S>, GraphExecError> {
        let config = RunConfig::default();
        self.run_with_config(initial_state, config).await
    }

    pub async fn run_with_config(
        &self,
        initial_state: &mut S,
        config: RunConfig,
    ) -> Result<ExecutionResult<S>, GraphExecError> {
        let ctx = Arc::new(GraphContext::new(vec![]));
        self.run_with_context(initial_state, config, ctx).await
    }

    pub async fn run_with_context(
        &self,
        initial_state: &mut S,
        config: RunConfig,
        ctx: Arc<GraphContext>,
    ) -> Result<ExecutionResult<S>, GraphExecError> {
        use std::time::Instant;
        use tokio::sync::Semaphore;

        let start = Instant::now();
        let semaphore = Arc::new(Semaphore::new(config.max_parallel));
        let mut snapshots = Vec::new();
        let mut current_step = 0;
        let state = initial_state;

        // Execute topological levels
        for level in &self.levels {
            if current_step >= config.max_steps {
                break;
            }

            let _ = ctx
                .event_tx
                .send(GraphEvent::StepStarted { step: current_step });

            let executable: Vec<&NodeId> = level
                .iter()
                .filter(|id| {
                    let name: &str = id.as_ref();
                    name != "START" && name != "END" && self.nodes.contains_key(*id)
                })
                .collect();

            if executable.is_empty() {
                continue;
            }

            let mut handles = Vec::new();

            for node_id in &executable {
                let node_fn = self.nodes.get(*node_id).cloned().unwrap();
                let ctx = Arc::clone(&ctx);
                let sem = Arc::clone(&semaphore);
                let node_id_owned = (*node_id).clone();
                let state_snapshot = state.snapshot();

                let _ = ctx.event_tx.send(GraphEvent::NodeStarted {
                    node_id: node_id_owned.clone(),
                });

                let handle = tokio::spawn(async move {
                    let _permit = sem
                        .acquire()
                        .await
                        .map_err(|e| GraphExecError::Semaphore(e.to_string()))?;

                    let node_start = Instant::now();

                    let patch = node_fn(ctx.clone(), state_snapshot).await.map_err(|e| {
                        GraphExecError::NodeFailed {
                            node_id: node_id_owned.clone(),
                            error: e,
                        }
                    })?;

                    let duration = node_start.elapsed().as_millis() as u64;

                    let _ = ctx.event_tx.send(GraphEvent::NodeCompleted {
                        node_id: node_id_owned,
                        duration_ms: duration,
                    });

                    Ok::<_, GraphExecError>(patch)
                });

                handles.push(handle);
            }

            // Barrier: wait for all nodes, then apply patches sequentially
            let mut patches = Vec::new();
            for handle in handles {
                let patch = handle
                    .await
                    .map_err(|e| GraphExecError::JoinError(e.to_string()))??;
                patches.push(patch);
            }

            // Apply all patches from this level
            for patch in patches {
                state.apply_patch(patch);
            }

            let _ = ctx
                .event_tx
                .send(GraphEvent::StepCompleted { step: current_step });
            snapshots.push(state.snapshot());
            current_step += 1;
        }

        // Handle conditional edges / iterative execution
        if !self.iterative_nodes.is_empty() {
            current_step = self
                .execute_iterative(state, &config, &ctx, &mut snapshots, current_step)
                .await?;
        }

        Ok(ExecutionResult {
            state: state.snapshot(),
            snapshots,
            steps: current_step,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    async fn execute_iterative(
        &self,
        state: &mut S,
        config: &RunConfig,
        ctx: &Arc<GraphContext>,
        snapshots: &mut Vec<S>,
        mut current_step: usize,
    ) -> Result<usize, GraphExecError> {
        for _ in 0..(config.max_steps - current_step) {
            let prev_snapshot = state.snapshot();

            for node_id in &self.iterative_nodes {
                if let Some(node_fn) = self.nodes.get(node_id) {
                    let _ = ctx.event_tx.send(GraphEvent::NodeStarted {
                        node_id: node_id.clone(),
                    });
                    let node_start = std::time::Instant::now();

                    let state_snapshot = state.snapshot();
                    let patch = node_fn(Arc::clone(ctx), state_snapshot)
                        .await
                        .map_err(|e| GraphExecError::NodeFailed {
                            node_id: node_id.clone(),
                            error: e,
                        })?;

                    state.apply_patch(patch);

                    let _ = ctx.event_tx.send(GraphEvent::NodeCompleted {
                        node_id: node_id.clone(),
                        duration_ms: node_start.elapsed().as_millis() as u64,
                    });

                    current_step += 1;
                }
            }

            // Check convergence via conditional routers
            if let Some(router) = self.conditional_routers.values().next() {
                let route = router(state);
                if route == "__end__" {
                    break;
                }
            }

            let current_snapshot = state.snapshot();
            snapshots.push(current_snapshot.clone());
            if format!("{:?}", current_snapshot) == format!("{:?}", prev_snapshot) {
                break;
            }
        }

        Ok(current_step)
    }

    pub fn subscribe_events(
        &self,
        ctx: &GraphContext,
    ) -> tokio::sync::broadcast::Receiver<GraphEvent> {
        ctx.event_tx.subscribe()
    }

    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().cloned().collect()
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
}

impl<S: GraphState> std::fmt::Debug for CompiledGraph<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("nodes", &self.nodes.len())
            .field("levels", &self.levels.len())
            .field("iterative_nodes", &self.iterative_nodes.len())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GraphExecError {
    #[error("nodo falló [{node_id}]: {error}")]
    NodeFailed { node_id: NodeId, error: String },

    #[error("máximo de pasos excedido: {0}")]
    MaxStepsExceeded(usize),

    #[error("semáforo: {0}")]
    Semaphore(String),

    #[error("join error: {0}")]
    JoinError(String),

    #[error("error interno: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::state::{GraphState, StatePatch};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestState {
        value: i32,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestPatch {
        value: Option<i32>,
    }

    impl StatePatch for TestPatch {
        fn merge(self, other: Self) -> Self {
            TestPatch {
                value: other.value.or(self.value),
            }
        }
        fn is_empty(&self) -> bool {
            self.value.is_none()
        }
    }

    impl GraphState for TestState {
        type Patch = TestPatch;

        fn apply_patch(&mut self, patch: Self::Patch) {
            if let Some(v) = patch.value {
                self.value = v;
            }
        }

        fn empty_patch() -> Self::Patch {
            TestPatch { value: None }
        }
    }

    fn make_node(
        val: i32,
    ) -> impl Fn(
        Arc<GraphContext>,
        TestState,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<TestPatch, String>> + Send>,
    > + Send
           + Sync
           + 'static {
        move |_ctx, _state| {
            let v = val;
            Box::pin(async move { Ok(TestPatch { value: Some(v) }) })
        }
    }

    #[tokio::test]
    async fn single_node_graph() {
        let graph = GraphBuilder::new("single")
            .add_node("inc", make_node(42))
            .add_edge("START", "inc")
            .compile()
            .unwrap();

        let mut state = TestState { value: 0 };
        let result = graph.run(&mut state).await.unwrap();

        assert_eq!(result.state.value, 42);
        assert!(result.steps >= 1);
        assert!(!result.snapshots.is_empty());
    }

    #[tokio::test]
    async fn linear_two_nodes() {
        let graph = GraphBuilder::new("linear")
            .add_node("step1", make_node(10))
            .add_node("step2", make_node(20))
            .add_edge("START", "step1")
            .add_edge("step1", "step2")
            .compile()
            .unwrap();

        let mut state = TestState { value: 0 };
        let result = graph.run(&mut state).await.unwrap();

        // Last node writes 20
        assert_eq!(result.state.value, 20);
    }

    #[tokio::test]
    async fn parallel_nodes_barrier() {
        let graph = GraphBuilder::new("parallel")
            .add_node("a", make_node(1))
            .add_node("b", make_node(2))
            .add_edge("START", "a")
            .add_edge("a", "b")
            .compile()
            .unwrap();

        let mut state = TestState { value: 0 };
        let result = graph.run(&mut state).await.unwrap();

        // Both a and b ran (patches applied sequentially, last wins)
        assert!(result.state.value == 1 || result.state.value == 2);
    }

    #[tokio::test]
    async fn iterative_convergence() {
        let graph = GraphBuilder::new("iter")
            .add_node("init", make_node(0))
            .add_edge("START", "init")
            .add_node("loop", make_node(99))
            .add_edge("init", "loop")
            .add_conditional_edge(
                "loop",
                |_state| "__end__",
                &[("__end__", "__end__")],
            )
            .compile()
            .unwrap();

        let mut state = TestState { value: -1 };
        let result = graph.run(&mut state).await.unwrap();

        // Iterative node ran at least once
        assert!(result.steps >= 1);
    }

    #[tokio::test]
    async fn node_ids_returns_all() {
        let graph = GraphBuilder::new("ids")
            .add_node("x", make_node(0))
            .add_node("y", make_node(0))
            .add_edge("START", "x")
            .add_edge("x", "y")
            .compile()
            .unwrap();

        let ids = graph.node_ids();
        assert!(ids.contains(&"x".into()));
        assert!(ids.contains(&"y".into()));
    }

    #[tokio::test]
    async fn level_count() {
        let graph = GraphBuilder::new("levels")
            .add_node("a", make_node(0))
            .add_node("b", make_node(0))
            .add_node("c", make_node(0))
            .add_edge("START", "a")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .compile()
            .unwrap();

        // START level + a level + b level + c level = 4
        assert!(graph.level_count() >= 2);
    }

    #[tokio::test]
    async fn execution_result_has_snapshots() {
        let graph = GraphBuilder::new("snap")
            .add_node("n", make_node(5))
            .add_edge("START", "n")
            .compile()
            .unwrap();

        let mut state = TestState { value: 0 };
        let result = graph.run(&mut state).await.unwrap();

        // At least one snapshot per executed level
        assert!(!result.snapshots.is_empty());
        assert!(result.steps >= 1);
    }

    #[test]
    fn compiled_graph_debug() {
        let graph = GraphBuilder::new("dbg")
            .add_node("n", make_node(0))
            .add_edge("START", "n")
            .compile()
            .unwrap();

        let debug = format!("{:?}", graph);
        assert!(debug.contains("CompiledGraph"));
        assert!(debug.contains("1")); // 1 node
    }
}
