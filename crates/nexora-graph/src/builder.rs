use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;

use crate::state::GraphState;
use crate::types::{NodeFn, NodeId, RouterFn};

/// EdgeSource: punto de origen de una arista.
#[derive(Debug, Clone)]
pub enum EdgeSource {
    Start,
    Node(NodeId),
    Nodes(Vec<NodeId>),
}

/// Arista del grafo (estática).
#[derive(Debug, Clone)]
struct Edge {
    from: EdgeSource,
    to: NodeId,
}

/// Arista condicional con router function.
struct ConditionalEdge {
    from: NodeId,
}

/// Builder para construir grafos de estado de forma declarativa.
pub struct GraphBuilder<S: GraphState> {
    name: String,
    nodes: HashMap<NodeId, NodeFn<S>>,
    edges: Vec<Edge>,
    conditional_edges: Vec<ConditionalEdge>,
    routers: HashMap<NodeId, RouterFn<S>>,
    max_parallel: usize,
    max_steps: usize,
}

impl<S: GraphState> GraphBuilder<S> {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: HashMap::new(),
            edges: Vec::new(),
            conditional_edges: Vec::new(),
            routers: HashMap::new(),
            max_parallel: 4,
            max_steps: 100,
        }
    }

    /// Añadir un nodo al grafo.
    pub fn add_node<F, Fut>(mut self, id: &str, func: F) -> Self
    where
        F: Fn(Arc<crate::types::GraphContext>, S) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<S::Patch, String>> + Send + 'static,
    {
        let node_fn: NodeFn<S> = Arc::new(move |ctx, state| Box::pin(func(ctx, state)));
        self.nodes.insert(id.into(), node_fn);
        self
    }

    /// Añadir arista estática: A → B.
    pub fn add_edge(mut self, from: &str, to: &str) -> Self {
        let source = if from == "START" {
            EdgeSource::Start
        } else {
            EdgeSource::Node(from.into())
        };
        self.edges.push(Edge {
            from: source,
            to: to.into(),
        });
        self
    }

    /// Añadir arista desde múltiples fuentes (barrier/AND-join).
    pub fn add_edge_from_many(mut self, from: &[&str], to: &str) -> Self {
        let node_ids: Vec<NodeId> = from.iter().map(|s| (*s).into()).collect();
        self.edges.push(Edge {
            from: EdgeSource::Nodes(node_ids),
            to: to.into(),
        });
        self
    }

    /// Añadir arista condicional con tabla de routing.
    ///
    /// ```ignore
    /// graph.add_conditional_edge("judge", |state| {
    ///     if state.score > 0.8 { "end" } else { "retry" }
    /// }, &[("end", "__end__"), ("retry", "generate")]);
    /// ```
    pub fn add_conditional_edge<F>(
        mut self,
        from: &str,
        router: F,
        _routes: &[(&str, &str)],
    ) -> Self
    where
        F: Fn(&S) -> &'static str + Send + Sync + 'static,
    {
        let from_id: NodeId = from.into();
        self.routers.insert(from_id.clone(), Box::new(router));
        self.conditional_edges
            .push(ConditionalEdge { from: from_id });
        self
    }

    /// Configurar máximo de nodos en paralelo.
    pub fn with_max_parallel(mut self, max: usize) -> Self {
        self.max_parallel = max;
        self
    }

    /// Configurar máximo de pasos (prevenir loops infinitos).
    pub fn with_max_steps(mut self, max: usize) -> Self {
        self.max_steps = max;
        self
    }

    /// Compilar el grafo: valida estructura, calcula niveles topológicos.
    pub fn compile(self) -> Result<crate::executor::CompiledGraph<S>, GraphBuildError> {
        // Validar que todos los nodos referenciados existen
        for edge in &self.edges {
            match &edge.from {
                EdgeSource::Start => {}
                EdgeSource::Node(id) => {
                    if !self.nodes.contains_key(id) {
                        return Err(GraphBuildError::NodeNotFound(id.to_string()));
                    }
                }
                EdgeSource::Nodes(ids) => {
                    for id in ids {
                        if !self.nodes.contains_key(id) {
                            return Err(GraphBuildError::NodeNotFound(id.to_string()));
                        }
                    }
                }
            }
            if !self.nodes.contains_key(&edge.to) && edge.to.as_ref() != "__end__" {
                return Err(GraphBuildError::NodeNotFound(edge.to.to_string()));
            }
        }

        // Validar nodos en aristas condicionales
        for ce in &self.conditional_edges {
            if !self.nodes.contains_key(&ce.from) {
                return Err(GraphBuildError::NodeNotFound(ce.from.to_string()));
            }
        }

        // Validar que START tiene exactamente una arista
        let start_count = self
            .edges
            .iter()
            .filter(|e| matches!(e.from, EdgeSource::Start))
            .count();
        if start_count != 1 {
            return Err(GraphBuildError::InvalidStartEdges(start_count));
        }

        // Detectar nodos en ciclos (los que tienen conditional edges)
        let iterative_nodes: HashSet<NodeId> = self
            .conditional_edges
            .iter()
            .map(|ce| ce.from.clone())
            .collect();

        // Calcular niveles topológicos (solo nodos sin ciclos)
        let levels = self.topological_levels(&iterative_nodes)?;

        // Construir mapa de sucesores (solo aristas estáticas)
        let mut successors: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        for edge in &self.edges {
            match &edge.from {
                EdgeSource::Node(id) => {
                    successors
                        .entry(id.clone())
                        .or_default()
                        .insert(edge.to.clone());
                }
                EdgeSource::Nodes(ids) => {
                    for id in ids {
                        successors
                            .entry(id.clone())
                            .or_default()
                            .insert(edge.to.clone());
                    }
                }
                _ => {}
            }
        }

        // Construir mapa de predecesores
        let mut predecessors: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        for edge in &self.edges {
            match &edge.from {
                EdgeSource::Node(id) => {
                    predecessors
                        .entry(edge.to.clone())
                        .or_default()
                        .insert(id.clone());
                }
                EdgeSource::Nodes(ids) => {
                    for id in ids {
                        predecessors
                            .entry(edge.to.clone())
                            .or_default()
                            .insert(id.clone());
                    }
                }
                _ => {}
            }
        }

        // Mover routers al compiled graph
        let conditional_routers = self.routers;

        Ok(crate::executor::CompiledGraph {
            nodes: self.nodes,
            successors,
            predecessors,
            levels,
            iterative_nodes,
            conditional_routers,
            max_parallel: self.max_parallel,
            max_steps: self.max_steps,
        })
    }

    /// Calcular niveles topológicos (Kahn's algorithm), excluyendo nodos iterativos.
    fn topological_levels(
        &self,
        iterative_nodes: &HashSet<NodeId>,
    ) -> Result<Vec<Vec<NodeId>>, GraphBuildError> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for node_id in self.nodes.keys() {
            if !iterative_nodes.contains(node_id) {
                in_degree.entry(node_id.clone()).or_insert(0);
            }
        }

        // Inicializar in-degrees
        for edge in &self.edges {
            match &edge.from {
                EdgeSource::Start => {
                    if !iterative_nodes.contains(&edge.to) {
                        in_degree.entry(edge.to.clone()).or_insert(0);
                    }
                }
                EdgeSource::Node(id) => {
                    if !iterative_nodes.contains(id)
                        && !iterative_nodes.contains(&edge.to)
                        && self.nodes.contains_key(id)
                    {
                        in_degree.entry(edge.to.clone()).or_insert(0);
                        in_degree.entry(edge.to.clone()).and_modify(|d| *d += 1);
                    }
                }
                EdgeSource::Nodes(ids) => {
                    let valid_count = ids
                        .iter()
                        .filter(|id| !iterative_nodes.contains(*id) && self.nodes.contains_key(*id))
                        .count();
                    if valid_count > 0
                        && !iterative_nodes.contains(&edge.to)
                        && self.nodes.contains_key(&edge.to)
                    {
                        in_degree.entry(edge.to.clone()).or_insert(0);
                        in_degree
                            .entry(edge.to.clone())
                            .and_modify(|d| *d += valid_count);
                    }
                }
            }
        }

        let mut current_level: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(id, _)| id.clone())
            .collect();

        let mut levels = Vec::new();

        while !current_level.is_empty() {
            current_level.sort();
            levels.push(current_level.clone());

            let mut next_level = Vec::new();
            for node in &current_level {
                for edge in &self.edges {
                    let source_matches = match &edge.from {
                        EdgeSource::Node(id) => id == node,
                        EdgeSource::Nodes(ids) => ids.contains(node),
                        EdgeSource::Start => false,
                    };
                    if source_matches
                        && self.nodes.contains_key(&edge.to)
                        && !iterative_nodes.contains(&edge.to)
                    {
                        let deg = in_degree.get_mut(&edge.to).unwrap();
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 && !levels.iter().any(|l| l.contains(&edge.to)) {
                            next_level.push(edge.to.clone());
                        }
                    }
                }
            }
            current_level = next_level;
        }

        Ok(levels)
    }
}

impl<S: GraphState> std::fmt::Debug for GraphBuilder<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphBuilder")
            .field("name", &self.name)
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("edges", &self.edges.len())
            .field("conditional_edges", &self.conditional_edges.len())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GraphBuildError {
    #[error("nodo no encontrado: {0}")]
    NodeNotFound(String),

    #[error("arista desde START inválida: se esperaba 1, se encontraron {0}")]
    InvalidStartEdges(usize),

    #[error("ciclo detectado en el grafo (usa iterative execution)")]
    CycleDetected,

    #[error("error de topological sort: {0}")]
    TopologicalSort(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::CompiledGraph;
    use crate::state::{GraphState, StatePatch};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct SimpleState {
        count: u32,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct SimplePatch {
        delta: u32,
    }

    impl StatePatch for SimplePatch {
        fn merge(self, other: Self) -> Self {
            SimplePatch {
                delta: self.delta + other.delta,
            }
        }
        fn is_empty(&self) -> bool {
            self.delta == 0
        }
    }

    impl GraphState for SimpleState {
        type Patch = SimplePatch;

        fn apply_patch(&mut self, patch: Self::Patch) {
            self.count += patch.delta;
        }

        fn empty_patch() -> Self::Patch {
            SimplePatch { delta: 0 }
        }
    }

    fn noop_node(
        _ctx: Arc<crate::types::GraphContext>,
        _state: SimpleState,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<SimplePatch, String>> + Send>,
    > {
        Box::pin(async { Ok(SimplePatch { delta: 1 }) })
    }

    #[test]
    fn builder_new() {
        let graph: GraphBuilder<SimpleState> = GraphBuilder::new("test");
        assert_eq!(graph.name, "test");
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn builder_add_node_and_compile() {
        let graph = GraphBuilder::new("linear")
            .add_node("a", noop_node)
            .add_edge("START", "a")
            .compile();
        assert!(graph.is_ok());
    }

    #[test]
    fn builder_missing_node_error() {
        let graph: Result<CompiledGraph<SimpleState>, _> = GraphBuilder::new("broken")
            .add_edge("START", "nonexistent")
            .compile();
        assert!(graph.is_err());
    }

    #[test]
    fn builder_multiple_start_edges_error() {
        let graph: Result<CompiledGraph<SimpleState>, _> = GraphBuilder::new("bad")
            .add_node("a", noop_node)
            .add_edge("START", "a")
            .add_edge("START", "a")
            .compile();
        assert!(graph.is_err());
    }

    #[test]
    fn builder_conditional_edge() {
        let graph = GraphBuilder::new("cond")
            .add_node("a", noop_node)
            .add_node("b", noop_node)
            .add_edge("START", "a")
            .add_edge("a", "b")
            .add_conditional_edge("b", |_state| "__end__", &[("__end__", "__end__")])
            .compile();
        assert!(graph.is_ok());
        let g = graph.unwrap();
        assert!(!g.iterative_nodes.is_empty());
    }

    #[test]
    fn builder_debug_format() {
        let graph = GraphBuilder::new("debug_test")
            .add_node("n1", noop_node)
            .add_edge("START", "n1");
        let debug = format!("{:?}", graph);
        assert!(debug.contains("debug_test"));
        assert!(debug.contains("n1"));
    }

    #[test]
    fn builder_edge_source_variants() {
        let start = EdgeSource::Start;
        let node = EdgeSource::Node("a".into());
        let nodes = EdgeSource::Nodes(vec!["a".into(), "b".into()]);

        assert!(matches!(start, EdgeSource::Start));
        assert!(matches!(node, EdgeSource::Node(_)));
        assert!(matches!(nodes, EdgeSource::Nodes(_)));
    }
}
