use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;

/// Trait base para estados de grafo.
///
/// Cualquier struct que implemente este trait puede usarse como estado
/// en un `GraphBuilder`. El framework es completamente genérico sobre el tipo de estado.
///
/// # Ejemplo
///
/// ```ignore
/// #[derive(Debug, Clone, Serialize, Deserialize)]
/// pub struct MyAgentState {
///     pub messages: Vec<Message>,
///     pub iteration: u32,
/// }
///
/// impl GraphState for MyAgentState {
///     type Patch = MyAgentStatePatch;
///
///     fn apply_patch(&mut self, patch: Self::Patch) {
///         if let Some(msgs) = patch.messages {
///             self.messages.extend(msgs);
///         }
///         if let Some(i) = patch.iteration {
///             self.iteration = i;
///         }
///     }
///
///     fn empty_patch() -> Self::Patch {
///         MyAgentStatePatch::default()
///     }
/// }
/// ```
pub trait GraphState: Clone + Send + Sync + 'static + Debug + Serialize + DeserializeOwned {
    /// Tipo del patch parcial que los nodos retornan.
    type Patch: StatePatch;

    /// Aplicar un patch al estado.
    fn apply_patch(&mut self, patch: Self::Patch);

    /// Crear un patch vacío (identity para reducers).
    fn empty_patch() -> Self::Patch;

    /// Tomar un snapshot del estado actual (clonar).
    fn snapshot(&self) -> Self {
        self.clone()
    }
}

/// Trait para patches parciales de estado.
pub trait StatePatch: Clone + Send + Sync + 'static + Debug {
    /// Merge de dos patches (cuando múltiples nodos escriben en el mismo step).
    fn merge(self, other: Self) -> Self;

    /// Verificar si el patch es vacío (no-op).
    fn is_empty(&self) -> bool;
}

/// Trait para reducers: determinan cómo se fusinan actualizaciones concurrentes.
pub trait Reducer<S: GraphState>: Send + Sync {
    /// Fusionar un patch en el estado.
    fn reduce(&self, state: &mut S, patch: S::Patch) -> Result<(), String>;
}

// ── Reducers Built-in ─────────────────────────────────────

/// Reducer de último valor (default para escalares).
/// Reemplaza el campo con el valor más reciente.
pub struct LastValueReducer;

impl<S: GraphState> Reducer<S> for LastValueReducer {
    fn reduce(&self, state: &mut S, patch: S::Patch) -> Result<(), String> {
        state.apply_patch(patch);
        Ok(())
    }
}

/// Reducer de append (para listas de mensajes, logs, etc.).
/// El usuario debe implementar la lógica de append en `GraphState::apply_patch`.
pub struct AppendReducer;

impl<S: GraphState> Reducer<S> for AppendReducer {
    fn reduce(&self, state: &mut S, patch: S::Patch) -> Result<(), String> {
        state.apply_patch(patch);
        Ok(())
    }
}

/// Reducer personalizado via closure (zero-cost, monomorphizado).
pub struct FnReducer<F> {
    f: F,
}

impl<F> FnReducer<F> {
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<S, F> Reducer<S> for FnReducer<F>
where
    S: GraphState,
    F: Fn(&mut S, S::Patch) -> Result<(), String> + Send + Sync,
{
    fn reduce(&self, state: &mut S, patch: S::Patch) -> Result<(), String> {
        (self.f)(state, patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestState {
        value: i32,
        log: Vec<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestPatch {
        value: Option<i32>,
        log_entry: Option<String>,
    }

    impl StatePatch for TestPatch {
        fn merge(self, other: Self) -> Self {
            TestPatch {
                value: other.value.or(self.value),
                log_entry: other.log_entry.or(self.log_entry),
            }
        }

        fn is_empty(&self) -> bool {
            self.value.is_none() && self.log_entry.is_none()
        }
    }

    impl GraphState for TestState {
        type Patch = TestPatch;

        fn apply_patch(&mut self, patch: Self::Patch) {
            if let Some(v) = patch.value {
                self.value = v;
            }
            if let Some(entry) = patch.log_entry {
                self.log.push(entry);
            }
        }

        fn empty_patch() -> Self::Patch {
            TestPatch {
                value: None,
                log_entry: None,
            }
        }
    }

    #[test]
    fn state_patch_apply_value() {
        let mut state = TestState {
            value: 0,
            log: vec![],
        };
        let patch = TestPatch {
            value: Some(42),
            log_entry: None,
        };
        state.apply_patch(patch);
        assert_eq!(state.value, 42);
    }

    #[test]
    fn state_patch_apply_log() {
        let mut state = TestState {
            value: 0,
            log: vec![],
        };
        let patch = TestPatch {
            value: None,
            log_entry: Some("step1".into()),
        };
        state.apply_patch(patch);
        assert_eq!(state.log, vec!["step1"]);
    }

    #[test]
    fn state_snapshot_clone() {
        let state = TestState {
            value: 7,
            log: vec!["a".into()],
        };
        let snap = state.snapshot();
        assert_eq!(snap.value, 7);
        assert_eq!(snap.log, vec!["a"]);
    }

    #[test]
    fn state_empty_patch_is_empty() {
        let patch = TestState::empty_patch();
        assert!(patch.is_empty());
    }

    #[test]
    fn patch_merge_value_wins() {
        let p1 = TestPatch {
            value: Some(1),
            log_entry: None,
        };
        let p2 = TestPatch {
            value: Some(2),
            log_entry: None,
        };
        let merged = p1.merge(p2);
        assert_eq!(merged.value, Some(2));
    }

    #[test]
    fn patch_merge_fallback() {
        let p1 = TestPatch {
            value: Some(1),
            log_entry: None,
        };
        let p2 = TestPatch {
            value: None,
            log_entry: None,
        };
        let merged = p1.merge(p2);
        assert_eq!(merged.value, Some(1));
    }

    #[test]
    fn last_value_reducer() {
        let mut state = TestState {
            value: 0,
            log: vec![],
        };
        let reducer = LastValueReducer;
        let patch = TestPatch {
            value: Some(99),
            log_entry: None,
        };
        reducer.reduce(&mut state, patch).unwrap();
        assert_eq!(state.value, 99);
    }

    #[test]
    fn fn_reducer_custom_logic() {
        let mut state = TestState {
            value: 10,
            log: vec![],
        };
        let reducer = FnReducer::new(|state: &mut TestState, patch: TestPatch| {
            if let Some(v) = patch.value {
                state.value += v;
            }
            Ok(())
        });
        let patch = TestPatch {
            value: Some(5),
            log_entry: None,
        };
        reducer.reduce(&mut state, patch).unwrap();
        assert_eq!(state.value, 15);
    }
}
