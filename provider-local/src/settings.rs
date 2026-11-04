//! The `provider-local` configuration entry: which engines this worker
//! serves, and what window to use for a server that does not publish one.
//!
//! Deliberately narrow. Everything a server itself owns - `api_url`,
//! `api_key`, `max_tokens` - stays in llm-router's per-provider slices, which
//! `router::provider::resolve` already hands this worker per request. What is
//! left is the one thing the router cannot know: a context window the server
//! never publishes (issue #2201's failure mode, mirrored).
//!
//! The value is re-read on every discovery and on every config update, so an
//! operator edit reaches the catalog without a restart.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const CONFIG_ID: &str = "provider-local";

/// `enabled` is on unless an operator says otherwise.
fn enabled_by_default() -> bool {
    true
}

/// One engine's knobs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    /// Serve this engine at all. A disabled engine registers nothing, so it
    /// never appears in the console's provider list. An omitted field means
    /// enabled: a block that only sets `context_window` must not switch the
    /// engine off by accident.
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
    /// The context window to publish for every model this server lists, when
    /// the server itself does not declare one. Unset means "use what the
    /// server says, else the engine default".
    pub context_window: Option<u64>,
    /// Cap on output tokens for this engine's models. Unset keeps the shared
    /// default (a quarter of the window); the router still applies its own
    /// `output_token_max`.
    pub max_output_tokens: Option<u64>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            enabled: true,
            context_window: None,
            max_output_tokens: None,
        }
    }
}

/// The stored entry.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerConfig {
    /// Per-engine knobs keyed by provider id (`vllm`, `sglang`, `prisml`,
    /// `mistralrs`). An absent key means "enabled with built-in defaults",
    /// so an operator who has never opened the settings still gets a working
    /// worker.
    pub engines: std::collections::BTreeMap<String, EngineConfig>,
}

impl WorkerConfig {
    /// The effective knobs for `id`: the operator's block, or the built-in
    /// default (enabled) when the entry does not name it.
    pub fn engine(&self, id: &str) -> EngineConfig {
        self.engines.get(id).cloned().unwrap_or_default()
    }

    /// The `initial_value` seeded into an empty entry: every known engine
    /// present and enabled with no overrides. Seeding the full set keeps the
    /// settings form self-describing, and `configuration::ensure` never
    /// overwrites a value an operator already stored.
    pub fn seeded() -> Self {
        let engines = crate::ENGINES
            .iter()
            .map(|engine| (engine.id.to_string(), EngineConfig::default()))
            .collect();
        WorkerConfig { engines }
    }

    pub fn from_json(value: &Value) -> Result<Self, String> {
        serde_json::from_value(value.clone())
            .map_err(|e| format!("provider-local configuration does not match its schema: {e}"))
    }
}

/// The live snapshot every stream, discovery and count shares.
pub type ConfigCell = Arc<tokio::sync::RwLock<Arc<WorkerConfig>>>;

pub fn new_cell() -> ConfigCell {
    Arc::new(tokio::sync::RwLock::new(Arc::new(WorkerConfig::seeded())))
}

pub async fn snapshot(cell: &ConfigCell) -> Arc<WorkerConfig> {
    cell.read().await.clone()
}

pub async fn apply(cell: &ConfigCell, cfg: WorkerConfig) {
    *cell.write().await = Arc::new(cfg);
}

/// The engine block for `id`, read from the live snapshot.
async fn engine_knobs(cell: &ConfigCell, id: &str) -> EngineConfig {
    snapshot(cell).await.engine(id)
}

/// Whether `id` should be served, read from the live snapshot.
pub async fn enabled(cell: &ConfigCell, id: &str) -> bool {
    engine_knobs(cell, id).await.enabled
}

/// The operator's window override for `id`, if any.
pub async fn context_window_override(cell: &ConfigCell, id: &str) -> Option<u64> {
    engine_knobs(cell, id).await.context_window
}

/// The operator's output cap for `id`, if any.
pub async fn max_output_override(cell: &ConfigCell, id: &str) -> Option<u64> {
    engine_knobs(cell, id).await.max_output_tokens
}

/// The JSON Schema shown by the console's settings form.
pub fn schema() -> Value {
    let mut engine_props = serde_json::Map::new();
    for engine in crate::ENGINES.iter() {
        engine_props.insert(
            engine.id.to_string(),
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "default": true,
                        "description": format!("Serve {} models through llm-router.", engine.display_name),
                    },
                    "context_window": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": format!(
                            "Context window to publish when {} does not declare one (the server sets this with {}). Unset keeps the server's own value, else the engine default.",
                            engine.display_name, engine.window_flag
                        ),
                    },
                    "max_output_tokens": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": format!(
                            "Cap on output tokens for {} models. Unset uses a quarter of the window.",
                            engine.display_name
                        ),
                    },
                },
            }),
        );
    }
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "engines": {
                "type": "object",
                "additionalProperties": false,
                "properties": engine_props,
            }
        }
    })
}

/// The value written into a fresh entry.
pub fn initial_value() -> Value {
    serde_json::to_value(WorkerConfig::seeded()).expect("seeded config serializes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absent_engine_block_is_enabled_with_no_overrides() {
        let cfg = WorkerConfig::default();
        let vllm = cfg.engine("vllm");
        assert!(vllm.enabled);
        assert_eq!(vllm.context_window, None);
        assert_eq!(vllm.max_output_tokens, None);
    }

    #[test]
    fn an_explicit_false_disables_and_an_explicit_window_wins() {
        let cfg = WorkerConfig::from_json(&json!({
            "engines": {
                "sglang": { "enabled": false },
                "vllm": { "context_window": 262144, "max_output_tokens": 8192 }
            }
        }))
        .expect("valid config");
        assert!(!cfg.engine("sglang").enabled);
        let vllm = cfg.engine("vllm");
        assert!(vllm.enabled, "omitted enabled is still on");
        assert_eq!(vllm.context_window, Some(262_144));
        assert_eq!(vllm.max_output_tokens, Some(8192));
    }

    #[test]
    fn a_zero_window_is_rejected_instead_of_starving_every_turn() {
        // A zero usable budget is exactly bug #2201's failure; the schema
        // refuses it so an operator cannot configure it by hand.
        let schema = schema();
        for engine in crate::ENGINES.iter() {
            let window = &schema["properties"]["engines"]["properties"][engine.id]["properties"]
                ["context_window"];
            assert_eq!(window["minimum"], 1, "{}", engine.id);
        }
    }

    #[test]
    fn seeded_value_names_every_engine_the_schema_advertises() {
        let seeded = WorkerConfig::seeded();
        for engine in crate::ENGINES.iter() {
            assert!(seeded.engines.contains_key(engine.id), "missing {}", engine.id);
            assert!(seeded.engines[engine.id].enabled, "{}", engine.id);
        }
        let value = initial_value();
        assert!(value["engines"]["mistralrs"]["enabled"].as_bool().unwrap());
        let schema = schema();
        let props = schema["properties"]["engines"]["properties"]
            .as_object()
            .unwrap();
        assert_eq!(props.len(), seeded.engines.len());
    }

    #[test]
    fn unknown_fields_are_rejected_rather_than_silently_ignored() {
        let err = WorkerConfig::from_json(&json!({ "engine": {} })).unwrap_err();
        assert!(err.contains("does not match its schema"), "{err}");
        let err = WorkerConfig::from_json(&json!({
            "engines": { "vllm": { "context_size": 8192 } }
        }))
        .unwrap_err();
        assert!(err.contains("does not match its schema"), "{err}");
    }
}
