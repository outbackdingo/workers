//! Registry-publish manifest emitted by `provider-local --manifest`
//! (binary-worker.md section manifest; same shape as the other providers).
use serde::Serialize;

const DESCRIPTION: &str = "Local OpenAI-compatible inference servers (vLLM, SGLang, \
     PrismML, mistral.rs) as llm-router providers, each declaring its real context window.";

#[derive(Serialize)]
pub struct ModuleManifest {
    pub name: String,
    pub version: String,
    pub description: String,
    pub default_config: serde_json::Value,
    pub supported_targets: Vec<String>,
}

/// Build the manifest for the currently-compiled binary. `default_config`
/// carries the seeded entry so a fresh install registers with every engine
/// enabled: operator configuration lives in this worker's `provider-local`
/// entry, and per-server settings in the router's slices.
pub fn build_manifest() -> ModuleManifest {
    ModuleManifest {
        name: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: DESCRIPTION.to_string(),
        default_config: crate::settings::initial_value(),
        supported_targets: vec![env!("TARGET").to_string()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_names_all_four_engines() {
        let m = build_manifest();
        let json = serde_json::to_string_pretty(&m).expect("serialize manifest");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["name"], "provider-local");
        assert!(!parsed["version"].as_str().unwrap().is_empty());
        assert!(parsed["description"].as_str().unwrap().contains("mistral.rs"));
        assert!(!parsed["supported_targets"].as_array().unwrap().is_empty());
        for engine in crate::ENGINES.iter() {
            assert!(
                parsed["default_config"]["engines"].get(engine.id).is_some(),
                "{} missing from the seeded default_config",
                engine.id
            );
        }
    }
}
