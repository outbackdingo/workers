//! provider-local: one worker, four local inference engines behind
//! llm-router.
//!
//! vLLM, SGLang, PrismML and mistral.rs all speak the OpenAI Chat
//! Completions wire, so streaming, abort, error classification and the wire
//! mappers are shared. What differs is discovery: issue #2201 is the story of
//! a provider that guessed a 4096-token window for an SGLang server with 262k
//! and starved every turn. Each engine here therefore reads the field its
//! server actually publishes and, when nothing is published, says so instead
//! of inventing a small number (see `discovery` and `settings`).
//!
//! Every engine is its own provider id, so `router::chat` picks one by name,
//! the console shows one entry per engine, and each keeps its own catalog
//! slice.

pub mod config;
pub mod count_tokens;
pub mod discovery;
pub mod embed;
pub mod errors;
pub mod manifest;
pub mod register;
pub mod request;
pub mod router_client;
pub mod settings;
pub mod sse;
pub mod state;
pub mod stream_fn;
pub mod surface;
pub mod upstream;
pub mod wire;

/// How a server declares the context window its models are served with.
/// Issue #2201's root cause: a provider that knows only one shape silently
/// falls back to a small default for every other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowSource {
    /// vLLM and SGLang both put `max_model_len` on every `/v1/models` row.
    MaxModelLen,
    /// An OpenAI-compatible row carrying `context_length` (PrismML, and the
    /// Unsloth-style frontends llama.cpp's own provider accepts).
    ContextLength,
    /// mistral.rs: a local server that publishes ids but no window, so the
    /// operator's `context_window` (or the engine default) supplies it.
    Configured,
}

/// One local inference engine this worker can serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Engine {
    /// The provider id: the `provider::<id>::*` prefix, the router config
    /// slice key, and the key under `engines` in this worker's own
    /// configuration entry.
    pub id: &'static str,
    pub display_name: &'static str,
    /// Where the server listens when the operator configured nothing.
    pub default_api_url: &'static str,
    /// Environment variable the router reads for this engine's key.
    pub credential_env_var: &'static str,
    /// The server flag that sets the served window, quoted in the overflow
    /// hint so an operator knows what to change.
    pub window_flag: &'static str,
    pub window_source: WindowSource,
    /// Window used only when neither the server nor the operator declares
    /// one. Each is that engine's documented default context length, never a
    /// one-size-fits-all 4096.
    pub fallback_context_window: u64,
}

impl Engine {
    /// Guidance a console shows under a context-overflow failure on one of
    /// this engine's models. Compacting cannot fix a server started with a
    /// small window, so the hint names the flag and the override.
    pub fn context_overflow_hint(&self) -> String {
        format!(
            "{name} serves each model with the context size it was started with ({flag}) and \
             rejects anything larger, so compacting cannot fix this. Raise it on the server, or \
             set engines.{id}.context_window in the provider-local configuration entry when the \
             server does not publish its window, then restart the provider so it re-discovers.",
            name = self.display_name,
            flag = self.window_flag,
            id = self.id,
        )
    }
}

/// The engines, in registration order.
pub const ENGINES: [Engine; 5] = [
    Engine {
        id: "vllm",
        display_name: "vLLM",
        default_api_url: "http://127.0.0.1:8000/v1/chat/completions",
        credential_env_var: "VLLM_API_KEY",
        window_flag: "--max-model-len",
        window_source: WindowSource::MaxModelLen,
        fallback_context_window: 32_768,
    },
    Engine {
        id: "sglang",
        display_name: "SGLang",
        default_api_url: "http://127.0.0.1:30000/v1/chat/completions",
        credential_env_var: "SGLANG_API_KEY",
        window_flag: "--context-length",
        window_source: WindowSource::MaxModelLen,
        fallback_context_window: 32_768,
    },
    Engine {
        id: "prisml",
        display_name: "PrismML",
        default_api_url: "http://127.0.0.1:8080/v1/chat/completions",
        credential_env_var: "PRISML_API_KEY",
        window_flag: "its context-window flag",
        window_source: WindowSource::ContextLength,
        fallback_context_window: 32_768,
    },
    Engine {
        id: "mistralrs",
        display_name: "mistral.rs",
        default_api_url: "http://127.0.0.1:1234/v1/chat/completions",
        credential_env_var: "MISTRALRS_API_KEY",
        window_flag: "--max-seq-len",
        window_source: WindowSource::Configured,
        fallback_context_window: 32_768,
    },
    // The machine's own llama.cpp-derived server (titan-engine). It publishes
    // `context_length` per row and no /props, so it reads like PrismML rather
    // than like llama-server: never fall back to llama.cpp's 4096.
    Engine {
        id: "local",
        display_name: "Local (titan-engine)",
        default_api_url: "http://127.0.0.1:1234/v1/chat/completions",
        credential_env_var: "LOCAL_API_KEY",
        window_flag: "--ctx-size",
        window_source: WindowSource::ContextLength,
        fallback_context_window: 32_768,
    },
];

/// Look up an engine by provider id; `None` for an id this worker does not
/// serve.
pub fn engine(id: &str) -> Option<&'static Engine> {
    ENGINES.iter().find(|e| e.id == id)
}

/// Millisecond timestamps for AssistantMessage frames.
#[allow(dead_code)]
pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_engine_has_a_distinct_provider_id() {
        let mut ids: Vec<&str> = ENGINES.iter().map(|e| e.id).collect();
        let total = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), total, "provider ids must be unique");
        assert_eq!(engine("vllm").map(|e| e.display_name), Some("vLLM"));
        assert_eq!(engine("nope"), None);
    }

    #[test]
    fn overflow_hint_names_the_flag_and_the_override() {
        let hint = engine("sglang").unwrap().context_overflow_hint();
        assert!(hint.contains("--context-length"), "{hint}");
        assert!(hint.contains("engines.sglang.context_window"), "{hint}");
    }

    #[test]
    fn default_endpoints_are_the_engines_own_documented_ports() {
        assert!(engine("vllm").unwrap().default_api_url.contains(":8000/"));
        assert!(engine("sglang")
            .unwrap()
            .default_api_url
            .contains(":30000/"));
        assert!(engine("mistralrs")
            .unwrap()
            .default_api_url
            .contains(":1234/"));
    }

    #[test]
    fn fallbacks_are_not_the_llamacpp_4096_that_caused_2201() {
        for engine in ENGINES {
            assert!(
                engine.fallback_context_window > 4096,
                "{} falls back to {} tokens",
                engine.id,
                engine.fallback_context_window
            );
        }
    }
}
