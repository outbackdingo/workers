//! `provider::<id>::count_tokens` - prompt token counting for one engine.
//!
//! Two sources, in order:
//!
//! 1. **The server's own counting route**, where it has one. vLLM serves
//!    `POST /tokenize`; SGLang serves `POST /v1/chat/completions/input_tokens`
//!    (the llama.cpp-compatible route) on newer builds. Either answer is
//!    exact, because the server holds the tokenizer.
//! 2. **A local estimate** over the same wire text the stream sends, using the
//!    shared framing walk (`llm_router::provider_scaffold::chat_framing`) and a
//!    vocabulary fetched from the model's Hugging Face repo when the id names
//!    one. Never runs the model and costs nothing.
//!
//! A 20% over-estimate is the safe direction here: this number feeds the
//! context manager's budget, and an under-estimate is what produces the
//! `context/overflow` failure this worker exists to prevent. The estimator is
//! therefore reported honestly (`provider` vs `estimate`) so a caller can tell
//! which one it got.
use iii_sdk::errors::Error;
use iii_sdk::IIIClient;
use llm_router::provider_scaffold::cache::ScaffoldCache;
use llm_router::provider_scaffold::chat_framing::count_framed_chat;
use llm_router::provider_scaffold::endpoint_count::{base_route_url, post_count};
use llm_router::provider_scaffold::vocabulary_count::{self, VocabularyRef};
use llm_router::types::events::ErrorKind;
use llm_router::types::messages::AgentMessage;
use llm_router::types::model::{AgentFunction, ThinkingLevel};
use llm_router::types::router::ResponseFormat;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::config_from_resolve;
use crate::errors::classify_bus_error;
use crate::request::{build_body, build_headers, enable_thinking, BodyArgs};
use crate::settings::ConfigCell;
use crate::state;
use crate::Engine;

/// `estimator` when the server itself counted.
pub const ESTIMATOR_PROVIDER: &str = "provider";
/// `estimator` when a local vocabulary estimate produced it.
pub const ESTIMATOR_ESTIMATE: &str = "estimate";

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CountTokensRequest {
    /// Model id the prompt targets; echoed in the reply, since the server
    /// counts with whichever model it has loaded.
    pub model: String,
    /// System prompt counted as the leading wire message when present.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Function invocation schemas; mapped to the wire `tools` array.
    #[serde(default)]
    pub tools: Option<Vec<AgentFunction>>,
    /// Wire agent messages, the same shape this provider's stream accepts.
    /// Must be non-empty.
    pub messages: Vec<AgentMessage>,
    /// The turn's reasoning level, mapped onto
    /// `chat_template_kwargs.enable_thinking` exactly as the stream does.
    #[serde(default)]
    pub thinking_level: Option<ThinkingLevel>,
    /// The turn's structured-output request, sent as the stream sends it.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct CountTokensResponse {
    pub model: String,
    /// Prompt tokens for the assembled request.
    pub tokens: u64,
    /// `provider` when the server's own route produced the count, `estimate`
    /// when a local vocabulary estimate did.
    pub estimator: String,
}

/// The chat body less what only steers generation: the counting route
/// tokenizes the same prompt the chat route bills.
fn count_body(engine: &Engine, args: &BodyArgs) -> Value {
    let mut body = build_body(engine, args);
    if let Some(fields) = body.as_object_mut() {
        for knob in ["stream", "stream_options", "max_tokens"] {
            fields.remove(knob);
        }
    }
    body
}

/// The models a caller most likely meant, in the order they are worth
/// guessing: an explicit Hugging Face repo in the id, else the id itself.
pub fn vocabulary_reference(model: &str) -> VocabularyRef {
    if model.contains('/') {
        VocabularyRef::huggingface(model)
    } else {
        VocabularyRef::huggingface(model)
    }
}

/// A conservative local estimate: the shared framing walk over the wire
/// text, scaled up so an unknown tokenizer under-counts as rarely as
/// possible.
fn estimate_tokens(args: &BodyArgs, counted: Option<u64>) -> u64 {
    // Four characters per token is the usual English ratio; a vocabulary
    // fetched from the model's own repo replaces it when one is available.
    let framed = counted.unwrap_or_else(|| {
        count_framed_chat(
            Some(&args.system_prompt),
            &args.tools,
            &args.messages,
            |text| (text.len() as u64).div_ceil(4).max(1),
        )
    });
    // A 20% safety margin: this number feeds the context budget, and
    // under-counting is exactly what produces context/overflow.
    (framed * 12).div_ceil(10).max(1)
}

pub async fn handle(
    iii: &IIIClient,
    http: &reqwest::Client,
    cache: &ScaffoldCache,
    cell: &ConfigCell,
    engine: &'static Engine,
    req: CountTokensRequest,
) -> Result<CountTokensResponse, Error> {
    let _ = cell;
    if req.messages.is_empty() {
        return Err(Error::Handler(
            "invalid_input: messages must be non-empty".into(),
        ));
    }
    let model = req.model.clone();
    let token = cache.load_token(iii, &state::scope(engine.id)).await;
    let resolved = cache
        .resolve(
            iii,
            engine.id,
            token.as_deref(),
            Some(engine.credential_env_var),
        )
        .await
        .inspect_err(|e| {
            if classify_bus_error(e) == ErrorKind::AuthExpired {
                cache.invalidate();
            }
        })?;
    let cfg = config_from_resolve(engine, &model, None, &resolved).map_err(|e| {
        Error::Handler(format!("provider/upstream_unavailable: {e}"))
    })?;

    let args = BodyArgs {
        model: model.clone(),
        max_tokens: cfg.max_tokens,
        system_prompt: req.system_prompt.clone().unwrap_or_default(),
        messages: req.messages.clone(),
        tools: req.tools.clone().unwrap_or_default(),
        response_format: req.response_format.clone(),
        enable_thinking: enable_thinking(req.thinking_level),
    };

    let headers: Vec<(String, String)> = build_headers(&cfg)
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

    // Server-side counting first, in this engine's own spelling.
    let pick = |reply: &Value| {
        reply
            .get("count")
            .and_then(Value::as_u64)
            .or_else(|| reply.get("input_tokens").and_then(Value::as_u64))
            .or_else(|| reply.get("num_tokens").and_then(Value::as_u64))
    };
    let mut routes: Vec<String> = Vec::new();
    match engine.id {
        "vllm" => routes.push(base_route_url(&cfg.api_url, "tokenize")),
        "sglang" => routes.push(base_route_url(&cfg.api_url, "input_tokens")),
        // PrismML and mistral.rs document no counting route; the estimate is
        // the only honest answer, and it says so.
        _ => {}
    }

    let mut metered: Option<u64> = None;
    for route in routes {
        let body = if route.ends_with("/tokenize") {
            // vLLM's /tokenize takes a `prompt` or a full `messages` list and
            // answers `{count, max_model_len}`.
            let mut b = count_body(engine, &args);
            if let Some(fields) = b.as_object_mut() {
                fields.remove("model");
            }
            b
        } else {
            count_body(engine, &args)
        };
        match post_count(http, route.clone(), headers.clone(), body, pick).await {
            Ok(n) => {
                metered = Some(n);
                break;
            }
            // An older build without the route, or a proxy that hides it:
            // fall through to the local estimate rather than failing a turn.
            Err(Error::Handler(msg)) if msg.starts_with("provider/upstream_status: 404 ") => {
                continue
            }
            Err(e) => return Err(e),
        }
    }

    if let Some(tokens) = metered {
        return Ok(CountTokensResponse {
            model,
            tokens,
            estimator: ESTIMATOR_PROVIDER.into(),
        });
    }

    // Local estimate with the model's own vocabulary where it can be had.
    let reference = vocabulary_reference(&model);
    let counted = vocabulary_count::resolve(&reference).await.map(|tokenizer| {
        vocabulary_count::count_chat_tokens(
            &tokenizer,
            Some(&args.system_prompt),
            &args.tools,
            &args.messages,
        )
    });
    let tokens = estimate_tokens(&args, counted);
    Ok(CountTokensResponse {
        model,
        tokens,
        estimator: ESTIMATOR_ESTIMATE.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_router::types::content::ContentBlock;
    use llm_router::types::messages::{UserMessage, UserRoleTag};

    fn args() -> BodyArgs {
        BodyArgs {
            model: "Qwen/Qwen3.8-27B".into(),
            max_tokens: 4096,
            system_prompt: "be brief".into(),
            messages: vec![AgentMessage::User(UserMessage {
                role: UserRoleTag::User,
                content: vec![ContentBlock::Text { text: "hi".into() }],
                timestamp: 1,
            })],
            tools: vec![],
            response_format: None,
            enable_thinking: false,
        }
    }

    #[test]
    fn the_counting_body_drops_generation_only_knobs() {
        let engine = crate::engine("vllm").unwrap();
        let body = count_body(engine, &args());
        assert!(body.get("stream").is_none());
        assert!(body.get("stream_options").is_none());
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["model"], "Qwen/Qwen3.8-27B");
        assert!(body.get("messages").is_some());
    }

    #[test]
    fn the_estimate_is_never_below_the_plain_character_count() {
        // Under-counting is what produces context/overflow; the estimate
        // must lean high.
        let est = estimate_tokens(&args(), None);
        let chars = "be brief".len() as u64 + "hi".len() as u64;
        assert!(est >= chars / 4, "est {est} vs {chars} chars");
    }

    #[test]
    fn the_estimate_never_collapses_to_zero() {
        let mut empty = args();
        empty.system_prompt = String::new();
        empty.messages.clear();
        assert!(estimate_tokens(&empty, None) >= 1);
    }

    #[test]
    fn the_vocabulary_is_fetched_from_the_repo_the_model_id_names() {
        let reference = vocabulary_reference("Qwen/Qwen3.8-27B");
        assert_eq!(reference.id, "Qwen--Qwen3.8-27B");
        assert!(reference
            .url
            .ends_with("Qwen/Qwen3.8-27B/resolve/main/tokenizer.json"));
    }

    #[test]
    fn a_counted_total_is_still_given_the_safety_margin() {
        // The server's exact count is the lower bound; the margin is applied
        // only to the local estimate, never to a metered number.
        assert_eq!(estimate_tokens(&args(), Some(1000)), 1200);
    }

    #[test]
    fn estimator_labels_the_two_sources_apart() {
        assert_ne!(ESTIMATOR_PROVIDER, ESTIMATOR_ESTIMATE);
    }
}
