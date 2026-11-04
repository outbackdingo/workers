//! Live model discovery per engine: `GET /v1/models`, with the context window
//! read from the field each server actually publishes.
//!
//! This module exists because of issue #2201. `provider-llamacpp` reads a
//! window from `.llama.cpp`'s `/props` (or an Unsloth-style `context_length`)
//! and falls back to a 4096 default otherwise. vLLM and SGLang have no
//! `/props` and declare the window per row as `max_model_len`; pointing that
//! provider at either one produced a catalog entry claiming 4096 tokens for a
//! 262k model, so every turn died with `context/overflow` before the model saw
//! a byte. The fix is not a bigger guess - it is reading the right field, and
//! being loud when no field is there at all.
//!
//! Per engine ([`WindowSource`]):
//!
//! | engine | window field on the row |
//! |---|---|
//! | vLLM | `max_model_len`, else `context_length` |
//! | SGLang | `max_model_len`, else `context_length` |
//! | PrismML | `context_length`, else `max_model_len` |
//! | Mistral.rs | none published; operator override or engine default |
//!
//! Order of precedence for every engine is the same: the operator's
//! `context_window` override, then the server's own per-row declaration, then
//! the engine's default. Whenever the last one is used the model carries a
//! warning on its catalog entry, so a console shows the operator that the
//! number is this worker's assumption rather than the server's fact.
use crate::config::{credential_parts, DEFAULT_MAX_TOKENS};
use crate::settings::{self, ConfigCell};
use crate::{router_client, state, Engine, WindowSource};
use futures::future::BoxFuture;
use iii_sdk::errors::Error;
use iii_sdk::IIIClient;
use llm_router::provider_scaffold::sse_transport::error_chain;
use llm_router::types::model::Model;
use llm_router::types::router::{RefreshModelsRequest, RefreshModelsResponse};
use serde_json::Value;

/// Derive the models endpoint from the configured completions endpoint
/// (`.../v1/chat/completions` -> `.../v1/models`).
pub fn models_url(api_url: &str) -> String {
    match api_url.strip_suffix("/chat/completions") {
        Some(base) => format!("{base}/models"),
        None => format!("{}/models", api_url.trim_end_matches('/')),
    }
}

/// The per-row context window `value` under the field name this engine
/// reads first, falling back to the other standard spelling.
fn declared_window(engine: &Engine, row: &Value) -> Option<u64> {
    let read = |key: &str| row.get(key).and_then(Value::as_u64).filter(|&n| n > 0);
    let primary = match engine.window_source {
        WindowSource::MaxModelLen | WindowSource::ContextLength => {
            // Both OpenAI-compatible spellings are honoured whichever one the
            // engine prefers first: reading only one is the bug in #2201.
            let (first, second) = match engine.window_source {
                WindowSource::MaxModelLen => ("max_model_len", "context_length"),
                _ => ("context_length", "max_model_len"),
            };
            read(first).or_else(|| read(second))
        }
        // mistral.rs publishes ids only.
        WindowSource::Configured => None,
    };
    primary
}

/// One model with its window resolved, plus whether that window is the
/// worker's assumption.
struct Resolved {
    id: String,
    context_window: u64,
    assumed: bool,
}

/// The window, the assumption flag, and why. `override_window` is the
/// operator's `context_window`; when set it always wins and is never marked
/// as an assumption - an operator who typed a number knows what it is.
fn resolve_window(engine: &Engine, row: &Value, override_window: Option<u64>) -> (u64, bool) {
    if let Some(n) = override_window {
        return (n, false);
    }
    match declared_window(engine, row) {
        Some(n) => (n, false),
        None => (engine.fallback_context_window, true),
    }
}

/// Turn a `/v1/models` listing into catalog rows for `engine`.
pub fn models_from_listing(
    engine: &Engine,
    listing: &Value,
    override_window: Option<u64>,
    override_output: Option<u64>,
) -> (Vec<Model>, Vec<String>) {
    let Some(rows) = listing.get("data").and_then(Value::as_array) else {
        return (Vec::new(), Vec::new());
    };
    let mut resolved = Vec::new();
    for raw in rows {
        let Some(id) = raw.get("id").and_then(Value::as_str).filter(|s| !s.is_empty()) else {
            continue;
        };
        let (context_window, assumed) = resolve_window(engine, raw, override_window);
        resolved.push(Resolved {
            id: id.to_string(),
            context_window,
            assumed,
        });
    }
    let warnings = resolved
        .iter()
        .filter(|r| r.assumed)
        .map(|r| {
            format!(
                "{name} did not declare a context window for `{id}`; {window} tokens was assumed. \
                 Set engines.{engine_id}.context_window in the provider-local configuration entry if the \
                 server actually serves more (it sets this with {flag}).",
                name = engine.display_name,
                id = r.id,
                window = r.context_window,
                engine_id = engine.id,
                flag = engine.window_flag,
            )
        })
        .collect();
    let models = resolved
        .into_iter()
        .map(|r| enrich(engine, r, override_output))
        .collect();
    (models, warnings)
}

/// Build one catalog row.
///
/// `max_output_tokens` never claims the whole window: usable input budget is
/// `window - output - reserve`, so a claim of the full window makes the last
/// term zero and every turn fails. A quarter of the window, capped at the
/// shared default, keeps small-window servers usable and a 262k server
/// generous.
fn enrich(engine: &Engine, resolved: Resolved, override_output: Option<u64>) -> Model {
    let quarter = (resolved.context_window / 4).max(1);
    Model {
        id: resolved.id,
        provider: engine.id.to_string(),
        display_name: None,
        context_window: resolved.context_window,
        max_output_tokens: override_output.unwrap_or_else(|| quarter.min(DEFAULT_MAX_TOKENS)),
        input_limit: None,
        supports_thinking: None,
        supports_xhigh: None,
        supports_thinking_off: None,
        reasoning_efforts: None,
        // Best-effort: every one of these servers exposes OpenAI-style tools,
        // but whether a loaded model was trained for them is not something
        // `/v1/models` reports. A template that cannot use tools ignores them.
        supports_tools: Some(true),
        supports_vision: None,
        supports_cache: None,
        supports_structured_output: Some(true),
        thinking_budgets: None,
        pricing: None, // self-hosted
        speech: None,
    }
}

/// Parameter count in billions read from the id (`Qwen3.8-27B` -> 27): a
/// number followed by `B`, standing alone between separators. Hand-named
/// aliases carry no size and rank after every sized model.
pub fn params_billions(id: &str) -> Option<f64> {
    let b: Vec<char> = id.chars().collect();
    let mut i = 0;
    while i < b.len() {
        let starts_token = i == 0 || !b[i - 1].is_ascii_alphanumeric();
        if starts_token && b[i].is_ascii_digit() {
            let mut j = i;
            while j < b.len() && (b[j].is_ascii_digit() || b[j] == '.') {
                j += 1;
            }
            let ends_token = j + 1 >= b.len() || !b[j + 1].is_ascii_alphanumeric();
            if j < b.len() && b[j].eq_ignore_ascii_case(&'b') && ends_token {
                if let Ok(n) = b[i..j].iter().collect::<String>().parse::<f64>() {
                    return Some(n);
                }
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    None
}

/// The default-model preference list for this listing: larger models first,
/// ties in the server's own order.
pub fn rank_default_models(listing: &Value) -> Vec<String> {
    let Some(rows) = listing.get("data").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut ranked: Vec<(f64, &str)> = rows
        .iter()
        .filter_map(|row| {
            let id = row.get("id").and_then(Value::as_str).filter(|s| !s.is_empty())?;
            Some((params_billions(id).unwrap_or(-1.0), id))
        })
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    ranked.into_iter().map(|(_, id)| id.to_string()).collect()
}

enum FetchOutcome {
    Ok(Value),
    AuthFailed,
    Transient(String),
}

async fn fetch_listing(
    http: &reqwest::Client,
    url: &str,
    credential: Option<&str>,
) -> FetchOutcome {
    let mut req = http.get(url);
    if let Some(c) = credential {
        req = req.header("authorization", format!("Bearer {c}"));
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            return FetchOutcome::Transient(format!("models fetch failed: {}", error_chain(&e)))
        }
    };
    let status = resp.status().as_u16();
    if status == 401 || status == 403 {
        return FetchOutcome::AuthFailed;
    }
    if !(200..300).contains(&status) {
        return FetchOutcome::Transient(format!("models fetch http {status}"));
    }
    match resp.json::<Value>().await {
        Ok(v) => FetchOutcome::Ok(v),
        Err(e) => FetchOutcome::Transient(format!("models response not json: {e}")),
    }
}

/// The refresh flow for one engine; returns the reconciled slice size.
pub async fn refresh_models(
    iii: &IIIClient,
    http: &reqwest::Client,
    engine: &'static Engine,
    cell: &ConfigCell,
) -> Result<usize, Error> {
    let token = state::load_token(iii, engine.id).await;
    let resolved = router_client::resolve(iii, engine, token.as_deref()).await?;

    let api_url = resolved
        .api_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(engine.default_api_url)
        .to_string();
    let credential_value = resolved
        .credential
        .as_ref()
        .map(|c| credential_parts(c).trim().to_string())
        .filter(|s| !s.is_empty());

    let override_window = settings::context_window_override(cell, engine.id).await;
    let override_output = settings::max_output_override(cell, engine.id).await;

    match fetch_listing(http, &models_url(&api_url), credential_value.as_deref()).await {
        FetchOutcome::Ok(listing) => {
            let (models, warnings) =
                models_from_listing(engine, &listing, override_window, override_output);
            for warning in &warnings {
                eprintln!("[provider-local:{}] {warning}", engine.id);
            }
            // Re-declare the default list before the slice lands so the
            // router never holds a slice without its preference.
            if let Err(e) = crate::register::redeclare_with_defaults(
                iii,
                engine,
                rank_default_models(&listing),
            )
            .await
            {
                eprintln!("[provider-local] default-model redeclare failed ({e})");
            }
            let count = models.len();
            router_client::reconcile(iii, engine, models, token.as_deref()).await?;
            Ok(count)
        }
        FetchOutcome::AuthFailed => {
            // A key is configured on the server and ours is missing/wrong:
            // the models are genuinely unusable, so the slice empties.
            router_client::reconcile(iii, engine, vec![], token.as_deref()).await?;
            Ok(0)
        }
        // Blip: keep the previous slice rather than reporting an empty
        // catalog the operator would read as "my server has no models".
        FetchOutcome::Transient(msg) => Err(Error::Remote {
            code: "provider/upstream_unavailable".into(),
            message: format!(
                "{} at {api_url} did not answer /v1/models: {msg}",
                engine.display_name
            ),
            stacktrace: None,
        }),
    }
}

pub fn make_refresh_models(
    iii: IIIClient,
    http: reqwest::Client,
    engine: &'static Engine,
    cell: ConfigCell,
) -> impl Fn(RefreshModelsRequest) -> BoxFuture<'static, Result<RefreshModelsResponse, Error>>
       + Send
       + Sync
       + 'static {
    move |_req: RefreshModelsRequest| {
        let (iii, http, cell) = (iii.clone(), http.clone(), cell.clone());
        Box::pin(async move {
            let count = refresh_models(&iii, &http, engine, &cell).await?;
            Ok(RefreshModelsResponse { ok: true, count })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn vllm() -> &'static Engine {
        crate::engine("vllm").unwrap()
    }

    #[test]
    fn models_url_derives_from_completions_endpoint() {
        assert_eq!(
            models_url("http://127.0.0.1:8000/v1/chat/completions"),
            "http://127.0.0.1:8000/v1/models"
        );
        assert_eq!(
            models_url("https://box.example/custom"),
            "https://box.example/custom/models"
        );
    }

    #[test]
    fn vllm_and_sglang_read_max_model_len_from_the_row() {
        // The regression in issue #2201: this listing is what SGLang returns
        // for a 262144-token model, and the old provider read none of it.
        let listing = json!({ "object": "list", "data": [
            { "id": "qwen3.8-27b-sglang", "object": "model", "max_model_len": 262_144 }
        ]});
        for id in ["vllm", "sglang"] {
            let engine = crate::engine(id).unwrap();
            let (models, warnings) = models_from_listing(engine, &listing, None, None);
            assert_eq!(models[0].context_window, 262_144, "{id}");
            assert!(warnings.is_empty(), "{id}: nothing was assumed");
            assert_eq!(models[0].provider, id);
        }
    }

    #[test]
    fn a_row_may_declare_the_window_under_either_spelling() {
        let listing = json!({ "data": [
            { "id": "a", "context_length": 128_000 },
            { "id": "b", "max_model_len": 32_768 },
        ]});
        let (models, warnings) = models_from_listing(vllm(), &listing, None, None);
        assert_eq!(models[0].context_window, 128_000);
        assert_eq!(models[1].context_window, 32_768);
        assert!(warnings.is_empty());
    }

    #[test]
    fn mistralrs_uses_the_operator_window_not_an_invented_row_field() {
        // mistral.rs lists ids without a window; a row that happens to carry
        // max_model_len must not be trusted by an engine that does not
        // publish it.
        let listing = json!({ "data": [{ "id": "mistral-small", "max_model_len": 262_144 }] });
        let engine = crate::engine("mistralrs").unwrap();
        let (models, warnings) = models_from_listing(engine, &listing, None, None);
        assert_eq!(models[0].context_window, engine.fallback_context_window);
        assert_eq!(warnings.len(), 1, "an assumed window is reported");
        assert!(warnings[0].contains("mistral-small"));
        assert!(warnings[0].contains("engines.mistralrs.context_window"));

        let (models, warnings) = models_from_listing(engine, &listing, Some(65_536), None);
        assert_eq!(models[0].context_window, 65_536);
        assert!(warnings.is_empty(), "an operator window is not a guess");
    }

    #[test]
    fn the_operator_override_beats_a_server_declaration() {
        // An operator who caps a shared GPU by hand must win even when the
        // server advertises the model's full trained window.
        let listing = json!({ "data": [{ "id": "m", "max_model_len": 262_144 }] });
        let (models, _) = models_from_listing(vllm(), &listing, Some(8192), None);
        assert_eq!(models[0].context_window, 8192);
    }

    #[test]
    fn output_budget_never_claims_the_whole_window() {
        // usable = window - output - reserve; a full-window claim makes the
        // last term zero and every turn dies with context/overflow.
        let listing = json!({ "data": [{ "id": "m", "max_model_len": 262_144 }] });
        let (models, _) = models_from_listing(vllm(), &listing, None, None);
        assert!(models[0].max_output_tokens < models[0].context_window);
        assert_eq!(models[0].max_output_tokens, DEFAULT_MAX_TOKENS);

        let small = json!({ "data": [{ "id": "m", "max_model_len": 2048 }] });
        let (models, _) = models_from_listing(vllm(), &small, None, None);
        assert!(
            models[0].max_output_tokens < models[0].context_window,
            "small windows must still leave input room"
        );
        assert_eq!(models[0].max_output_tokens, 512);
    }

    #[test]
    fn an_operator_output_cap_is_honoured() {
        let listing = json!({ "data": [{ "id": "m", "max_model_len": 262_144 }] });
        let (models, _) = models_from_listing(vllm(), &listing, None, Some(4096));
        assert_eq!(models[0].max_output_tokens, 4096);
    }

    #[test]
    fn a_zero_or_negative_window_on_the_row_is_ignored() {
        let listing = json!({ "data": [
            { "id": "zero", "max_model_len": 0 },
            { "id": "negative", "max_model_len": -1 },
        ]});
        let (models, warnings) = models_from_listing(vllm(), &listing, None, None);
        assert_eq!(models[0].context_window, vllm().fallback_context_window);
        assert_eq!(models[1].context_window, vllm().fallback_context_window);
        assert_eq!(warnings.len(), 2);
    }

    #[test]
    fn rows_without_an_id_are_skipped_and_missing_data_is_empty() {
        let listing = json!({ "data": [ { "object": "model" }, { "id": "" }, { "id": "ok" } ] });
        let (models, _) = models_from_listing(vllm(), &listing, None, None);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "ok");
        assert!(models_from_listing(vllm(), &json!({}), None, None).0.is_empty());
        assert!(models_from_listing(vllm(), &json!({ "data": "nope" }), None, None)
            .0
            .is_empty());
    }

    #[test]
    fn default_preference_is_largest_first_then_listing_order() {
        let listing = json!({ "data": [
            { "id": "small-7B" },
            { "id": "big-27B" },
            { "id": "other-7B" },
            { "id": "unsized" },
        ]});
        assert_eq!(
            rank_default_models(&listing),
            vec!["big-27B", "small-7B", "other-7B", "unsized"]
        );
        assert!(rank_default_models(&json!({ "object": "list" })).is_empty());
    }

    #[test]
    fn parameter_count_comes_from_the_id() {
        assert_eq!(params_billions("Qwen3.8-27B-GGUF:Q8_0"), Some(27.0));
        assert_eq!(params_billions("Llama-3.2-1.5B-Instruct"), Some(1.5));
        assert_eq!(params_billions("gemma-3-270m"), None);
        assert_eq!(params_billions("model-7"), None, "a trailing number is not a size");
        assert_eq!(params_billions("Q8_0-8bit"), None, "8bit is not a size");
    }
}
