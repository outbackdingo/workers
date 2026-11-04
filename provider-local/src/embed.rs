//! `provider::<id>::embed` - batch text embeddings against one engine's
//! `/v1/embeddings`.
//!
//! vLLM, SGLang and PrismML all serve the OpenAI embeddings route when the
//! server was started with an embedding-capable model (`--task embed` on
//! vLLM, `--is-embedding` on SGLang); mistral.rs serves it for embedding
//! models. A server started without one answers 404, which is reported as
//! such rather than as an internal failure - the operator's fix is a flag,
//! not a bug.
//!
//! Consumers (llm-router's `router::embed`, and through it the memory
//! worker's semantic recall) treat this as the provider protocol's embedding
//! surface. Fully local: no cloud, no credential unless the server requires a
//! key.
use iii_sdk::errors::Error;
use iii_sdk::IIIClient;
use llm_router::provider_scaffold::cache::ScaffoldCache;
use llm_router::types::events::ErrorKind;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::config::{config_from_resolve, ConfigError};
use crate::errors::{classify, classify_bus_error};
use crate::state;
use crate::Engine;

/// Embeddings requests are bounded and non-streaming; a tight budget keeps
/// hook callers (memory recall) inside their own timeouts.
const EMBED_TIMEOUT_SECS: u64 = 20;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EmbedRequest {
    /// Model id to embed with; omitted means the server's loaded model, which
    /// the reply then names.
    #[serde(default)]
    pub model: Option<String>,
    /// Texts to embed, one vector returned per input, order preserved.
    #[schemars(length(min = 1, max = 512))]
    pub input: Vec<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct EmbedResponse {
    pub model: String,
    /// One embedding per input, in input order.
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct WireEmbedding {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct WireResponse {
    data: Vec<WireEmbedding>,
    #[serde(default)]
    model: Option<String>,
}

/// The `/embeddings` body. No `model` unless the caller named one: these
/// servers echo a sent `model` back verbatim, so a placeholder would make
/// every reply claim that placeholder whatever model is loaded.
fn request_body(model: Option<&str>, input: &[String]) -> serde_json::Value {
    let mut body = serde_json::json!({ "input": input });
    if let Some(model) = model {
        body["model"] = serde_json::json!(model);
    }
    body
}

/// The embeddings endpoint for the engine's chat `api_url`: the sibling
/// `/embeddings` on the same host and version prefix, so a nonstandard port
/// or path prefix keeps working.
fn embed_url(chat_api_url: &str, engine: &Engine) -> String {
    let Ok(mut url) = reqwest::Url::parse(chat_api_url) else {
        return engine.default_api_url.replace("/chat/completions", "/embeddings");
    };
    let path = url.path().trim_end_matches('/').to_string();
    let new_path = if let Some(base) = path.strip_suffix("/chat/completions") {
        format!("{base}/embeddings")
    } else {
        format!("{path}/embeddings")
    };
    url.set_path(&new_path);
    url.set_query(None);
    url.to_string()
}

pub async fn handle(
    iii: &IIIClient,
    http: &reqwest::Client,
    cache: &ScaffoldCache,
    engine: &'static Engine,
    req: EmbedRequest,
) -> Result<EmbedResponse, Error> {
    if req.input.is_empty() || req.input.len() > 512 {
        return Err(Error::Handler(
            "invalid_input: input must contain 1..=512 texts".into(),
        ));
    }
    let requested = req.model.filter(|m| !m.trim().is_empty());
    let model = requested.clone().unwrap_or_else(|| "default".to_string());

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
    let cfg = config_from_resolve(engine, &model, None, &resolved).map_err(|e| match e {
        ConfigError::InvalidApiUrl(_) => Error::Handler(format!(
            "provider/upstream_unavailable: {e}"
        )),
    })?;

    let url = embed_url(&cfg.api_url, engine);
    let mut request = http
        .post(&url)
        .timeout(std::time::Duration::from_secs(EMBED_TIMEOUT_SECS));
    if let Some(credential) = &cfg.credential_value {
        request = request.header("authorization", format!("Bearer {credential}"));
    }
    let response = request
        .json(&request_body(requested.as_deref(), &req.input))
        .send()
        .await
        .map_err(|e| Error::Handler(format!("provider/upstream: {e}")))?;

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        let kind = classify(Some(status.as_u16()), &body);
        let detail = if body.is_empty() {
            format!("{} http {status}", engine.display_name)
        } else {
            body
        };
        return Err(match kind {
            ErrorKind::AuthExpired => {
                cache.invalidate();
                Error::Handler(format!(
                    "provider/auth_error: {} rejected the embedding request: {detail}",
                    engine.display_name
                ))
            }
            _ if status.as_u16() == 404 => Error::Handler(format!(
                "provider/no_embeddings: {url} answered 404 - start {} with an embedding-capable \
                 model to serve /v1/embeddings",
                engine.display_name
            )),
            _ => Error::Handler(format!("provider/upstream: {detail}")),
        });
    }

    let parsed: WireResponse = serde_json::from_str(&body).map_err(|e| {
        Error::Handler(format!(
            "provider/upstream: embeddings response from {url} is not the OpenAI shape: {e}"
        ))
    })?;
    let mut ordered: Vec<(usize, Vec<f32>)> = parsed
        .data
        .into_iter()
        .map(|e| (e.index, e.embedding))
        .collect();
    // The wire promises one vector per input in input order, but these are
    // four independent implementations: sorting by the index they sent is
    // what keeps a caller's vectors aligned with its texts.
    ordered.sort_by_key(|(index, _)| *index);
    Ok(EmbedResponse {
        model: parsed.model.unwrap_or(model),
        embeddings: ordered.into_iter().map(|(_, v)| v).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_url_replaces_the_chat_route_on_the_same_prefix() {
        let engine = crate::engine("vllm").unwrap();
        assert_eq!(
            embed_url("http://127.0.0.1:8000/v1/chat/completions", engine),
            "http://127.0.0.1:8000/v1/embeddings"
        );
        // A reverse proxy with a path prefix keeps it.
        assert_eq!(
            embed_url("https://box.example/gpu/v1/chat/completions", engine),
            "https://box.example/gpu/v1/embeddings"
        );
    }

    #[test]
    fn embed_url_falls_back_to_the_engines_own_default() {
        let engine = crate::engine("sglang").unwrap();
        assert_eq!(
            embed_url("not a url", engine),
            "http://127.0.0.1:30000/v1/embeddings"
        );
    }

    #[test]
    fn the_body_omits_an_unset_model_so_the_server_names_what_it_used() {
        let body = request_body(None, &["a".to_string()]);
        assert_eq!(body["input"][0], "a");
        assert!(
            body.get("model").is_none(),
            "a placeholder model would be echoed back as a lie"
        );
        let body = request_body(Some("bge-m3"), &["a".to_string()]);
        assert_eq!(body["model"], "bge-m3");
    }
}
