//! Effective per-request config: credential + url + max_tokens, for one
//! engine.
//!
//! Precedence for max_tokens: router-resolved effective budget
//! (`ProviderStreamInput.max_output_tokens`) -> the operator's configured
//! `max_tokens` (from resolve) -> the shared default.
use llm_router::types::credential::Credential;
use llm_router::types::router::ProviderResolveResponse;

use crate::Engine;

/// Output cap when neither the router nor the operator names one. Modest on
/// purpose: it is clamped against the model's window by the caller, and a
/// local server's reply rarely exceeds it.
pub const DEFAULT_MAX_TOKENS: u64 = 8192;

#[derive(Debug, Clone)]
pub struct EngineConfigResolved {
    /// `None` when the server was resolved with no credential - the normal
    /// case for a local server started without an API key. Streaming and
    /// discovery both send no `Authorization` header in that case rather
    /// than treating it as a configuration error.
    pub credential_value: Option<String>,
    pub model: String,
    pub max_tokens: u64,
    pub api_url: String,
}

/// Why an effective config could not be built - the caller turns each into a
/// permanent error frame naming the actual problem.
#[derive(Debug, PartialEq, Eq)]
pub enum ConfigError {
    /// `api_url` is set but is not an absolute http(s) URL. Carries the
    /// offending value so the error frame can show it (a reqwest "builder
    /// error" otherwise hides which value was bad).
    InvalidApiUrl(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::InvalidApiUrl(u) => write!(
                f,
                "provider {id} has an invalid endpoint url: {u:?} \
                 (must be an absolute http(s) URL)",
                id = "local"
            ),
        }
    }
}

/// The single Credential -> bearer secret mapping; streaming, discovery and
/// counting must agree on it.
pub fn credential_parts(credential: &Credential) -> &str {
    match credential {
        Credential::ApiKey { key } => key,
        Credential::Oauth { access_token, .. } => access_token,
    }
}

pub fn config_from_resolve(
    engine: &Engine,
    model: &str,
    effective_max_tokens: Option<u64>,
    resolved: &ProviderResolveResponse,
) -> Result<EngineConfigResolved, ConfigError> {
    // Trim: a credential pasted with a trailing newline makes an invalid
    // `Authorization` header value, and a stray space breaks URL parsing -
    // both surface only as an opaque reqwest "builder error" at send time.
    // Blank-after-trim is treated the same as "no credential configured",
    // not an error: local servers commonly run without a key.
    let credential_value = resolved
        .credential
        .as_ref()
        .map(|c| credential_parts(c).trim().to_string())
        .filter(|s| !s.is_empty());
    let api_url = match resolved.api_url.as_deref().map(str::trim) {
        Some(u) if !u.is_empty() => u.to_string(),
        _ => engine.default_api_url.to_string(),
    };
    // Reject anything reqwest can't build a request from, with a clear message.
    match reqwest::Url::parse(&api_url) {
        Ok(u) if matches!(u.scheme(), "http" | "https") => {}
        _ => return Err(ConfigError::InvalidApiUrl(api_url)),
    }
    Ok(EngineConfigResolved {
        credential_value,
        model: model.to_string(),
        max_tokens: effective_max_tokens
            .or(resolved.max_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS),
        api_url,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_router::types::router::CredentialSource;

    fn engine() -> &'static Engine {
        crate::engine("vllm").unwrap()
    }

    fn resolved(
        credential: Option<Credential>,
        max_tokens: Option<u64>,
    ) -> ProviderResolveResponse {
        ProviderResolveResponse {
            configured: credential.is_some(),
            source: CredentialSource::Config,
            credential,
            api_url: None,
            max_tokens,
        }
    }

    fn resolved_with_url(api_url: Option<&str>) -> ProviderResolveResponse {
        ProviderResolveResponse {
            api_url: api_url.map(str::to_string),
            ..resolved(None, None)
        }
    }

    #[test]
    fn missing_credential_resolves_with_no_auth_header_needed() {
        let cfg = config_from_resolve(engine(), "m", None, &resolved(None, None)).unwrap();
        assert_eq!(cfg.credential_value, None);
        assert_eq!(cfg.api_url, engine().default_api_url);
    }

    #[test]
    fn each_engine_falls_back_to_its_own_endpoint() {
        for engine in crate::ENGINES.iter() {
            let cfg = config_from_resolve(engine, "m", None, &resolved(None, None)).unwrap();
            assert_eq!(cfg.api_url, engine.default_api_url, "{}", engine.id);
        }
    }

    #[test]
    fn credential_is_trimmed_and_whitespace_only_is_absent() {
        let cred = Some(Credential::ApiKey {
            key: " sk-abc\n".into(),
        });
        let cfg = config_from_resolve(engine(), "m", None, &resolved(cred, None)).unwrap();
        assert_eq!(cfg.credential_value.as_deref(), Some("sk-abc"));

        let cred = Some(Credential::ApiKey { key: "  \n".into() });
        let cfg = config_from_resolve(engine(), "m", None, &resolved(cred, None)).unwrap();
        assert_eq!(cfg.credential_value, None);
    }

    #[test]
    fn api_url_is_trimmed_and_non_http_is_rejected_up_front() {
        let cfg = config_from_resolve(
            engine(),
            "m",
            None,
            &resolved_with_url(Some(" http://h/v1 ")),
        )
        .unwrap();
        assert_eq!(cfg.api_url, "http://h/v1");

        let err = config_from_resolve(engine(), "m", None, &resolved_with_url(Some("localhost:1")))
            .unwrap_err();
        assert_eq!(err, ConfigError::InvalidApiUrl("localhost:1".into()));

        let cfg = config_from_resolve(engine(), "m", None, &resolved_with_url(Some("   "))).unwrap();
        assert_eq!(cfg.api_url, engine().default_api_url);
    }

    #[test]
    fn max_tokens_precedence_effective_then_configured_then_default() {
        let cfg = config_from_resolve(engine(), "m", Some(1000), &resolved(None, Some(2000)))
            .unwrap();
        assert_eq!(cfg.max_tokens, 1000);
        let cfg = config_from_resolve(engine(), "m", None, &resolved(None, Some(2000))).unwrap();
        assert_eq!(cfg.max_tokens, 2000);
        let cfg = config_from_resolve(engine(), "m", None, &resolved(None, None)).unwrap();
        assert_eq!(cfg.max_tokens, DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn oauth_credential_yields_its_access_token() {
        let cred = Some(Credential::Oauth {
            access_token: "at".into(),
            refresh_token: None,
            expires_at: None,
            scopes: None,
            provider_extra: None,
        });
        let cfg = config_from_resolve(engine(), "m", None, &resolved(cred, None)).unwrap();
        assert_eq!(cfg.credential_value.as_deref(), Some("at"));
    }
}
