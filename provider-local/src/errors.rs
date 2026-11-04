//! Upstream failure -> shared ErrorKind taxonomy (provider protocol rule 5:
//! the providers MUST NOT invent one taxonomy each).
//!
//! The envelopes here are the real ones these servers return, and they differ:
//!
//! - **vLLM** answers 400 with `{"object":"error","message":"This model's
//!   maximum context length is 262144 tokens. However, you requested 295444
//!   tokens ...","type":"BadRequestError"}`.
//! - **SGLang** answers 400 with the same sentence in a flatter envelope, and
//!   its gRPC-fronted variant returns `{"code":"invalid-argument","error":"..."}`
//!   with `error` as a bare string.
//! - **mistral.rs** and **PrismML** emit OpenAI-shaped `{"error":{"message",
//!   "type","code"}}` envelopes.
//!
//! A context-overflow classification is the one that matters most: it is what
//! tells the harness to compact and retry instead of failing the turn, so a
//! missed match is a dead session.
use iii_sdk::errors::Error;
use llm_router::types::events::ErrorKind;
use serde_json::Value;

/// Map an HTTP status + error body to the shared taxonomy.
/// `None` status = the request never got a response (connect/read failure).
pub fn classify(status: Option<u16>, message: &str) -> ErrorKind {
    if let Ok(v) = serde_json::from_str::<Value>(message) {
        if let Some(kind) = classify_value(&v, status) {
            return kind;
        }
    }
    match status {
        Some(401) | Some(403) => ErrorKind::AuthExpired,
        Some(429) => ErrorKind::RateLimited,
        Some(413) => ErrorKind::ContextOverflow,
        Some(s) if s >= 500 => ErrorKind::Transient,
        Some(_) if is_context_overflow_message(message) => ErrorKind::ContextOverflow,
        Some(_) => ErrorKind::Permanent,
        None => ErrorKind::Transient,
    }
}

/// Map router bus errors surfaced through `router::provider::resolve`.
pub fn classify_bus_error(err: &Error) -> ErrorKind {
    match err {
        Error::Remote { code, .. } if code == "router/registration_rejected" => {
            ErrorKind::Permanent
        }
        _ => ErrorKind::Transient,
    }
}

/// Both envelopes, plus the bare-string `error` SGLang's gRPC frontend uses.
fn classify_value(v: &Value, status: Option<u16>) -> Option<ErrorKind> {
    // Envelope 1: `error` is a bare string message - sniff it directly.
    if let Some(text) = v.get("error").and_then(Value::as_str) {
        return is_context_overflow_message(text).then_some(ErrorKind::ContextOverflow);
    }
    // Envelope 2: OpenAI-style `{ "error": { "message", "type", "code" } }`
    // (mistral.rs, PrismML, and vLLM's newer frontends).
    if let Some(err) = v.get("error").filter(|e| e.is_object()) {
        let code = err.get("code").and_then(Value::as_str).unwrap_or("");
        let err_type = err.get("type").and_then(Value::as_str).unwrap_or("");
        let msg = err.get("message").and_then(Value::as_str).unwrap_or("");
        match code {
            "context_length_exceeded" => return Some(ErrorKind::ContextOverflow),
            // 429 with insufficient_quota is a billing wall, not a rate limit.
            "insufficient_quota" => return Some(ErrorKind::Permanent),
            "invalid_api_key" | "account_deactivated" => return Some(ErrorKind::AuthExpired),
            _ => {}
        }
        match err_type {
            "authentication_error" | "permission_error" => return Some(ErrorKind::AuthExpired),
            "rate_limit_error" => return Some(ErrorKind::RateLimited),
            "server_error" => return Some(ErrorKind::Transient),
            "invalid_request_error" => {
                return Some(if status == Some(413) || is_context_overflow_message(msg) {
                    ErrorKind::ContextOverflow
                } else {
                    ErrorKind::Permanent
                })
            }
            _ => {}
        }
        return None;
    }
    // Envelope 3: vLLM's flat `{ "object": "error", "message": "..." }`.
    if let Some(msg) = v.get("message").and_then(Value::as_str) {
        if is_context_overflow_message(msg) {
            return Some(ErrorKind::ContextOverflow);
        }
        // vLLM names the class in `type` (`BadRequestError`); the OpenAI
        // spellings are accepted too in case a proxy rewrote it.
        match v.get("type").and_then(Value::as_str).unwrap_or("") {
            "authentication_error" => return Some(ErrorKind::AuthExpired),
            "rate_limit_error" => return Some(ErrorKind::RateLimited),
            "server_error" | "InternalServerError" => return Some(ErrorKind::Transient),
            _ => {}
        }
    }
    // Envelope 4: `{ "code": "invalid-argument", "error": "..." }` - the
    // `error` key was already handled above only when it was a string, and it
    // is here, so this arm covers a message carried under `detail` instead.
    if let Some(detail) = v.get("detail").and_then(Value::as_str) {
        if is_context_overflow_message(detail) {
            return Some(ErrorKind::ContextOverflow);
        }
    }
    None
}

/// The sentences these servers use for a prompt over the served window. The
/// first four are vLLM's and SGLang's; the rest come from wrappers and
/// gateways in front of them.
fn is_context_overflow_message(message: &str) -> bool {
    let m = message.to_lowercase();
    m.contains("maximum context length")
        || m.contains("context length")
        || m.contains("maximum context")
        || m.contains("too many tokens")
        || m.contains("exceeds context")
        || m.contains("context window")
        || m.contains("maximum prompt length")
        || m.contains("prompt is too long")
        || m.contains("input is too long")
        || m.contains("longer than the maximum")
}

/// Invalid handler input surfaced on the bus in the `{ code, message }`
/// convention (same shape RouterError uses on the router side).
pub fn invalid_request(message: impl Into<String>) -> Error {
    Error::Remote {
        code: "provider/invalid_request".to_string(),
        message: message.into(),
        stacktrace: None,
    }
}

/// Map a serde deserialization failure to the provider's stable
/// `invalid_request` wire error. The shared typed registration adapter keeps
/// schemas emitted while preserving `provider/invalid_request`.
pub fn invalid_request_from_serde(e: serde_json::Error) -> Error {
    invalid_request(format!("bad ProviderStreamInput: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_map_to_the_shared_taxonomy() {
        assert_eq!(classify(Some(401), ""), ErrorKind::AuthExpired);
        assert_eq!(classify(Some(403), ""), ErrorKind::AuthExpired);
        assert_eq!(classify(Some(429), ""), ErrorKind::RateLimited);
        assert_eq!(classify(Some(413), ""), ErrorKind::ContextOverflow);
        assert_eq!(classify(Some(500), ""), ErrorKind::Transient);
        assert_eq!(classify(Some(400), "bad request"), ErrorKind::Permanent);
        assert_eq!(classify(None, "connect refused"), ErrorKind::Transient);
    }

    #[test]
    fn vllm_flat_error_envelope_overflow_is_context_overflow() {
        // The exact vLLM body reported against a 262144-token model.
        let body = r#"{"object":"error","message":"This model's maximum context length is 262144 tokens. However, you requested 295444 tokens (295444 in the messages). Please reduce the length of the messages.","type":"BadRequestError","param":null,"code":400}"#;
        assert_eq!(classify(Some(400), body), ErrorKind::ContextOverflow);
    }

    #[test]
    fn vllm_flat_error_5xx_is_transient_and_auth_is_auth() {
        let body = r#"{"object":"error","message":"upstream died","type":"InternalServerError"}"#;
        assert_eq!(classify(Some(500), body), ErrorKind::Transient);
        let body = r#"{"object":"error","message":"no key","type":"authentication_error"}"#;
        assert_eq!(classify(Some(401), body), ErrorKind::AuthExpired);
    }

    #[test]
    fn sglang_bare_string_error_overflow_is_context_overflow() {
        // SGLang's gRPC frontend: top-level code, `error` as a bare string.
        let body = r#"{"code":"invalid-argument","error":"This model's maximum prompt length is 262144 but the request contains 295444 tokens."}"#;
        assert_eq!(classify(Some(400), body), ErrorKind::ContextOverflow);
    }

    #[test]
    fn openai_envelope_codes_are_honored() {
        let body = r#"{"error":{"message":"This model's maximum context length is 400000 tokens.","type":"invalid_request_error","code":"context_length_exceeded"}}"#;
        assert_eq!(classify(Some(400), body), ErrorKind::ContextOverflow);
        let body = r#"{"error":{"message":"You exceeded your current quota.","type":"insufficient_quota","code":"insufficient_quota"}}"#;
        assert_eq!(classify(Some(429), body), ErrorKind::Permanent);
        let body = r#"{"error":{"message":"Incorrect API key provided.","type":"invalid_request_error","code":"invalid_api_key"}}"#;
        assert_eq!(classify(Some(401), body), ErrorKind::AuthExpired);
    }

    #[test]
    fn overflow_phrasing_does_not_false_positive_on_tool_validation() {
        // A 400 whose text merely mentions "context" must stay permanent, or
        // the harness compacts forever against a broken tool call.
        assert_eq!(
            classify(
                Some(400),
                r#"tool_call_id "ctx-1" not found in context"#
            ),
            ErrorKind::Permanent
        );
        // 5xx wins over message sniffing.
        assert_eq!(classify(Some(500), "context blah"), ErrorKind::Transient);
    }

    #[test]
    fn registration_rejected_is_permanent_on_the_bus() {
        let err = Error::Remote {
            code: "router/registration_rejected".into(),
            message: "bad token".into(),
            stacktrace: None,
        };
        assert_eq!(classify_bus_error(&err), ErrorKind::Permanent);
        let err = Error::Remote {
            code: "engine/timeout".into(),
            message: "t".into(),
            stacktrace: None,
        };
        assert_eq!(classify_bus_error(&err), ErrorKind::Transient);
    }

    #[test]
    fn bus_error_codes_are_worker_prefixed() {
        match invalid_request("x") {
            Error::Remote { code, .. } => assert_eq!(code, "provider/invalid_request"),
            other => panic!("want Remote, got {other:?}"),
        }
    }
}
