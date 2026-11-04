//! Chat Completions request assembly for one engine: body (messages, tools,
//! response_format) + headers, with the per-engine differences that actually
//! matter.
//!
//! These four servers agree on the OpenAI wire, so the differences are small
//! and deliberate rather than a general extension mechanism:
//!
//! - **Thinking.** vLLM and SGLang both render Qwen-style reasoning models and
//!   gate the thinking channel on the chat-template kwarg `enable_thinking`,
//!   which is the only per-request lever either one exposes. mistral.rs has no
//!   such kwarg and PrismML's template is unknown, so neither receives one:
//!   sending an unknown kwarg is harmless on paper but silently means the
//!   console's thinking toggle does nothing, and it is better to warn.
//! - **max_tokens.** All four accept `max_tokens`; none needs OpenAI's
//!   `max_completion_tokens` replacement.
//! - **usage.** All four report usage natively on the final chunk and tolerate
//!   `stream_options.include_usage`, which a reverse proxy needs to pass it on.
use crate::config::EngineConfigResolved;
use crate::wire::messages::to_wire_messages;
use crate::wire::tools::functions_to_wire;
use crate::Engine;
use llm_router::types::messages::AgentMessage;
use llm_router::types::model::{AgentFunction, ThinkingLevel};
use llm_router::types::router::ResponseFormat;
use serde_json::{json, Value};

pub struct BodyArgs {
    pub model: String,
    pub max_tokens: u64,
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<AgentFunction>,
    pub response_format: Option<ResponseFormat>,
    /// Caller's reasoning preference mapped onto `enable_thinking` for the
    /// engines that render it (see [`supports_enable_thinking`]).
    pub enable_thinking: bool,
}

/// Any requested level but `off` means "think"; absent or `off` means "off".
pub fn enable_thinking(level: Option<ThinkingLevel>) -> bool {
    level.is_some_and(|level| level != ThinkingLevel::Off)
}

/// Whether this engine's chat template reads `enable_thinking`. SGLang and
/// vLLM both serve the Qwen/HuggingFace template family that gates its
/// reasoning channel on exactly this key; mistral.rs's template does not take
/// it and PrismML is not documented either way, so both are excluded and the
/// stream reports a warning instead of pretending.
pub fn supports_enable_thinking(engine: &Engine) -> bool {
    matches!(engine.id, "vllm" | "sglang")
}

/// `ResponseFormat { type: "json", schema? }` -> the server's structured
/// output. vLLM, SGLang and PrismML all take the OpenAI `json_schema` wrapper
/// (`{"type":"json_schema","json_schema":{"name","schema"}}`), which is the
/// shape their guided-decoding backends expect; `json_object` is the looser
/// fallback when the caller supplied no schema.
pub fn build_response_format(rf: &ResponseFormat) -> Value {
    match &rf.schema {
        Some(schema) => json!({
            "type": "json_schema",
            "json_schema": { "name": "response", "schema": schema }
        }),
        None => json!({ "type": "json_object" }),
    }
}

/// Build the request body for `engine`.
pub fn build_body(engine: &Engine, args: &BodyArgs) -> Value {
    let mut body = json!({
        "model": args.model,
        "max_tokens": args.max_tokens,
        "messages": to_wire_messages(&args.messages, &args.system_prompt),
        "stream": true,
        // All four report usage on the final chunk; the OpenAI knob is
        // tolerated by each and needed when a reverse proxy sits in front.
        "stream_options": { "include_usage": true },
    });
    if supports_enable_thinking(engine) {
        body["chat_template_kwargs"] = json!({ "enable_thinking": args.enable_thinking });
    }
    let wire_tools = functions_to_wire(&args.tools);
    if !wire_tools.is_empty() {
        body["tools"] = Value::Array(wire_tools);
    }
    if let Some(rf) = &args.response_format {
        body["response_format"] = build_response_format(rf);
    }
    body
}

pub fn build_headers(cfg: &EngineConfigResolved) -> Vec<(&'static str, String)> {
    let mut headers = vec![("content-type", "application/json".to_string())];
    if let Some(credential) = &cfg.credential_value {
        headers.push(("authorization", format!("Bearer {credential}")));
    }
    headers
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_router::types::content::ContentBlock;
    use llm_router::types::messages::{UserMessage, UserRoleTag};

    fn vllm() -> &'static Engine {
        crate::engine("vllm").unwrap()
    }

    fn args() -> BodyArgs {
        BodyArgs {
            model: "qwen3.8-27b".into(),
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
    fn body_has_required_fields_for_every_engine() {
        for engine in crate::ENGINES.iter() {
            let body = build_body(engine, &args());
            assert_eq!(body["model"], "qwen3.8-27b", "{}", engine.id);
            assert_eq!(body["max_tokens"], 4096, "{}", engine.id);
            assert_eq!(body["stream"], true);
            assert_eq!(body["stream_options"]["include_usage"], true);
            assert_eq!(body["messages"][0]["role"], "system");
            assert_eq!(body["messages"][1]["role"], "user");
            assert!(body.get("tools").is_none(), "empty tools array omitted");
            assert!(body.get("temperature").is_none());
            assert!(
                body.get("max_completion_tokens").is_none(),
                "all four accept max_tokens"
            );
        }
    }

    #[test]
    fn only_the_qwen_template_engines_get_enable_thinking() {
        for engine in crate::ENGINES.iter() {
            let body = build_body(engine, &args());
            let has_kwarg = body.get("chat_template_kwargs").is_some();
            assert_eq!(
                has_kwarg,
                supports_enable_thinking(engine),
                "{}",
                engine.id
            );
        }
        let mut on = args();
        on.enable_thinking = true;
        assert_eq!(
            build_body(vllm(), &on)["chat_template_kwargs"]["enable_thinking"],
            true
        );
        assert_eq!(
            build_body(vllm(), &args())["chat_template_kwargs"]["enable_thinking"],
            false
        );
    }

    #[test]
    fn only_a_level_other_than_off_enables_thinking() {
        assert!(!enable_thinking(None));
        assert!(!enable_thinking(Some(ThinkingLevel::Off)));
        assert!(enable_thinking(Some(ThinkingLevel::Low)));
    }

    #[test]
    fn tools_serialize_when_present() {
        let mut a = args();
        a.tools = vec![AgentFunction {
            name: "agent::call".into(),
            description: "d".into(),
            parameters: json!({ "type": "object" }),
            label: None,
            execution_mode: None,
        }];
        let body = build_body(vllm(), &a);
        assert_eq!(body["tools"][0]["function"]["name"], "agent__call");
    }

    #[test]
    fn response_format_uses_the_openai_json_schema_wrapper() {
        let schema = json!({ "type": "object", "properties": {} });
        let with_schema = build_response_format(&ResponseFormat {
            r#type: "json".into(),
            schema: Some(schema.clone()),
        });
        assert_eq!(with_schema["type"], "json_schema");
        assert_eq!(with_schema["json_schema"]["schema"], schema);
        assert!(
            with_schema.get("schema").is_none(),
            "not llama.cpp's direct nesting"
        );

        let without_schema = build_response_format(&ResponseFormat {
            r#type: "json".into(),
            schema: None,
        });
        assert_eq!(without_schema["type"], "json_object");
    }

    #[test]
    fn headers_carry_bearer_auth_only_when_a_credential_is_configured() {
        let cfg = EngineConfigResolved {
            credential_value: Some("sk-test".into()),
            model: "m".into(),
            max_tokens: 4096,
            api_url: "http://127.0.0.1:8000/v1/chat/completions".into(),
        };
        let h = build_headers(&cfg);
        assert!(h.contains(&("authorization", "Bearer sk-test".to_string())));

        let cfg = EngineConfigResolved {
            credential_value: None,
            ..cfg
        };
        let h = build_headers(&cfg);
        assert!(
            !h.iter().any(|(k, _)| *k == "authorization"),
            "no credential configured -> no Authorization header sent"
        );
    }
}
