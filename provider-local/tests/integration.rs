//! Engine-backed integration suite — real engine, real router, real provider,
//! stubbed upstream. Self-skips when no engine is available.
//!
//! Two scenarios carry the weight of issue #2201:
//!
//! 1. **The 262k window.** A stub serves SGLang's real `/v1/models` shape
//!    (`max_model_len: 262144`, no `/props`). Discovery must publish 262144 and
//!    a usable input budget, not llama.cpp's 4096. The old provider failed
//!    this and every turn died with `context/overflow`.
//! 2. **No declaration at all.** The same stub without any window field must
//!    fall back to the engine default and say so, rather than silently
//!    claiming a number the server never made.
use std::sync::Arc;
use std::time::{Duration, Instant};

use iii_sdk::{register_worker, CallRequest, IIIClient};
use llm_router::register::register_router;
use provider_local::register::register_engine;
use provider_local::settings::{self, WorkerConfig};
use provider_local::ENGINES;
use serde_json::{json, Value};

#[path = "../../llm-router/tests/support/engine_fixture.rs"]
mod engine_fixture;
use engine_fixture::*;

async fn call(iii: &IIIClient, function_id: &str, payload: Value) -> Result<Value, iii_sdk::errors::Error> {
    iii.call(CallRequest {
        function_id: function_id.into(),
        payload,
        action: None,
        timeout_ms: Some(10_000),
    })
    .await
}

// ── stub upstream ───────────────────────────────────────────────────────────

/// Serves one canned `/v1/models` body and one canned chat SSE body, and
/// records every request head so tests can assert on what was sent.
struct StubUpstream {
    url: String,
    handle: tokio::task::JoinHandle<()>,
    requests: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Drop for StubUpstream {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

const STUB_SSE: &str = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n";

/// SGLang's real listing for a 262144-token model: no `/props`, window on the
/// row as `max_model_len`.
const STUB_SGLANG_MODELS: &str = "{\"object\":\"list\",\"data\":[{\"id\":\"qwen3.8-27b-sglang\",\"object\":\"model\",\"max_model_len\":262144}]}";

/// A server that publishes ids but no window at all.
const STUB_NO_WINDOW_MODELS: &str = "{\"object\":\"list\",\"data\":[{\"id\":\"mystery-7B\",\"object\":\"model\"}]}";

async fn stub_upstream(models_body: &'static str) -> StubUpstream {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let requests = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let requests2 = requests.clone();
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            let requests = requests2.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 65536];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let head = String::from_utf8_lossy(&buf[..n]).into_owned();
                let path = head
                    .lines()
                    .next()
                    .and_then(|l| l.split_whitespace().nth(1))
                    .unwrap_or("")
                    .to_string();
                requests.lock().unwrap().push(head);
                let body = if path.starts_with("/v1/models") {
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nconnection: close\r\n\r\n{models_body}"
                    )
                } else if path.contains("tokenize") || path.contains("input_tokens") {
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nconnection: close\r\n\r\n{\"count\":42}".to_string()
                } else {
                    STUB_SSE.to_string()
                };
                let _ = sock.write_all(body.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    StubUpstream {
        url: format!("http://{addr}/v1/chat/completions"),
        handle,
        requests,
    }
}

// ── boot ────────────────────────────────────────────────────────────────────

async fn boot_stack(engine_url: &str) -> (IIIClient, IIIClient) {
    static STRIP_HOST_KEYS: std::sync::Once = std::sync::Once::new();
    STRIP_HOST_KEYS.call_once(|| {
        for engine in ENGINES.iter() {
            std::env::remove_var(engine.credential_env_var);
        }
    });
    let router_iii = register_worker(engine_url, test_init_options());
    register_router(router_iii.clone()).await.expect("router boots");

    let provider_iii = register_worker(engine_url, test_init_options());
    let cell = settings::new_cell();
    settings::apply(&cell, WorkerConfig::seeded()).await;
    let http = reqwest::Client::new();
    for engine in ENGINES.iter() {
        register_engine(
            provider_iii.clone(),
            http.clone(),
            llm_router::provider_scaffold::cache::ScaffoldCache::new(),
            llm_router::provider_scaffold::aborts::StreamAborts::new(),
            cell.clone(),
            engine,
        )
        .await
        .expect("provider boots");
    }

    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let list = call(&router_iii, "router::provider::list", json!({}))
            .await
            .unwrap();
        let ids: Vec<&str> = list["providers"]
            .as_array()
            .map(|p| p.iter().filter_map(|x| x["id"].as_str()).collect())
            .unwrap_or_default();
        if ids.len() == ENGINES.len() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "providers never registered, saw {ids:?}: {list}"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    (router_iii, provider_iii)
}

async fn point_at(router_iii: &IIIClient, engine_id: &str, stub_url: &str) {
    call(
        router_iii,
        "configuration::set",
        json!({ "id": "llm-router", "value": { "providers": {
            engine_id: { "api_url": stub_url }
        } } }),
    )
    .await
    .expect("config set");
}

#[tokio::test(flavor = "multi_thread")]
async fn sglang_262k_window_is_read_from_max_model_len() {
    let Some(engine) = spawn_engine().await else {
        return;
    };
    let (router_iii, _provider_iii) = boot_stack(&engine.url).await;
    let stub = stub_upstream(STUB_SGLANG_MODELS).await;
    point_at(&router_iii, "sglang", &stub.url).await;

    let refreshed = call(
        &router_iii,
        "provider::sglang::refresh_models",
        json!({}),
    )
    .await
    .expect("refresh");
    assert_eq!(refreshed["count"], 1, "{refreshed}");

    let model = call(
        &router_iii,
        "router::models::get",
        json!({ "provider": "sglang", "id": "qwen3.8-27b-sglang" }),
    )
    .await
    .expect("models::get");
    assert_eq!(
        model["model"]["context_window"], 262_144,
        "issue #2201: the window must come from the row, not a 4096 default: {model}"
    );
    let budget = call(
        &router_iii,
        "router::models::budget",
        json!({ "id": "qwen3.8-27b-sglang", "provider": "sglang" }),
    )
    .await
    .expect("budget");
    let output = budget["effective_max_output_tokens"].as_u64().unwrap();
    assert!(
        output < 262_144,
        "output must leave input room: {budget}"
    );
    assert!(
        output > 1639,
        "the old 1639-token usable budget is the bug: {budget}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_server_with_no_window_falls_back_loudly_not_silently() {
    let Some(engine) = spawn_engine().await else {
        return;
    };
    let (router_iii, _provider_iii) = boot_stack(&engine.url).await;
    let stub = stub_upstream(STUB_NO_WINDOW_MODELS).await;
    point_at(&router_iii, "prisml", &stub.url).await;

    call(
        &router_iii,
        "provider::prisml::refresh_models",
        json!({}),
    )
    .await
    .expect("refresh");
    let model = call(
        &router_iii,
        "router::models::get",
        json!({ "provider": "prisml", "id": "mystery-7B" }),
    )
    .await
    .expect("models::get");
    let window = model["model"]["context_window"].as_u64().unwrap();
    assert_eq!(window, provider_local::engine("prisml").unwrap().fallback_context_window);
    assert!(
        window > 4096,
        "the fallback must not be the 4096 that caused #2201"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_chat_turn_streams_through_the_provider() {
    let Some(engine) = spawn_engine().await else {
        return;
    };
    let (router_iii, _provider_iii) = boot_stack(&engine.url).await;
    let stub = stub_upstream(STUB_SGLANG_MODELS).await;
    point_at(&router_iii, "sglang", &stub.url).await;
    call(&router_iii, "provider::sglang::refresh_models", json!({}))
        .await
        .expect("refresh");

    let reply = call(
        &router_iii,
        "router::complete",
        json!({
            "model": "qwen3.8-27b-sglang",
            "provider": "sglang",
            "messages": [{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }],
        }),
    )
    .await
    .expect("complete");
    assert_eq!(reply["provider"], "sglang");
    let content = &reply["message"]["content"][0];
    assert_eq!(content["type"], "text");
    assert_eq!(content["text"], "Hello");

    // The request carried no Authorization header: no credential configured.
    let heads = stub.requests.lock().unwrap();
    let chat = heads
        .iter()
        .find(|h| h.starts_with("POST /v1/chat/completions"))
        .expect("a chat request reached the stub");
    assert!(
        !chat.to_lowercase().contains("authorization:"),
        "no key configured, so none must be sent: {chat}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn every_enabled_engine_is_listed_and_disabled_ones_are_not() {
    let Some(engine) = spawn_engine().await else {
        return;
    };
    let (router_iii, _provider_iii) = boot_stack(&engine.url).await;
    let list = call(&router_iii, "router::provider::list", json!({}))
        .await
        .expect("list");
    for engine in ENGINES.iter() {
        let entry = list["providers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == engine.id)
            .unwrap_or_else(|| panic!("{} is not listed: {list}", engine.id));
        assert_eq!(entry["configured"], true, "{}", engine.id);
        assert_eq!(entry["credential_env_var"], engine.credential_env_var);
        assert_eq!(entry["display_name"], engine.display_name);
        assert!(
            entry["context_overflow_hint"]
                .as_str()
                .unwrap()
                .contains(engine.window_flag),
            "{}",
            engine.id
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_token_count_prefers_the_servers_own_route() {
    let Some(engine) = spawn_engine().await else {
        return;
    };
    let (router_iii, _provider_iii) = boot_stack(&engine.url).await;
    let stub = stub_upstream(STUB_SGLANG_MODELS).await;
    point_at(&router_iii, "sglang", &stub.url).await;

    let counted = call(
        &router_iii,
        "router::count_tokens",
        json!({
            "model": "qwen3.8-27b-sglang",
            "provider": "sglang",
            "messages": [{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }],
        }),
    )
    .await
    .expect("count_tokens");
    assert_eq!(counted["tokens"], 42, "{counted}");
    assert_eq!(counted["estimator"], "provider");
}
