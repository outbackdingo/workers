//! POST the configured Chat Completions endpoint (stream:true) -> SSE ->
//! mpsc<AssistantMessageEvent>.
//! The receiver dropping aborts the upstream: every send error returns, which
//! drops the reqwest response mid-body and closes the connection.
use crate::errors::classify;
use crate::sse::{build_final, build_partial, handle_chunk, synthetic_error_event, PartialState};
use futures::StreamExt;
use llm_router::provider_scaffold::sse_transport::{
    append_utf8_chunk, classify_stream_end, drain_sse_blocks, error_chain, flush_tail,
    truncated_stream_error, CloseFraming, StreamEnd, TailFlush,
};
use llm_router::types::events::{AssistantMessageEvent, ErrorKind};
use serde_json::Value;
use tokio::sync::mpsc;

pub struct UpstreamArgs {
    /// The provider id, stamped on error frames so a shared log names the
    /// engine that failed.
    pub provider_id: &'static str,
    pub api_url: String,
    pub model: String,
    pub body: Value,
    pub headers: Vec<(&'static str, String)>,
    /// Report-and-continue notices for the final message (stream contract).
    pub warnings: Vec<String>,
}

pub fn spawn_upstream(
    client: reqwest::Client,
    args: UpstreamArgs,
) -> mpsc::Receiver<AssistantMessageEvent> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        // Race the call against receiver-side closure: send errors alone only
        // observe a dropped receiver at the next send, so a silent upstream
        // (parked in a chunk read, nothing to send) would otherwise keep the
        // HTTP stream - and billed generation - alive until the next frame.
        let closed = tx.clone();
        tokio::select! {
            _ = run_upstream(client, args, tx) => {}
            _ = closed.closed() => {}
        }
    });
    rx
}

/// Last `data: ` payload in an SSE block, if any.
fn data_line(block: &str) -> Option<&str> {
    block
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .next_back()
}

async fn run_upstream(
    client: reqwest::Client,
    args: UpstreamArgs,
    tx: mpsc::Sender<AssistantMessageEvent>,
) {
    let mut req = client.post(&args.api_url);
    for (name, value) in &args.headers {
        req = req.header(*name, value);
    }
    let resp = match req.json(&args.body).send().await {
        Ok(r) => r,
        Err(e) => {
            let _ = tx
                .send(synthetic_error_event(
                    args.provider_id,
                    &format!("{} fetch failed: {}", args.provider_id, error_chain(&e)),
                    &args.model,
                    ErrorKind::Transient,
                ))
                .await;
            return;
        }
    };

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        let kind = classify(Some(status.as_u16()), &text);
        let msg = if text.is_empty() {
            format!("{} http {status}", args.provider_id)
        } else {
            text
        };
        let _ = tx
            .send(synthetic_error_event(
                args.provider_id,
                &msg,
                &args.model,
                kind,
            ))
            .await;
        return;
    }

    let mut state = PartialState::new(args.provider_id, args.warnings);
    if tx
        .send(AssistantMessageEvent::Start {
            partial: build_partial(&state, &args.model),
        })
        .await
        .is_err()
    {
        return; // receiver gone before the first frame
    }

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    // Cross-chunk UTF-8 buffering: network chunks split multibyte
    // codepoints, and a per-chunk lossy conversion corrupts them to U+FFFD.
    let mut byte_buf = Vec::new();
    // Block decoder: [DONE] closes the stream (Stop + Done, Done terminal);
    // anything else parses and runs the chunk state machine.
    let decode = |data_block: &str, state: &mut PartialState, model: &str| -> Vec<AssistantMessageEvent> {
        let Some(data) = data_line(data_block) else {
            return vec![];
        };
        if data == "[DONE]" {
            return vec![
                AssistantMessageEvent::Stop {
                    stop_reason: state.stop_reason(),
                    error_message: None,
                    error_kind: None,
                },
                AssistantMessageEvent::Done {
                    message: build_final(state, model),
                },
            ];
        }
        let Ok(parsed) = serde_json::from_str::<Value>(data) else {
            return vec![];
        };
        handle_chunk(&parsed, state, model)
    };
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                let _ = tx
                    .send(synthetic_error_event(
                        args.provider_id,
                        &format!("stream read failed: {e}"),
                        &args.model,
                        ErrorKind::Transient,
                    ))
                    .await;
                return;
            }
        };
        append_utf8_chunk(&mut byte_buf, &mut buf, &chunk);
        if drain_sse_blocks(&mut buf, &tx, &mut |block: &str| {
            decode(block, &mut state, &args.model)
        })
        .await
        {
            return; // terminal forwarded, or receiver dropped -> abort upstream
        }
    }
    // Body closed: one terminal frame, decided by the shared end-of-stream policy.
    let tail = flush_tail(&mut buf, &tx, &mut |block: &str| {
        decode(block, &mut state, &args.model)
    })
    .await;
    if tail == TailFlush::Terminal {
        return;
    }
    let event = match classify_stream_end(&state, tail, CloseFraming::Accepted) {
        StreamEnd::Complete => AssistantMessageEvent::Done {
            message: build_final(&state, &args.model),
        },
        StreamEnd::Truncated(truncation) => truncated_stream_error(
            build_final(&state, &args.model),
            args.provider_id,
            truncation,
        ),
    };
    let _ = tx.send(event).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// One-shot HTTP stub: accepts a single connection, consumes the request
    /// head, writes `response` verbatim, closes (read-until-close framing).
    async fn stub(response: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 65536];
                let _ = sock.read(&mut buf).await;
                let _ = sock.write_all(response.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        format!("http://{addr}/v1/chat/completions")
    }

    fn args(api_url: String) -> UpstreamArgs {
        UpstreamArgs {
            provider_id: "vllm",
            api_url,
            model: "qwen-test".into(),
            body: serde_json::json!({ "stream": true }),
            headers: vec![("authorization", "Bearer sk-test".into())],
            warnings: vec![],
        }
    }

    async fn drain(mut rx: mpsc::Receiver<AssistantMessageEvent>) -> Vec<AssistantMessageEvent> {
        let mut out = Vec::new();
        while let Some(ev) = rx.recv().await {
            out.push(ev);
        }
        out
    }

    const HAPPY: &str = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":4}}}\n\ndata: [DONE]\n\n";

    #[tokio::test(flavor = "multi_thread")]
    async fn happy_stream_yields_start_through_stop_and_done() {
        let url = stub(HAPPY).await;
        let events = drain(spawn_upstream(reqwest::Client::new(), args(url))).await;
        assert!(matches!(
            events.first(),
            Some(AssistantMessageEvent::Start { .. })
        ));
        assert!(
            matches!(
                events[events.len() - 2],
                AssistantMessageEvent::Stop {
                    stop_reason: llm_router::types::events::StopReason::End,
                    ..
                }
            ),
            "stop precedes done"
        );
        match events.last() {
            Some(AssistantMessageEvent::Done { message }) => {
                let usage = message.usage.as_ref().unwrap();
                assert_eq!(usage.input, Some(8));
                assert_eq!(usage.output, Some(2));
                assert_eq!(usage.cache_read, Some(4));
                assert_eq!(message.provider, "vllm");
            }
            other => panic!("want done, got {other:?}"),
        }
        assert_eq!(events.iter().filter(|e| e.is_terminal()).count(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn http_401_yields_auth_expired_error_frame() {
        let url = stub(
            "HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\nconnection: close\r\n\r\n{\"error\":{\"message\":\"Incorrect API key provided.\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key\"}}",
        )
        .await;
        let events = drain(spawn_upstream(reqwest::Client::new(), args(url))).await;
        assert_eq!(events.len(), 1);
        match &events[0] {
            AssistantMessageEvent::Error { error } => {
                assert_eq!(error.error_kind, Some(ErrorKind::AuthExpired));
            }
            other => panic!("want error, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn vllm_context_overflow_body_is_classified_as_overflow() {
        // The failure the harness must retry by compacting, not by giving up.
        let body = "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\nconnection: close\r\n\r\n{\"object\":\"error\",\"message\":\"This model's maximum context length is 262144 tokens. However, you requested 295444 tokens.\",\"type\":\"BadRequestError\"}";
        let url = stub(body).await;
        let events = drain(spawn_upstream(reqwest::Client::new(), args(url))).await;
        match &events[0] {
            AssistantMessageEvent::Error { error } => {
                assert_eq!(error.error_kind, Some(ErrorKind::ContextOverflow));
            }
            other => panic!("want error, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn builder_error_surfaces_its_source_not_just_builder_error() {
        let mut a = args("http://127.0.0.1:1/v1/chat/completions".into());
        a.headers = vec![("authorization", "Bearer sk-bad\ninjected".into())];
        let events = drain(spawn_upstream(reqwest::Client::new(), a)).await;
        assert_eq!(events.len(), 1);
        match &events[0] {
            AssistantMessageEvent::Error { error } => {
                let msg = error.error_message.as_deref().unwrap_or_default();
                assert!(msg.starts_with("vllm fetch failed: "), "got {msg:?}");
                assert_ne!(msg, "vllm fetch failed: builder error", "source dropped");
            }
            other => panic!("want error, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn connect_failure_yields_transient_error_frame() {
        let dead = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            format!("http://{}/v1/chat/completions", l.local_addr().unwrap())
        };
        let events = drain(spawn_upstream(reqwest::Client::new(), args(dead))).await;
        assert_eq!(events.len(), 1);
        match &events[0] {
            AssistantMessageEvent::Error { error } => {
                assert_eq!(error.error_kind, Some(ErrorKind::Transient));
            }
            other => panic!("want error, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn empty_stream_ends_as_one_error_not_an_empty_done() {
        let url = stub("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n").await;
        let events = drain(spawn_upstream(reqwest::Client::new(), args(url))).await;
        assert_eq!(events.iter().filter(|e| e.is_terminal()).count(), 1);
        match events.last() {
            Some(AssistantMessageEvent::Error { error }) => {
                let message = error.error_message.as_deref().unwrap_or_default();
                assert!(message.contains("reason=empty"), "{message}");
            }
            other => panic!("want error, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn warnings_arrive_on_the_final_message() {
        let url = stub(HAPPY).await;
        let mut a = args(url);
        a.warnings = vec!["thinking_level ignored".into()];
        let events = drain(spawn_upstream(reqwest::Client::new(), a)).await;
        match events.last() {
            Some(AssistantMessageEvent::Done { message }) => {
                assert_eq!(
                    message.warnings.as_deref(),
                    Some(&["thinking_level ignored".to_string()][..])
                );
            }
            other => panic!("want done, got {other:?}"),
        }
    }
}
