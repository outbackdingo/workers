//! A `provider::<id>::stream` iii function for one engine (provider stream
//! contract): write AssistantMessageEvent frames as JSON text messages into
//! the router-owned channel, terminal done/error last, then close.
use crate::config::config_from_resolve;
use crate::errors::classify_bus_error;
use crate::request::{build_body, build_headers, enable_thinking, supports_enable_thinking, BodyArgs};
use crate::sse::synthetic_error_event;
use crate::state;
use crate::upstream::{spawn_upstream, UpstreamArgs};
use crate::Engine;
use futures::future::BoxFuture;
use iii_sdk::errors::Error;
use iii_sdk::IIIClient;
use llm_router::channels::open_sink;
use llm_router::chat::relay::FrameSink;
use llm_router::provider_scaffold::aborts::{AbortGuard, StreamAborts};
use llm_router::provider_scaffold::cache::ScaffoldCache;
use llm_router::provider_scaffold::pump::{pump, pump_abortable, send_event, PING_INTERVAL};
use llm_router::types::events::ErrorKind;
use llm_router::types::router::{ProviderStreamInput, ProviderStreamOutput};

pub fn make_stream(
    iii: IIIClient,
    http: reqwest::Client,
    cache: ScaffoldCache,
    aborts: StreamAborts,
    engine: &'static Engine,
) -> impl Fn(ProviderStreamInput) -> BoxFuture<'static, Result<ProviderStreamOutput, Error>>
       + Send
       + Sync
       + 'static {
    move |input: ProviderStreamInput| {
        let (iii, http, cache, aborts) = (iii.clone(), http.clone(), cache.clone(), aborts.clone());
        Box::pin(async move {
            // Register BEFORE the first await: an abort landing while the sink
            // opens must latch, not hit an unknown id. The RAII guard
            // deregisters on every exit - early returns and an executor
            // cancelling this future mid-await alike.
            let abort_reg = input
                .resolution_key
                .as_ref()
                .map(|rid| aborts.register(rid));
            let sink = open_sink(&iii, &input.writer_ref).await?;
            run_stream_call(&iii, http, &cache, abort_reg.as_ref(), input, sink.as_ref(), engine)
                .await;
            sink.close();
            Ok(ProviderStreamOutput { ok: true })
        })
    }
}

async fn run_stream_call(
    iii: &IIIClient,
    http: reqwest::Client,
    cache: &ScaffoldCache,
    abort_reg: Option<&AbortGuard>,
    input: ProviderStreamInput,
    sink: &dyn FrameSink,
    engine: &'static Engine,
) {
    let model = input.model.clone();
    let mut warnings = Vec::new();

    // Token + resolve are cached (ScaffoldCache): zero engine round trips on
    // the hot path within the TTL. An auth-classified resolve failure drops the
    // cache so the next attempt re-resolves fresh - retrying stays the
    // router's job.
    let token = cache.load_token(iii, &state::scope(engine.id)).await;
    let resolved = match cache
        .resolve(iii, engine.id, token.as_deref(), Some(engine.credential_env_var))
        .await
    {
        Ok(r) => r,
        Err(e) => {
            let kind = classify_bus_error(&e);
            if kind == ErrorKind::AuthExpired {
                cache.invalidate();
            }
            let _ = send_event(
                sink,
                &synthetic_error_event(
                    engine.id,
                    &format!("router::provider::resolve failed: {e}"),
                    &model,
                    kind,
                ),
            );
            return;
        }
    };
    let cfg = match config_from_resolve(engine, &model, input.max_output_tokens, &resolved) {
        Ok(c) => c,
        Err(e) => {
            let _ = send_event(
                sink,
                &synthetic_error_event(engine.id, &e.to_string(), &model, ErrorKind::Permanent),
            );
            return;
        }
    };

    // Mapped onto the `enable_thinking` chat-template kwarg for the engines
    // whose templates read it (see request::supports_enable_thinking).
    let thinks = enable_thinking(input.thinking_level);
    if input.thinking_level.is_some() && !supports_enable_thinking(engine) {
        warnings.push(format!(
            "thinking_level ignored: {} exposes no per-request reasoning switch, so the level is \
             whatever the server started with",
            engine.display_name
        ));
    }

    let body = build_body(
        engine,
        &BodyArgs {
            model: cfg.model.clone(),
            max_tokens: cfg.max_tokens,
            system_prompt: input.system_prompt.unwrap_or_default(),
            messages: input.messages,
            tools: input.tools.unwrap_or_default(),
            response_format: input.response_format,
            enable_thinking: thinks,
        },
    );
    let headers = build_headers(&cfg);

    // Aborted while we were setting up - never start the upstream request.
    if abort_reg.is_some_and(|g| g.is_fired()) {
        return;
    }
    let rx = spawn_upstream(
        http,
        UpstreamArgs {
            provider_id: engine.id,
            api_url: cfg.api_url.clone(),
            model,
            body,
            headers,
            warnings,
        },
    );
    let kind = match abort_reg {
        Some(g) => pump_abortable(rx, sink, PING_INTERVAL, g.watch()).await,
        None => pump(rx, sink, PING_INTERVAL).await,
    };
    // An upstream auth terminal means the cached credential was rotated out
    // from under us: drop the cache so the next attempt re-resolves.
    if kind == Some(ErrorKind::AuthExpired) {
        cache.invalidate();
    }
}
