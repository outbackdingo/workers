//! Boot wiring per engine: the function surface, the `router::on-ready`
//! observer, and the declare-with-backoff loop.
//!
//! One `ScaffoldCache` and one `StreamAborts` are shared by all four engines -
//! they key by provider id and request id respectively - while each engine
//! keeps its own registration token and its own declaration.
use crate::config::DEFAULT_MAX_TOKENS;
use crate::discovery::{make_refresh_models, refresh_models};
use crate::errors::invalid_request_from_serde;
use crate::settings::ConfigCell;
use crate::stream_fn::make_stream;
use crate::{router_client, state, surface, Engine};
use iii_sdk::errors::Error;
use iii_sdk::{IIIClient, RegisterFunction};
use llm_router::provider_scaffold::aborts::{make_abort, StreamAborts};
use llm_router::provider_scaffold::cache::ScaffoldCache;
use llm_router::provider_scaffold::registration::typed_async_with_bad_request;
use llm_router::provider_scaffold::router_client::observe_router_ready;
use llm_router::types::router::{
    ProviderDeclaration, ProviderDefaults, ProviderReadyAck, RouterReadyEvent,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Duration;

/// The last default-model list sent, per provider id, so a refresh that finds
/// the same listing does not re-register for nothing.
static DECLARED_DEFAULTS: std::sync::Mutex<Option<Vec<(String, Vec<String>)>>> =
    std::sync::Mutex::new(None);

fn remembered_defaults(engine_id: &str) -> Option<Vec<String>> {
    let guard = DECLARED_DEFAULTS.lock().ok()?;
    guard
        .as_ref()?
        .iter()
        .find(|(id, _)| id == engine_id)
        .map(|(_, defaults)| defaults.clone())
}

fn remember_defaults(engine_id: &str, preferred: Vec<String>) {
    if let Ok(mut guard) = DECLARED_DEFAULTS.lock() {
        let entries = guard.get_or_insert_with(Vec::new);
        match entries.iter_mut().find(|(id, _)| id == engine_id) {
            Some((_, defaults)) => *defaults = preferred,
            None => entries.push((engine_id.to_string(), preferred)),
        }
    }
}

/// This engine's self-declaration.
pub fn declaration_with_defaults(
    engine: &Engine,
    defaults: Option<Vec<String>>,
) -> ProviderDeclaration {
    ProviderDeclaration {
        id: engine.id.into(),
        display_name: Some(engine.display_name.into()),
        credential_env_var: Some(engine.credential_env_var.into()),
        defaults: Some(ProviderDefaults {
            // Unset on purpose: the router returns `defaults.api_url` as the
            // resolved url, which would stop `config_from_resolve` from using
            // the engine's own default port.
            api_url: None,
            max_tokens: Some(DEFAULT_MAX_TOKENS),
            extra: BTreeMap::new(),
        }),
        config_schema: None, // the router's default {api_key, api_url, max_tokens}
        // Every one of these servers serves a real `GET /v1/models`, which is
        // what discovery needs; this also gates the router's
        // refresh-on-config-change call, so pointing `api_url` at a running
        // server populates the catalog at once.
        supports_model_listing: Some(true),
        // Ranked by discovery after each refresh (largest first).
        default_models: defaults,
        default_thinking_level: None,
        // A server that rejects over-long prompts cannot be fixed by
        // compacting in a console; the hint names the flag and the override.
        context_overflow_hint: Some(engine.context_overflow_hint()),
        // A local server started without an API key is the normal case, so
        // the router reports the provider configured with no credential.
        credential_optional: Some(true),
        // No static slice: refresh_models discovers the catalog live from the
        // resolved server's `/v1/models` right after registration.
        models: None,
        // Self-reported; availability mapping only, never authorization.
        worker_id: Some("provider-local".into()),
        // No icon: these are third-party engines, and painting this worker's
        // own mark beside vLLM's models would misattribute them.
        icon_svg: None,
    }
}

pub fn declaration(engine: &Engine) -> ProviderDeclaration {
    declaration_with_defaults(engine, remembered_defaults(engine.id))
}

/// One registration attempt for `engine`: declare (with the persisted token
/// when present) and persist the token the router returns.
pub async fn declare_once(iii: &IIIClient, engine: &Engine) -> Result<(), Error> {
    declare_payload(iii, engine, declaration(engine)).await
}

/// Re-register `engine` with discovery's ranked default list when it changed.
pub async fn redeclare_with_defaults(
    iii: &IIIClient,
    engine: &Engine,
    preferred: Vec<String>,
) -> Result<(), Error> {
    if preferred.is_empty() {
        return Ok(());
    }
    if remembered_defaults(engine.id).as_deref() == Some(&preferred) {
        return Ok(());
    }
    declare_payload(iii, engine, declaration_with_defaults(engine, Some(preferred.clone()))).await?;
    remember_defaults(engine.id, preferred);
    Ok(())
}

async fn declare_payload(
    iii: &IIIClient,
    engine: &Engine,
    declaration: ProviderDeclaration,
) -> Result<(), Error> {
    let token = state::load_token(iii, engine.id).await;
    let mut payload = serde_json::to_value(declaration).expect("serializable declaration");
    if let Some(t) = &token {
        payload["token"] = json!(t);
    }
    let resp = router_client::register(iii, payload).await?;
    if let Some(t) = resp.get("registration_token").and_then(Value::as_str) {
        if token.as_deref() != Some(t) {
            persist_registration_token(iii, engine, t).await?;
        }
    }
    Ok(())
}

async fn persist_registration_token(
    iii: &IIIClient,
    engine: &Engine,
    token: &str,
) -> Result<(), Error> {
    let mut delay = Duration::from_millis(200);
    for attempt in 0..5 {
        match state::store_token(iii, engine.id, token).await {
            Ok(()) => return Ok(()),
            Err(e) if attempt < 4 => {
                eprintln!(
                    "[provider-local:{}] store registration_token failed ({e}); retrying in {delay:?}",
                    engine.id
                );
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(Duration::from_secs(2));
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!("persist_registration_token loop always returns");
}

/// Retry until acknowledged: covers provider-before-router boot order.
pub async fn declare_with_backoff(iii: IIIClient, engine: &'static Engine) {
    let mut delay = Duration::from_millis(500);
    loop {
        match declare_once(&iii, engine).await {
            Ok(()) => {
                println!("[provider-local:{}] registered with llm-router", engine.id);
                return;
            }
            Err(e) => {
                eprintln!(
                    "[provider-local:{}] register failed ({e}); retrying in {delay:?}",
                    engine.id
                );
            }
        }
        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(Duration::from_secs(10));
    }
}

/// Register, then run live model discovery against the resolved server.
pub async fn declare_and_refresh(
    iii: IIIClient,
    http: reqwest::Client,
    engine: &'static Engine,
    cell: ConfigCell,
) {
    declare_with_backoff(iii.clone(), engine).await;
    match refresh_models(&iii, &http, engine, &cell).await {
        Ok(count) => println!("[provider-local:{}] catalog refreshed: {count} models", engine.id),
        Err(e) => eprintln!("[provider-local:{}] post-register refresh failed ({e})", engine.id),
    }
}

/// Register every function for one engine and start its declare loop.
pub async fn register_engine(
    iii: IIIClient,
    http: reqwest::Client,
    cache: ScaffoldCache,
    aborts: StreamAborts,
    cell: ConfigCell,
    engine: &'static Engine,
) -> Result<(), Error> {
    let ids = surface::ids_for(engine.id);

    iii.register_function(
        ids.stream,
        typed_async_with_bad_request(
            make_stream(
                iii.clone(),
                http.clone(),
                cache.clone(),
                aborts.clone(),
                engine,
            ),
            invalid_request_from_serde,
        )
        .description(surface::STREAM_DESC),
    );
    iii.register_function(
        ids.abort,
        typed_async_with_bad_request(make_abort(aborts), invalid_request_from_serde)
            .description(surface::ABORT_DESC)
            .metadata(json!({ "internal": true })),
    );
    iii.register_function(
        ids.refresh_models,
        RegisterFunction::new_async(make_refresh_models(
            iii.clone(),
            http.clone(),
            engine,
            cell.clone(),
        ))
        .description(surface::REFRESH_MODELS_DESC),
    );

    {
        let iii_count = iii.clone();
        let http_count = http.clone();
        let cache_count = cache.clone();
        let cell_count = cell.clone();
        iii.register_function(
            ids.count_tokens,
            RegisterFunction::new_async(move |req: crate::count_tokens::CountTokensRequest| {
                let (iii, http, cache, cell) = (
                    iii_count.clone(),
                    http_count.clone(),
                    cache_count.clone(),
                    cell_count.clone(),
                );
                async move { crate::count_tokens::handle(&iii, &http, &cache, &cell, engine, req).await }
            })
            .description(surface::COUNT_TOKENS_DESC)
            .metadata(json!({ "internal": true })),
        );
    }

    {
        let iii_embed = iii.clone();
        let http_embed = http.clone();
        let cache_embed = cache.clone();
        iii.register_function(
            ids.embed,
            RegisterFunction::new_async(move |req: crate::embed::EmbedRequest| {
                let (iii, http, cache) =
                    (iii_embed.clone(), http_embed.clone(), cache_embed.clone());
                async move { crate::embed::handle(&iii, &http, &cache, engine, req).await }
            })
            .description(surface::EMBED_DESC)
            .metadata(json!({ "internal": true })),
        );
    }

    // Re-declare whenever the router starts: this handler is the target of
    // the worker's observer of router::on-ready (observed below).
    {
        let iii_ready = iii.clone();
        let http_ready = http.clone();
        let cache_ready = cache.clone();
        let cell_ready = cell.clone();
        iii.register_function(
            ids.on_router_ready,
            RegisterFunction::new_async(move |_event: RouterReadyEvent| {
                let (iii, http, cell) =
                    (iii_ready.clone(), http_ready.clone(), cell_ready.clone());
                cache_ready.invalidate();
                async move {
                    tokio::spawn(declare_and_refresh(iii, http, engine, cell));
                    Ok::<_, Error>(ProviderReadyAck { ok: true })
                }
            })
            .description(surface::ON_ROUTER_READY_DESC)
            // Invoked by id (the router's ready fan-out and its re-discovery
            // nudge), never discovered - same as every other provider's ready
            // handler.
            .metadata(json!({ "internal": true })),
        );
    }
    // The observer outlives router restarts: it waits `pending` while
    // llm-router is down, and every router start's ready event reaches it.
    observe_router_ready(&iii, ids.on_router_ready);

    // Boot declare, off the boot path (a missing router must not block boot).
    tokio::spawn(declare_and_refresh(iii, http, engine, cell));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine;

    #[test]
    fn declaration_leaves_api_url_unset_so_each_engine_keeps_its_own_port() {
        for engine in crate::ENGINES.iter() {
            assert_eq!(
                declaration(engine).defaults.and_then(|d| d.api_url),
                None,
                "{}",
                engine.id
            );
        }
    }

    #[test]
    fn the_key_is_optional_for_every_local_engine() {
        for engine in crate::ENGINES.iter() {
            assert_eq!(declaration(engine).credential_optional, Some(true));
        }
    }

    #[test]
    fn each_declaration_carries_its_own_id_name_and_env_var() {
        for engine in crate::ENGINES.iter() {
            let decl = declaration(engine);
            assert_eq!(decl.id, engine.id);
            assert_eq!(decl.display_name.as_deref(), Some(engine.display_name));
            assert_eq!(
                decl.credential_env_var.as_deref(),
                Some(engine.credential_env_var)
            );
            // The hint must name whatever flag this engine uses, so an
            // operator knows what to change - PrismML's is not a --flag.
            assert!(
                decl.context_overflow_hint
                    .as_deref()
                    .unwrap()
                    .contains(engine.window_flag),
                "{} hint does not name {}",
                engine.id,
                engine.window_flag
            );
            assert_eq!(decl.worker_id.as_deref(), Some("provider-local"));
            assert_eq!(decl.icon_svg, None);
        }
    }

    #[test]
    fn discovered_defaults_ride_the_redeclaration_per_engine() {
        remember_defaults("vllm", vec!["a-27B".into(), "b-7B".into()]);
        assert_eq!(
            declaration(engine("vllm").unwrap()).default_models,
            Some(vec!["a-27B".to_string(), "b-7B".to_string()])
        );
        // SGLang keeps its own list; one engine's defaults never leak.
        assert_eq!(declaration(engine("sglang").unwrap()).default_models, None);
    }
}
