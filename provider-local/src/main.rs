//! `provider-local` binary entry.
//!
//! Boots the shared scaffolding once (one HTTP client, one resolve cache, one
//! in-flight abort map), registers the `provider-local` configuration entry,
//! then registers one provider per enabled engine. Operator settings for the
//! servers themselves (api_url, api_key, max_tokens) come from llm-router's
//! resolve step per request; the knobs only this worker owns are the enabled
//! set and the window override for servers that publish none.
use clap::Parser;
use iii_config_client::{self as config_client, EntrySpec};
use iii_sdk::runtime::WorkerMetadata;
use iii_sdk::{register_worker, IIIClient, InitOptions};
use llm_router::provider_scaffold::aborts::StreamAborts;
use llm_router::provider_scaffold::cache::ScaffoldCache;
use provider_local::register::register_engine;
use provider_local::settings::{self, ConfigCell, WorkerConfig};
use provider_local::ENGINES;
use std::sync::Arc;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(
    name = "provider-local",
    about = "Local OpenAI-compatible inference servers (vLLM, SGLang, PrismML, mistral.rs) \
             as llm-router providers."
)]
struct Cli {
    /// Accepted for the standard worker CLI contract; per-server config comes
    /// from llm-router's resolve step, not from a file.
    #[arg(long, default_value = "./config.yaml")]
    config: String,

    #[arg(long, env = "III_URL", default_value = "ws://127.0.0.1:49134")]
    url: String,

    #[arg(long)]
    manifest: bool,
}

/// True when the YAML contents carry anything beyond comments, blank lines,
/// or a bare empty mapping (`{}`).
fn has_config_keys(contents: &str) -> bool {
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .any(|line| line != "{}")
}

/// Read this worker's entry, falling back to the seeded defaults when nothing
/// is stored or the read fails (a first boot with no configuration worker
/// reachable yet must not take the providers down).
async fn load_config(iii: &IIIClient) -> WorkerConfig {
    match config_client::fetch(iii, settings::CONFIG_ID).await {
        Ok(Some(value)) => match WorkerConfig::from_json(&value) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("[provider-local] {e}; falling back to defaults");
                WorkerConfig::seeded()
            }
        },
        Ok(None) => WorkerConfig::seeded(),
        Err(e) => {
            eprintln!("[provider-local] configuration read failed ({e}); using defaults");
            WorkerConfig::seeded()
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // Registry publish pipeline: print the manifest JSON and exit.
    if cli.manifest {
        println!(
            "{}",
            serde_json::to_string_pretty(&provider_local::manifest::build_manifest())?
        );
        return Ok(());
    }

    if let Ok(contents) = std::fs::read_to_string(&cli.config) {
        if has_config_keys(&contents) {
            tracing::warn!(
                path = %cli.config,
                "provider-local takes no file-based config; configure the engines through the \
                 engine's `provider-local` configuration entry and the servers themselves through \
                 `llm-router` - ignoring this file's keys"
            );
        }
    }

    let iii = register_worker(
        &cli.url,
        InitOptions {
            metadata: Some(WorkerMetadata {
                runtime: "rust".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                name: "provider-local".to_string(),
                os: std::env::consts::OS.to_string(),
                pid: Some(std::process::id()),
                telemetry: None,
                ..WorkerMetadata::default()
            }),
            ..InitOptions::default()
        },
    );

    // Declare the entry (seeding only when nothing is stored), then read the
    // effective value before anything that depends on it is built.
    let spec = EntrySpec {
        id: settings::CONFIG_ID,
        form_id: settings::CONFIG_ID,
        name: "Local providers",
        description: "Which local inference engines this worker serves, and the context window \
                      to publish for one that does not declare its own.",
        schema: settings::schema(),
        default_value: settings::initial_value(),
    };
    if let Err(e) = config_client::ensure(&iii, &spec, None).await {
        eprintln!("[provider-local] configuration registration failed ({e}); using defaults");
    }
    let cell = settings::new_cell();
    settings::apply(&cell, load_config(&iii).await).await;

    // Reload on every update of this worker's entry, so a window override
    // reaches the catalog without a restart.
    let engine_client = Arc::new(iii.clone());
    let reload_cell = cell.clone();
    let reload = config_client::on_change(
        &engine_client.clone(),
        settings::CONFIG_ID,
        "provider-local::on-config-change",
        "Internal: reload the enabled engines and window overrides after a configuration update.",
        move || {
            let engine = engine_client.clone();
            let cell = reload_cell.clone();
            async move {
                let cfg = load_config(&engine).await;
                settings::apply(&cell, cfg).await;
            }
        },
    );
    // Close the boot gap: an update landing between the read above and the
    // observer becoming active reached no one.
    reload.run().await;

    let cache = ScaffoldCache::new();
    // Streaming uses no total timeout (the router owns stream budgets);
    // connect failures surface fast.
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("reqwest client");
    let aborts = StreamAborts::new();

    for engine in ENGINES.iter() {
        if !settings::enabled(&cell, engine.id).await {
            println!(
                "[provider-local:{}] disabled in the provider-local configuration; not registered",
                engine.id
            );
            continue;
        }
        register_engine(
            iii.clone(),
            http.clone(),
            cache.clone(),
            aborts.clone(),
            cell.clone(),
            engine,
        )
        .await?;
        println!("[provider-local:{}] registered", engine.id);
    }
    tracing::info!(url = %cli.url, "provider-local registered");

    tokio::signal::ctrl_c().await?;
    iii.shutdown_async().await;
    Ok(())
}

/// The engine ids this process serves, for the log line and tests.
#[allow(dead_code)]
fn served(cell: &ConfigCell) -> Vec<&'static str> {
    let _ = cell;
    ENGINES.iter().map(|e| e.id).collect()
}

#[cfg(test)]
mod tests {
    use super::has_config_keys;

    #[test]
    fn empty_and_comment_only_contents_have_no_keys() {
        assert!(!has_config_keys(""));
        assert!(!has_config_keys("\n\n"));
        assert!(!has_config_keys("# only a comment\n  # indented comment\n"));
        assert!(!has_config_keys("{}\n"));
    }

    #[test]
    fn real_keys_are_detected() {
        assert!(has_config_keys("api_url: https://example.com\n"));
        assert!(has_config_keys("# comment\nmax_tokens: 8192\n"));
    }
}
