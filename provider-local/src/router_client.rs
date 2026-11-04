//! Provider-scoped shims over the shared router-protocol client
//! (`llm_router::provider_scaffold::router_client`): every call binds one
//! engine's provider id and carries its registration token.
use crate::Engine;
use iii_sdk::errors::Error;
use iii_sdk::IIIClient;
use llm_router::provider_scaffold::router_client as scaffold;
use llm_router::types::model::Model;
use llm_router::types::router::ProviderResolveResponse;
use serde_json::Value;

/// `router::provider::resolve` - credential + effective settings for `engine`.
pub async fn resolve(
    iii: &IIIClient,
    engine: &Engine,
    token: Option<&str>,
) -> Result<ProviderResolveResponse, Error> {
    scaffold::resolve(iii, engine.id, token, Some(engine.credential_env_var)).await
}

/// `router::models::reconcile` - replace this engine's catalog slice.
pub async fn reconcile(
    iii: &IIIClient,
    engine: &Engine,
    models: Vec<Model>,
    token: Option<&str>,
) -> Result<(), Error> {
    scaffold::reconcile(iii, engine.id, models, token).await
}

/// `router::provider::register` - returns the registration token to persist.
pub async fn register(iii: &IIIClient, declaration: Value) -> Result<Value, Error> {
    scaffold::register(iii, declaration).await
}
