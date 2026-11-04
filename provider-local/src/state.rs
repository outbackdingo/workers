//! Registration-token persistence, scoped to one engine's worker id (shared
//! logic in `llm_router::provider_scaffold::state`). Each engine keeps its own
//! token, so a token mismatch on one never blocks the other three.
use iii_sdk::errors::Error;
use iii_sdk::IIIClient;
use llm_router::provider_scaffold::state as scaffold;

/// The iii-state scope for `engine`'s registration token.
pub fn scope(engine_id: &str) -> String {
    format!("provider-{engine_id}")
}

pub async fn load_token(iii: &IIIClient, engine_id: &str) -> Option<String> {
    scaffold::load_token(iii, &scope(engine_id)).await
}

pub async fn store_token(iii: &IIIClient, engine_id: &str, token: &str) -> Result<(), Error> {
    scaffold::store_token(iii, &scope(engine_id), token).await
}

#[cfg(test)]
mod tests {
    use super::scope;

    #[test]
    fn each_engine_gets_its_own_scope() {
        let scopes: Vec<String> = crate::ENGINES.iter().map(|e| scope(e.id)).collect();
        assert_eq!(scopes[0], "provider-vllm");
        let mut deduped = scopes.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(deduped.len(), scopes.len(), "scopes must not collide");
    }
}
