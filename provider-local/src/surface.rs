//! Wire-surface catalog for every `provider::<id>::*` function this worker
//! registers - the single source of truth for each function's id,
//! registration description, and schemars-derived request/response schemas.
//!
//! Golden-tested in `tests/schemas.rs`; keep in lockstep with
//! [`crate::register`]. Schema generation MUST mirror iii-sdk's internal
//! `json_schema_for` (`SchemaSettings::draft07()` on the handler's
//! request/response types) so a catalog snapshot pins exactly what
//! registration emits.
//!
//! Function ids are per engine and so must be `&'static str`: they are written
//! out literally in [`ids_for`] rather than built with `format!`, because the
//! registry publishes them and a typo in a runtime-built id is invisible until
//! a call fails.
use llm_router::types::router::{
    ProviderAbortRequest, ProviderAbortResponse, ProviderReadyAck, ProviderStreamInput,
    ProviderStreamOutput, RefreshModelsRequest, RefreshModelsResponse, RouterReadyEvent,
};

/// The function ids one engine registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ids {
    pub stream: &'static str,
    pub abort: &'static str,
    pub refresh_models: &'static str,
    pub on_router_ready: &'static str,
    pub count_tokens: &'static str,
    pub embed: &'static str,
}

/// Every function id, spelled out per engine.
pub fn ids_for(engine_id: &str) -> Ids {
    match engine_id {
        "vllm" => Ids {
            stream: "provider::vllm::stream",
            abort: "provider::vllm::abort",
            refresh_models: "provider::vllm::refresh_models",
            on_router_ready: "provider::vllm::on_router_ready",
            count_tokens: "provider::vllm::count_tokens",
            embed: "provider::vllm::embed",
        },
        "sglang" => Ids {
            stream: "provider::sglang::stream",
            abort: "provider::sglang::abort",
            refresh_models: "provider::sglang::refresh_models",
            on_router_ready: "provider::sglang::on_router_ready",
            count_tokens: "provider::sglang::count_tokens",
            embed: "provider::sglang::embed",
        },
        "prisml" => Ids {
            stream: "provider::prisml::stream",
            abort: "provider::prisml::abort",
            refresh_models: "provider::prisml::refresh_models",
            on_router_ready: "provider::prisml::on_router_ready",
            count_tokens: "provider::prisml::count_tokens",
            embed: "provider::prisml::embed",
        },
        "mistralrs" => Ids {
            stream: "provider::mistralrs::stream",
            abort: "provider::mistralrs::abort",
            refresh_models: "provider::mistralrs::refresh_models",
            on_router_ready: "provider::mistralrs::on_router_ready",
            count_tokens: "provider::mistralrs::count_tokens",
            embed: "provider::mistralrs::embed",
        },
        "local" => Ids {
            stream: "provider::local::stream",
            abort: "provider::local::abort",
            refresh_models: "provider::local::refresh_models",
            on_router_ready: "provider::local::on_router_ready",
            count_tokens: "provider::local::count_tokens",
            embed: "provider::local::embed",
        },
        other => panic!("no wire surface for unknown engine {other:?}"),
    }
}

/// One function's complete agent-facing wire surface.
pub struct FunctionSpec {
    pub function_id: &'static str,
    pub description: &'static str,
    pub request_schema: schemars::schema::RootSchema,
    pub response_schema: schemars::schema::RootSchema,
}

fn schema_of<T: schemars::JsonSchema>() -> schemars::schema::RootSchema {
    schemars::r#gen::SchemaSettings::draft07()
        .into_generator()
        .into_root_schema_for::<T>()
}

fn spec<Req, Resp>(function_id: &'static str, description: &'static str) -> FunctionSpec
where
    Req: schemars::JsonSchema,
    Resp: schemars::JsonSchema,
{
    FunctionSpec {
        function_id,
        description,
        request_schema: schema_of::<Req>(),
        response_schema: schema_of::<Resp>(),
    }
}

pub const STREAM_DESC: &str = "Stream a local OpenAI-compatible chat completion from this engine: \
     resolve credentials (optional - local servers commonly run without a key), call the upstream \
     Chat Completions API, and relay AssistantMessageEvent frames to writer_ref.";

pub const ABORT_DESC: &str = "Cancel the in-flight upstream stream for a request_id \
     (router::abort fan-out), stopping billed generation immediately.";

pub const REFRESH_MODELS_DESC: &str = "Discover the engine's live model catalog from GET \
     /v1/models, reading the context window each server actually publishes (max_model_len for \
     vLLM/SGLang, context_length for PrismML, the configured window for mistral.rs), and reconcile \
     it through the router; returns the model count written.";

pub const ON_ROUTER_READY_DESC: &str =
    "Internal: re-declares this provider and refreshes its catalog on every router::on-ready event.";

pub const COUNT_TOKENS_DESC: &str = "Count prompt tokens for {model, system_prompt?, tools?, \
     messages, thinking_level?, response_format?} through the engine's own counting route where it \
     has one, else with a local estimate; never runs the model and costs nothing.";

pub const EMBED_DESC: &str = "Batch text embeddings via the engine's /v1/embeddings, when the \
     server was started with an embedding-capable model. One vector per input, order preserved.";

/// The catalog for `engine_id`, in the order [`crate::register`] registers it.
pub fn catalog(engine_id: &str) -> Vec<FunctionSpec> {
    use crate::count_tokens::{CountTokensRequest, CountTokensResponse};
    use crate::embed::{EmbedRequest, EmbedResponse};

    let ids = ids_for(engine_id);
    vec![
        spec::<ProviderStreamInput, ProviderStreamOutput>(ids.stream, STREAM_DESC),
        spec::<ProviderAbortRequest, ProviderAbortResponse>(ids.abort, ABORT_DESC),
        spec::<RefreshModelsRequest, RefreshModelsResponse>(ids.refresh_models, REFRESH_MODELS_DESC),
        spec::<RouterReadyEvent, ProviderReadyAck>(ids.on_router_ready, ON_ROUTER_READY_DESC),
        spec::<CountTokensRequest, CountTokensResponse>(ids.count_tokens, COUNT_TOKENS_DESC),
        spec::<EmbedRequest, EmbedResponse>(ids.embed, EMBED_DESC),
    ]
}

/// Every function id this worker registers, across every engine, in
/// registration order.
pub fn all_ids() -> Vec<&'static str> {
    crate::ENGINES
        .iter()
        .flat_map(|engine| catalog(engine.id).into_iter().map(|s| s.function_id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_engine_catalog_uses_that_engine_s_own_ids() {
        for engine in crate::ENGINES.iter() {
            let ids: Vec<&str> = catalog(engine.id).iter().map(|s| s.function_id).collect();
            let expected = ids_for(engine.id);
            assert_eq!(
                ids,
                vec![
                    expected.stream,
                    expected.abort,
                    expected.refresh_models,
                    expected.on_router_ready,
                    expected.count_tokens,
                    expected.embed,
                ]
            );
            for id in ids {
                assert!(
                    id.starts_with(&format!("provider::{}::", engine.id)),
                    "{id} is not namespaced to {}",
                    engine.id
                );
            }
        }
    }

    #[test]
    fn all_ids_covers_every_engine_with_no_duplicates() {
        let all = all_ids();
        assert_eq!(all.len(), crate::ENGINES.len() * 6);
        let mut deduped = all.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(deduped.len(), all.len(), "ids must be unique");
    }

    #[test]
    fn every_description_names_what_the_function_does() {
        for engine in crate::ENGINES.iter() {
            for spec in catalog(engine.id) {
                assert!(
                    spec.description.len() > 30,
                    "{} has a stub description",
                    spec.function_id
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "no wire surface for unknown engine")]
    fn an_unknown_engine_has_no_surface_rather_than_an_improvised_one() {
        ids_for("ollama");
    }
}
