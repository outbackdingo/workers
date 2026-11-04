//! iii message/tool shapes -> OpenAI Chat Completions wire shapes.

pub mod messages;
pub mod tools;

// The tool-name codec is shared scaffolding; re-exported so this crate's
// wire mappers and its tests import it from one place.
pub use llm_router::provider_scaffold::names::{decode_tool_name, encode_tool_name};
