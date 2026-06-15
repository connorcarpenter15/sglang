// SPDX-License-Identifier: Apache-2.0
//
// Conversion between the OpenEngine v1 wire types and SGLang's `GenerateReqInput`
// request dicts / streamed chunk metadata. Mirrors the field mapping in
// `crate::utils::request_utils::build_generate_dict`, but sourced from the
// vendor-neutral OpenEngine `GenerateRequest` instead of SGLang's native proto.

use std::collections::HashMap;

use crate::openengine_proto as pb;

/// SGLang disaggregation role for a request, derived from the engine's
/// discovered role. Drives token capping (prefill) and bootstrap plumbing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Aggregated,
    Prefill,
    Decode,
}

impl Role {
    pub fn from_proto(role: i32) -> Self {
        match pb::EngineRole::try_from(role).unwrap_or(pb::EngineRole::Unspecified) {
            pb::EngineRole::Prefill => Role::Prefill,
            pb::EngineRole::Decode => Role::Decode,
            _ => Role::Aggregated,
        }
    }
}

/// Build the SGLang `GenerateReqInput` dict from an OpenEngine `GenerateRequest`.
///
/// `role` controls disaggregation behavior:
/// - `Prefill`: cap `max_new_tokens` to 1 (prefill only populates the KV cache).
/// - `Decode`: lift the prefill peer's `(bootstrap_host, bootstrap_port,
///   bootstrap_room)` triple out of `kv_session.attributes_struct` into the
///   request dict so SGLang's decode worker connects to the prefill bootstrap.
pub fn build_generate_dict(
    rid: &str,
    req: &pb::GenerateRequest,
    role: Role,
) -> HashMap<String, serde_json::Value> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));

    match &req.input {
        Some(pb::generate_request::Input::TokenIds(t)) => {
            d.insert("input_ids".into(), serde_json::json!(t.ids));
        }
        Some(pb::generate_request::Input::Prompt(p)) => {
            d.insert("text".into(), serde_json::json!(p));
        }
        None => {
            // No input — leave both unset; the engine rejects it.
        }
    }

    d.insert(
        "sampling_params".into(),
        sampling_params_to_map(req, role),
    );
    d.insert("stream".into(), serde_json::json!(true));

    // Multimodal: map OpenEngine MediaItem (url / data: URI) -> SGLang image_data.
    // The sidecar runs in URL-passthrough mode, so items are url/data_uri strings;
    // raw_bytes is not emitted by the sidecar and is skipped here.
    let mut images: Vec<serde_json::Value> = Vec::new();
    for m in &req.media {
        match &m.source {
            Some(pb::media_item::Source::Url(u)) => images.push(serde_json::json!(u)),
            Some(pb::media_item::Source::DataUri(d)) => images.push(serde_json::json!(d)),
            _ => {}
        }
    }
    if !images.is_empty() {
        d.insert("image_data".into(), serde_json::json!(images));
    }

    // KV-aware routing affinity: pin the request to the rank the router chose
    // from the indexed prefix (SGLang `routed_dp_rank`).
    if let Some(rank) = req.data_parallel_rank {
        d.insert("routed_dp_rank".into(), serde_json::json!(rank));
    }

    // Disaggregation: forward the bootstrap triple from kv_session to SGLang.
    // SGLang's Bootstrap path sends the router-assigned room to BOTH prefill and
    // decode; the Completed path sends it to decode only. Apply for any role.
    if let Some(bootstrap) = decode_bootstrap_from_session(req.kv_session.as_ref()) {
        for (k, v) in bootstrap {
            d.insert(k, v);
        }
    }

    d
}

/// Map OpenEngine sampling + stop conditions into SGLang's `sampling_params`.
fn sampling_params_to_map(req: &pb::GenerateRequest, role: Role) -> serde_json::Value {
    let mut map = serde_json::Map::new();

    if let Some(s) = &req.sampling {
        map.insert("temperature".into(), serde_json::json!(s.temperature));
        if s.top_p > 0.0 {
            map.insert("top_p".into(), serde_json::json!(s.top_p));
        }
        if s.top_k > 0 {
            map.insert("top_k".into(), serde_json::json!(s.top_k));
        }
        if s.frequency_penalty != 0.0 {
            map.insert("frequency_penalty".into(), serde_json::json!(s.frequency_penalty));
        }
        if s.presence_penalty != 0.0 {
            map.insert("presence_penalty".into(), serde_json::json!(s.presence_penalty));
        }
        if s.ignore_eos {
            map.insert("ignore_eos".into(), serde_json::json!(true));
        }
        // Prefill only needs to build the KV cache: cap to a single token.
        let max_tokens = if role == Role::Prefill {
            1
        } else {
            s.max_tokens
        };
        if max_tokens > 0 {
            map.insert("max_new_tokens".into(), serde_json::json!(max_tokens));
        }
    } else if role == Role::Prefill {
        map.insert("max_new_tokens".into(), serde_json::json!(1));
    }

    // Stop conditions: text strings and token ids land in separate SGLang fields.
    let mut stop_text: Vec<String> = Vec::new();
    let mut stop_token_ids: Vec<u32> = Vec::new();
    for sc in &req.stop {
        match &sc.condition {
            Some(pb::stop_condition::Condition::StopText(t)) => stop_text.push(t.clone()),
            Some(pb::stop_condition::Condition::StopTokenId(id)) => stop_token_ids.push(*id),
            None => {}
        }
    }
    if !stop_text.is_empty() {
        map.insert("stop".into(), serde_json::json!(stop_text));
    }
    if !stop_token_ids.is_empty() {
        map.insert("stop_token_ids".into(), serde_json::json!(stop_token_ids));
    }

    serde_json::Value::Object(map)
}

/// Extract the SGLang decode bootstrap triple from a decode request's
/// `kv_session.attributes_struct`. The prefill peer wrote
/// `{bootstrap_host, bootstrap_port, bootstrap_room}` there; the sidecar
/// relayed it verbatim. Returns the SGLang request-dict keys to merge in.
fn decode_bootstrap_from_session(
    session: Option<&pb::KvSessionRef>,
) -> Option<Vec<(String, serde_json::Value)>> {
    let session = session?;
    let attrs = session.attributes_struct.as_ref()?;
    let host = struct_get_str(attrs, "bootstrap_host")?;
    let port = struct_get_i64(attrs, "bootstrap_port")?;
    let room = struct_get_i64(attrs, "bootstrap_room")?;
    Some(vec![
        ("bootstrap_host".into(), serde_json::json!(host)),
        ("bootstrap_port".into(), serde_json::json!(port)),
        ("bootstrap_room".into(), serde_json::json!(room)),
    ])
}

fn struct_get_str(s: &prost_types::Struct, key: &str) -> Option<String> {
    match s.fields.get(key).and_then(|v| v.kind.as_ref())? {
        prost_types::value::Kind::StringValue(s) => Some(s.clone()),
        _ => None,
    }
}

fn struct_get_i64(s: &prost_types::Struct, key: &str) -> Option<i64> {
    match s.fields.get(key).and_then(|v| v.kind.as_ref())? {
        prost_types::value::Kind::NumberValue(n) => Some(*n as i64),
        prost_types::value::Kind::StringValue(s) => s.parse().ok(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Streamed chunk meta_info parsing
// ---------------------------------------------------------------------------

/// Usage counts parsed off a terminal chunk's `meta_info`. Bridge values are
/// JSON-encoded strings (see `bridge::extract_meta_info`), so each is decoded.
#[derive(Default, Debug, Clone, Copy)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl Usage {
    pub fn total(&self) -> u32 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

/// Parse prompt/completion token counts from a chunk's `meta_info`.
pub fn usage_from_meta(meta: &HashMap<String, String>) -> Usage {
    Usage {
        prompt_tokens: meta_u32(meta, "prompt_tokens"),
        completion_tokens: meta_u32(meta, "completion_tokens"),
    }
}

fn meta_u32(meta: &HashMap<String, String>, key: &str) -> u32 {
    meta.get(key)
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}

/// Build a `PrefillReady` KV session from a prefill terminal's
/// `disaggregated_params` (the SGLang `{bootstrap_host, bootstrap_port,
/// bootstrap_room}` triple). The sidecar relays `attributes_struct` opaquely to
/// the decode peer, where `decode_bootstrap_from_session` reads it back.
pub fn prefill_kv_session(request_id: &str, disagg: &serde_json::Value) -> pb::KvSessionRef {
    let backend = disagg
        .get("transfer_backend")
        .and_then(|v| v.as_str())
        .unwrap_or("nixl")
        .to_string();
    pb::KvSessionRef {
        session_id: request_id.to_string(),
        transfer_backend: backend,
        endpoints: Vec::new(),
        dp_rank: 0,
        attributes: std::collections::HashMap::new(),
        attributes_struct: json_to_prost_struct(disagg),
    }
}

/// Convert a JSON object into a `google.protobuf.Struct` (None for non-objects).
fn json_to_prost_struct(value: &serde_json::Value) -> Option<prost_types::Struct> {
    match value {
        serde_json::Value::Object(map) => Some(prost_types::Struct {
            fields: map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_prost_value(v)))
                .collect(),
        }),
        _ => None,
    }
}

fn json_to_prost_value(value: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(b) => Kind::BoolValue(*b),
        serde_json::Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Kind::StringValue(s.clone()),
        serde_json::Value::Array(arr) => Kind::ListValue(prost_types::ListValue {
            values: arr.iter().map(json_to_prost_value).collect(),
        }),
        serde_json::Value::Object(map) => Kind::StructValue(prost_types::Struct {
            fields: map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_prost_value(v)))
                .collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

/// Extract the prefill `disaggregated_params` JSON from a terminal chunk's
/// `meta_info` (set by RuntimeHandle for prefill requests). None if absent.
pub fn disagg_params_from_meta(
    meta: &std::collections::HashMap<String, String>,
) -> Option<serde_json::Value> {
    let raw = meta.get("disaggregated_params")?;
    serde_json::from_str(raw).ok()
}

/// Map SGLang's `meta_info.finish_reason` to an OpenEngine `FinishReason`.
///
/// SGLang encodes finish_reason as an object `{"type": "stop"|"length"|"abort", …}`
/// (JSON-encoded into the meta_info string). Absent / unrecognized → STOP.
pub fn finish_reason_from_meta(meta: &HashMap<String, String>) -> pb::FinishReason {
    let Some(raw) = meta.get("finish_reason") else {
        return pb::FinishReason::Stop;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) else {
        return pb::FinishReason::Stop;
    };
    let kind = value
        .get("type")
        .and_then(|v| v.as_str())
        .or_else(|| value.as_str());
    match kind {
        Some("length") => pb::FinishReason::Length,
        Some("abort") => pb::FinishReason::Cancelled,
        _ => pb::FinishReason::Stop,
    }
}
