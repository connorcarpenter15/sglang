use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use prost::Message;
use pyo3::PyErr;
use pyo3::Python;
use pyo3::exceptions::{PyTypeError, PyValueError};
use tokio::sync::{Notify, mpsc::Receiver};
use tokio::time::{Duration, timeout};
use tokio_stream::Stream;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use crate::bridge::{PyBridge, ResponseChunk, ResponseData, TerminalError};
use crate::proto;
use crate::utils::{
    build_classify_dict, build_embed_dict, build_generate_dict, extract_model_path,
};

pub struct SglangServiceImpl {
    pub bridge: Arc<PyBridge>,
    pub response_timeout: Duration,
    pub max_message_size: usize,
}

type StreamResult<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;
pub const DEFAULT_RESPONSE_TIMEOUT_SECS: u64 = 300;

/// 64 MiB — leaves headroom for multimodal inputs and OpenAI JSON pass-through bodies,
/// well above tonic's 4 MiB decode default.
pub const DEFAULT_GRPC_MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

/// Resolve the legacy environment override used when the Python caller does
/// not supply the explicit `--grpc-max-message-size` value.
pub(crate) fn resolve_max_message_size() -> usize {
    match std::env::var("SGLANG_TONIC_PAYLOAD") {
        Ok(raw) => match raw.parse::<usize>() {
            Ok(n) if n > 0 => {
                tracing::info!(
                    bytes = n,
                    "Using SGLANG_TONIC_PAYLOAD override for gRPC max message size"
                );
                n
            }
            _ => {
                tracing::warn!(
                    value = %raw,
                    default = DEFAULT_GRPC_MAX_MESSAGE_SIZE,
                    "Ignoring invalid SGLANG_TONIC_PAYLOAD; using default"
                );
                DEFAULT_GRPC_MAX_MESSAGE_SIZE
            }
        },
        Err(_) => DEFAULT_GRPC_MAX_MESSAGE_SIZE,
    }
}

/// Classify a bridge `PyErr` into the right gRPC `Status`.
///
/// `PyValueError` / `PyTypeError` mean the client sent bad input — surface as
/// `INVALID_ARGUMENT` so callers can distinguish them from server failures.
/// Everything else (typically `PyRuntimeError`, but also Python tracebacks
/// from inside the tokenizer manager) maps to `INTERNAL`.
fn pyerr_to_status(err: PyErr, context: &str) -> Status {
    let is_client_error = Python::with_gil(|py| {
        err.is_instance_of::<PyValueError>(py) || err.is_instance_of::<PyTypeError>(py)
    });
    let msg = format!("{}: {}", context, err);
    if is_client_error {
        Status::invalid_argument(msg)
    } else {
        Status::internal(msg)
    }
}

async fn recv_chunk_with_timeout(
    receiver: &mut Receiver<ResponseChunk>,
    response_timeout: Duration,
    timeout_message: impl FnOnce() -> String,
) -> Result<Option<ResponseChunk>, Status> {
    timeout(response_timeout, receiver.recv())
        .await
        .map_err(|_| Status::deadline_exceeded(timeout_message()))
}

struct RequestAbortGuard {
    bridge: Arc<PyBridge>,
    rid: String,
    armed: bool,
}

impl RequestAbortGuard {
    fn new(bridge: Arc<PyBridge>, rid: impl Into<String>) -> Self {
        Self {
            bridge,
            rid: rid.into(),
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }

    fn abort_now(&mut self) {
        if self.armed {
            self.armed = false;
            spawn_abort(self.bridge.clone(), self.rid.clone());
        }
    }
}

impl Drop for RequestAbortGuard {
    fn drop(&mut self) {
        if self.armed {
            // Dropping a response stream means the client stopped consuming; propagate
            // cancellation to Python without blocking the Tokio worker.
            spawn_abort(self.bridge.clone(), self.rid.clone());
        }
    }
}

fn spawn_abort(bridge: Arc<PyBridge>, rid: String) {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            let _ = handle.spawn_blocking(move || {
                let _ = bridge.abort(&rid, false);
            });
        }
        Err(_) => {
            tracing::warn!(
                rid,
                "Skipping gRPC request abort because no Tokio runtime is available"
            );
        }
    }
}

async fn recv_terminal_chunk_for_request(
    bridge: &Arc<PyBridge>,
    rid: &str,
    receiver: &mut Receiver<ResponseChunk>,
    response_timeout: Duration,
) -> Result<ResponseChunk, Status> {
    let mut abort_guard = RequestAbortGuard::new(bridge.clone(), rid.to_string());

    match recv_chunk_with_timeout(receiver, response_timeout, || {
        format!("Request timed out after {}s", response_timeout.as_secs())
    })
    .await
    {
        Ok(Some(ResponseChunk::Data(_))) => {
            tracing::warn!(
                rid,
                "Unary gRPC response received non-terminal Data chunk; expected Finished"
            );
            abort_guard.abort_now();
            Err(Status::internal(
                "Unary response protocol violation: expected Finished, got Data",
            ))
        }
        Ok(Some(chunk @ (ResponseChunk::Finished(_) | ResponseChunk::Error(_)))) => {
            abort_guard.disarm();
            Ok(chunk)
        }
        Ok(None) => {
            let (status, should_abort) = closed_stream_status(bridge, rid);
            if should_abort {
                abort_guard.abort_now();
            } else {
                abort_guard.disarm();
            }
            Err(status)
        }
        Err(status) => {
            if status.code() == tonic::Code::DeadlineExceeded {
                abort_guard.abort_now();
            } else {
                abort_guard.disarm();
            }
            Err(status)
        }
    }
}

fn closed_stream_status(bridge: &Arc<PyBridge>, rid: &str) -> (Status, bool) {
    if let Some(error) = bridge.take_terminal_error(rid) {
        (terminal_error_status(error), false)
    } else {
        (
            Status::internal("gRPC response stream closed before a terminal response"),
            true,
        )
    }
}

fn terminal_error_status(error: TerminalError) -> Status {
    let message = error.message();
    match error {
        TerminalError::ChannelFull { .. } => Status::resource_exhausted(message),
        TerminalError::ClientDisconnected { .. } | TerminalError::Aborted { .. } => {
            Status::cancelled(message)
        }
    }
}

fn openai_status_code(meta_info: &HashMap<String, String>, default: i32) -> i32 {
    meta_info
        .get("status_code")
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(default)
}

fn meta_value(meta: &HashMap<String, String>, key: &str) -> Option<serde_json::Value> {
    meta.get(key)
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
}

fn json_to_prost_value(value: serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(value) => Kind::BoolValue(value),
        serde_json::Value::Number(value) => Kind::NumberValue(value.as_f64().unwrap_or_default()),
        serde_json::Value::String(value) => Kind::StringValue(value),
        serde_json::Value::Array(values) => Kind::ListValue(prost_types::ListValue {
            values: values.into_iter().map(json_to_prost_value).collect(),
        }),
        serde_json::Value::Object(fields) => Kind::StructValue(prost_types::Struct {
            fields: fields
                .into_iter()
                .map(|(key, value)| (key, json_to_prost_value(value)))
                .collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

fn json_object_to_struct(value: serde_json::Value) -> Option<prost_types::Struct> {
    let serde_json::Value::Object(fields) = value else {
        return None;
    };
    Some(prost_types::Struct {
        fields: fields
            .into_iter()
            .map(|(key, value)| (key, json_to_prost_value(value)))
            .collect(),
    })
}

fn prost_value_to_json(value: prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match value.kind {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(value)) => serde_json::Value::Bool(value),
        Some(Kind::NumberValue(value)) => serde_json::json!(value),
        Some(Kind::StringValue(value)) => serde_json::Value::String(value),
        Some(Kind::ListValue(value)) => {
            serde_json::Value::Array(value.values.into_iter().map(prost_value_to_json).collect())
        }
        Some(Kind::StructValue(value)) => prost_struct_to_json(value),
    }
}

fn prost_struct_to_json(value: prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        value
            .fields
            .into_iter()
            .map(|(key, value)| (key, prost_value_to_json(value)))
            .collect(),
    )
}

fn meta_as_struct(meta: &HashMap<String, String>) -> Option<prost_types::Struct> {
    let fields = meta
        .iter()
        .filter_map(|(key, raw)| {
            serde_json::from_str(raw)
                .ok()
                .map(|value| (key.clone(), json_to_prost_value(value)))
        })
        .collect();
    Some(prost_types::Struct { fields })
}

fn meta_u64(meta: &HashMap<String, String>, keys: &[&str]) -> u64 {
    keys.iter()
        .find_map(|key| meta_value(meta, key).and_then(|value| value.as_u64()))
        .unwrap_or_default()
}

fn logprob_entry(
    selected: &serde_json::Value,
    top: Option<&serde_json::Value>,
) -> Option<proto::TokenLogprob> {
    let parts = selected.as_array()?;
    let logprob = parts.first()?.as_f64()? as f32;
    let token_id = i32::try_from(parts.get(1)?.as_i64()?).ok()?;
    let text = parts
        .get(2)
        .and_then(|value| value.as_str())
        .map(str::to_owned);
    let top_logprobs = top
        .and_then(|value| value.as_array())
        .map(|entries| {
            entries
                .iter()
                .enumerate()
                .filter_map(|(rank, entry)| {
                    let entry = entry.as_array()?;
                    Some(proto::LogprobAlternative {
                        logprob: entry.first()?.as_f64()? as f32,
                        token_id: i32::try_from(entry.get(1)?.as_i64()?).ok()?,
                        text: entry
                            .get(2)
                            .and_then(|value| value.as_str())
                            .map(str::to_owned),
                        rank: i32::try_from(rank + 1).ok(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Some(proto::TokenLogprob {
        logprob,
        token_id,
        text,
        top_logprobs,
    })
}

fn logprobs_from_meta(
    meta: &HashMap<String, String>,
    offset: &mut usize,
) -> Option<proto::Logprobs> {
    let output = meta_value(meta, "output_token_logprobs")?;
    let output = output.as_array()?;
    let top = meta_value(meta, "output_top_logprobs");
    let top = top.as_ref().and_then(|value| value.as_array());
    let mapped_output = output
        .iter()
        .enumerate()
        .skip(*offset)
        .filter_map(|(index, selected)| logprob_entry(selected, top.and_then(|all| all.get(index))))
        .collect();
    *offset = output.len();

    let prompt = meta_value(meta, "input_token_logprobs")
        .and_then(|value| value.as_array().cloned())
        .map(|selected| {
            let top = meta_value(meta, "input_top_logprobs");
            let top = top.as_ref().and_then(|value| value.as_array());
            selected
                .iter()
                .enumerate()
                .filter_map(|(index, selected)| {
                    logprob_entry(selected, top.and_then(|all| all.get(index)))
                })
                .collect()
        })
        .unwrap_or_default();
    Some(proto::Logprobs {
        output: mapped_output,
        prompt,
    })
}

fn flatten_i32(value: &serde_json::Value, values: &mut Vec<i32>, shape: &mut Vec<i64>) {
    if let Some(array) = value.as_array() {
        shape.push(i64::try_from(array.len()).unwrap_or(i64::MAX));
        if let Some(first) = array.first() {
            flatten_i32(first, &mut Vec::new(), shape);
        }
        for item in array {
            if let Some(number) = item.as_i64().and_then(|number| i32::try_from(number).ok()) {
                values.push(number);
            } else if item.is_array() {
                let mut ignored_shape = Vec::new();
                flatten_i32(item, values, &mut ignored_shape);
            }
        }
    }
}

fn finish_from_meta(meta: &HashMap<String, String>) -> proto::generate_response::Terminal {
    use proto::generate_response::Terminal;
    let Some(value) = meta_value(meta, "finish_reason").filter(|value| !value.is_null()) else {
        return Terminal::Error(proto::GenerationError {
            code: proto::GenerationErrorCode::Internal as i32,
            message: "SGLang stream ended without finish_reason".into(),
            retryable: false,
        });
    };
    let finish_type = value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .or_else(|| value.as_str())
        .unwrap_or("error");
    if matches!(finish_type, "error") {
        return Terminal::Error(proto::GenerationError {
            code: proto::GenerationErrorCode::Internal as i32,
            message: value
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("SGLang generation failed")
                .to_string(),
            retryable: false,
        });
    }
    let reason = match finish_type {
        "stop" => proto::FinishReason::Stop,
        "length" => proto::FinishReason::Length,
        "abort" => proto::FinishReason::Abort,
        "cancelled" => proto::FinishReason::Cancelled,
        _ => proto::FinishReason::Unspecified,
    };
    let stop_reason = value.get("matched").and_then(|matched| {
        use proto::stop_reason::Reason;
        let reason = if let Some(value) = matched.as_str() {
            Some(Reason::MatchedString(value.to_string()))
        } else {
            matched
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .map(Reason::MatchedTokenId)
        }?;
        Some(proto::StopReason {
            reason: Some(reason),
        })
    });
    Terminal::Finish(proto::GenerationFinish {
        reason: reason as i32,
        stop_reason,
    })
}

fn has_finish_reason(meta: &HashMap<String, String>) -> bool {
    meta_value(meta, "finish_reason").is_some_and(|value| !value.is_null())
}

fn generate_response_from_data(
    data: ResponseData,
    finished: bool,
    prefill_handoff: Option<proto::DisaggregatedParams>,
    token_offset: &mut usize,
    text_offset: &mut usize,
    logprob_offset: &mut usize,
) -> proto::GenerateResponse {
    let output_ids = data.output_ids.unwrap_or_default();
    let delta_output_ids = if *token_offset <= output_ids.len() {
        output_ids[*token_offset..].to_vec()
    } else {
        // A core reset must not panic the public server. Restart the delta
        // baseline and let the caller observe the new sequence.
        output_ids.clone()
    };
    *token_offset = output_ids.len();
    let text = data.text.unwrap_or_default();
    let delta_text = text
        .get(*text_offset..)
        .map(str::to_owned)
        .unwrap_or_else(|| text.clone());
    *text_offset = text.len();
    let usage = {
        let prompt_tokens = meta_u64(&data.meta_info, &["prompt_tokens", "input_tokens"]);
        let completion_tokens = meta_u64(&data.meta_info, &["completion_tokens", "output_tokens"]);
        ((prompt_tokens + completion_tokens) > 0).then_some(proto::Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            cached_prompt_tokens: meta_u64(
                &data.meta_info,
                &["cached_prompt_tokens", "cached_tokens"],
            ),
        })
    };
    let routed_experts = meta_value(&data.meta_info, "routed_experts").map(|value| {
        let mut expert_ids = Vec::new();
        let mut shape = Vec::new();
        flatten_i32(&value, &mut expert_ids, &mut shape);
        proto::RoutedExpertMetadata {
            expert_ids,
            shape,
            start_position: meta_u64(&data.meta_info, &["routed_experts_start_len"])
                .try_into()
                .unwrap_or_default(),
        }
    });
    proto::GenerateResponse {
        choice_index: data.choice_index,
        delta_output_ids,
        delta_text: (!delta_text.is_empty()).then_some(delta_text),
        logprobs: logprobs_from_meta(&data.meta_info, logprob_offset),
        usage,
        routed_experts,
        engine_metadata: meta_as_struct(&data.meta_info),
        prefill_handoff: finished.then_some(prefill_handoff).flatten(),
        terminal: finished.then(|| finish_from_meta(&data.meta_info)),
    }
}

fn operation_response(value: &serde_json::Value) -> proto::OperationResponse {
    proto::OperationResponse {
        success: value
            .get("success")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        message: value
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string(),
    }
}

fn update_response(value: &serde_json::Value) -> proto::UpdateWeightsResponse {
    proto::UpdateWeightsResponse {
        success: value
            .get("success")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        message: value
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string(),
        num_paused_requests: value
            .get("num_paused_requests")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or_default(),
    }
}

fn lora_adapters(value: &serde_json::Value) -> Vec<proto::LoRaAdapter> {
    let entries: Vec<&serde_json::Value> = if let Some(array) = value.as_array() {
        array.iter().collect()
    } else if let Some(object) = value.as_object() {
        object.values().collect()
    } else {
        Vec::new()
    };
    entries
        .into_iter()
        .map(|adapter| proto::LoRaAdapter {
            name: adapter
                .get("name")
                .or_else(|| adapter.get("lora_name"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            path: adapter
                .get("path")
                .or_else(|| adapter.get("lora_path"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            id: adapter
                .get("id")
                .or_else(|| adapter.get("lora_id"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            pinned: adapter
                .get("pinned")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        })
        .collect()
}

fn behavior_json(
    behavior: Option<proto::UpdateBehavior>,
) -> serde_json::Map<String, serde_json::Value> {
    let behavior = behavior.unwrap_or_default();
    serde_json::Map::from_iter([
        (
            "flush_cache".into(),
            serde_json::json!(behavior.flush_cache),
        ),
        (
            "abort_all_requests".into(),
            serde_json::json!(behavior.abort_all_requests),
        ),
        (
            "weight_version".into(),
            serde_json::json!(behavior.weight_version),
        ),
        (
            "torch_empty_cache".into(),
            serde_json::json!(behavior.torch_empty_cache),
        ),
    ])
}

#[tonic::async_trait]
impl proto::sglang_service_server::SglangService for SglangServiceImpl {
    // --- SGLang-native generation ---

    type GenerateStream = StreamResult<proto::GenerateResponse>;

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        self.enforce_message_size(&req)?;
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        tracing::debug!(
            rid,
            max_new_tokens = ?req.sampling_params.as_ref().and_then(|params| params.max_new_tokens),
            min_new_tokens = ?req.sampling_params.as_ref().and_then(|params| params.min_new_tokens),
            ignore_eos = ?req.sampling_params.as_ref().and_then(|params| params.ignore_eos),
            choices = ?req.sampling_params.as_ref().and_then(|params| params.n),
            return_logprobs = ?req.logprob_options.as_ref().map(|options| options.return_logprobs),
            top_logprobs = ?req.logprob_options.as_ref().map(|options| options.top_logprobs),
            priority = req.priority,
            "received native Generate request"
        );
        let req_dict = build_generate_dict(&rid, &req).map_err(Status::invalid_argument)?;
        let prefill_handoff = req.disaggregated_params.clone();

        let mut receiver = self
            .bridge
            .submit_request(&rid, "generate", req_dict)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let bridge = self.bridge.clone();
        let rid_clone = rid.clone();
        let response_timeout = self.response_timeout;

        let stream = async_stream::stream! {
            let mut abort_guard = RequestAbortGuard::new(bridge.clone(), rid_clone.clone());
            let mut token_offsets: HashMap<i32, usize> = HashMap::new();
            let mut text_offsets: HashMap<i32, usize> = HashMap::new();
            let mut logprob_offsets: HashMap<i32, usize> = HashMap::new();
            let expected_choices = req.sampling_params.as_ref()
                .and_then(|params| params.n)
                .unwrap_or(1)
                .max(1) as usize;
            let mut terminal_choices = std::collections::HashSet::<i32>::new();
            loop {
                match recv_chunk_with_timeout(&mut receiver, response_timeout, || "Stream chunk timed out".to_string()).await {
                    Ok(Some(ResponseChunk::Data(data))) => {
                        let choice_finished = has_finish_reason(&data.meta_info);
                        let choice_index = data.choice_index;
                        if choice_index < 0 || choice_index as usize >= expected_choices {
                            abort_guard.abort_now();
                            yield Err(Status::internal(format!(
                                "SGLang returned choice index {choice_index} outside 0..{expected_choices}"
                            )));
                            break;
                        }
                        let token_offset = token_offsets.entry(choice_index).or_default();
                        let text_offset = text_offsets.entry(choice_index).or_default();
                        let logprob_offset = logprob_offsets.entry(choice_index).or_default();
                        yield Ok(generate_response_from_data(
                            data,
                            choice_finished,
                            choice_finished.then_some(prefill_handoff.clone()).flatten(),
                            token_offset,
                            text_offset,
                            logprob_offset,
                        ));
                        if choice_finished {
                            if !terminal_choices.insert(choice_index) {
                                abort_guard.abort_now();
                                yield Err(Status::internal(format!("duplicate terminal for choice {choice_index}")));
                                break;
                            }
                            if terminal_choices.len() == expected_choices {
                                abort_guard.disarm();
                                break;
                            }
                        }
                    }
                    Ok(Some(ResponseChunk::Finished(data))) => {
                        let choice_index = data.choice_index;
                        if choice_index < 0 || choice_index as usize >= expected_choices {
                            abort_guard.abort_now();
                            yield Err(Status::internal(format!(
                                "SGLang returned choice index {choice_index} outside 0..{expected_choices}"
                            )));
                            break;
                        }
                        let token_offset = token_offsets.entry(choice_index).or_default();
                        let text_offset = text_offsets.entry(choice_index).or_default();
                        let logprob_offset = logprob_offsets.entry(choice_index).or_default();
                        let response = generate_response_from_data(
                            data,
                            true,
                            prefill_handoff.clone(),
                            token_offset,
                            text_offset,
                            logprob_offset,
                        );
                        yield Ok(response);
                        if !terminal_choices.insert(choice_index) {
                            abort_guard.abort_now();
                            yield Err(Status::internal(format!("duplicate terminal for choice {choice_index}")));
                            break;
                        }
                        if terminal_choices.len() == expected_choices {
                            abort_guard.disarm();
                            break;
                        }
                    }
                    Ok(Some(ResponseChunk::Error(msg))) => {
                        abort_guard.abort_now();
                        yield Err(Status::internal(msg));
                        break;
                    }
                    Ok(None) => {
                        let (status, should_abort) = closed_stream_status(&bridge, &rid_clone);
                        if should_abort {
                            abort_guard.abort_now();
                        } else {
                            abort_guard.disarm();
                        }
                        yield Err(status);
                        break;
                    }
                    Err(status) => {
                        abort_guard.abort_now();
                        yield Err(status);
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    // --- SGLang-native embeddings ---

    async fn embed(
        &self,
        request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        let req = request.into_inner();
        self.enforce_message_size(&req)?;
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_embed_dict(&rid, &req).map_err(Status::invalid_argument)?;
        let encoding = req.encoding;

        let mut receiver = self
            .bridge
            .submit_request(&rid, "embed", req_dict)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &rid,
            &mut receiver,
            self.response_timeout,
        )
        .await?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                let vectors = data
                    .embeddings
                    .or_else(|| data.embedding.map(|value| vec![value]))
                    .unwrap_or_default();
                let embeddings = vectors
                    .into_iter()
                    .enumerate()
                    .map(|(index, values)| {
                        use proto::embedding::Data;
                        let data = if encoding == proto::EmbeddingEncoding::Base64 as i32 {
                            Data::PackedFloat32(
                                values
                                    .iter()
                                    .flat_map(|value| value.to_le_bytes())
                                    .collect(),
                            )
                        } else {
                            Data::FloatValues(proto::FloatEmbedding { values })
                        };
                        proto::Embedding {
                            index: i32::try_from(index).unwrap_or(i32::MAX),
                            data: Some(data),
                        }
                    })
                    .collect();
                Ok(Response::new(proto::EmbedResponse {
                    embeddings,
                    usage: Some(proto::Usage {
                        prompt_tokens: meta_u64(&data.meta_info, &["prompt_tokens"]),
                        completion_tokens: 0,
                        total_tokens: meta_u64(&data.meta_info, &["prompt_tokens"]),
                        cached_prompt_tokens: meta_u64(&data.meta_info, &["cached_tokens"]),
                    }),
                }))
            }
            ResponseChunk::Error(msg) => Err(Status::internal(msg)),
        }
    }

    async fn media_generate(
        &self,
        request: Request<proto::MediaGenerateRequest>,
    ) -> Result<Response<proto::MediaGenerateResponse>, Status> {
        let request = request.into_inner();
        self.enforce_message_size(&request)?;
        let body = prost_struct_to_json(
            request
                .request
                .ok_or_else(|| Status::invalid_argument("MediaGenerate requires request"))?,
        );
        let json_body = serde_json::to_vec(&body)
            .map_err(|err| Status::invalid_argument(format!("Invalid media request: {err}")))?;
        let rid = uuid::Uuid::new_v4().to_string();
        let mut receiver = self
            .bridge
            .submit_media(&rid, &json_body, &request.trace_headers)
            .map_err(|err| pyerr_to_status(err, "Failed to submit media request"))?;
        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &rid,
            &mut receiver,
            self.response_timeout,
        )
        .await?;
        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                let value = serde_json::from_slice(data.json_bytes.as_deref().unwrap_or_default())
                    .map_err(|err| Status::internal(format!("Invalid media response: {err}")))?;
                Ok(Response::new(proto::MediaGenerateResponse {
                    response: json_object_to_struct(value),
                    status_code: openai_status_code(&data.meta_info, 200),
                }))
            }
            ResponseChunk::Error(message) => Err(Status::internal(message)),
        }
    }

    // --- SGLang-native RPCs: Classify ---

    async fn classify(
        &self,
        request: Request<proto::ClassifyRequest>,
    ) -> Result<Response<proto::ClassifyResponse>, Status> {
        let req = request.into_inner();
        self.enforce_message_size(&req)?;
        if req.text.is_empty() && req.input_ids.is_empty() {
            return Err(Status::invalid_argument(
                "Classify requires either text or input_ids",
            ));
        }
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_classify_dict(&rid, &req);

        let mut receiver = self
            .bridge
            .submit_request(&rid, "embed", req_dict)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &rid,
            &mut receiver,
            self.response_timeout,
        )
        .await?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                Ok(Response::new(proto::ClassifyResponse {
                    embedding: data.embedding.unwrap_or_default(),
                    meta_info: data.meta_info,
                }))
            }
            ResponseChunk::Error(msg) => Err(Status::internal(msg)),
        }
    }

    // --- SGLang-native RPCs: Tokenize / Detokenize (Rust-native with fallback) ---

    async fn tokenize(
        &self,
        request: Request<proto::TokenizeRequest>,
    ) -> Result<Response<proto::TokenizeResponse>, Status> {
        let req = request.into_inner();
        let add_special = req.add_special_tokens.unwrap_or(true);

        // Try Rust-native tokenizer first (no GIL)
        if let Some(tok) = self.bridge.rust_tokenizer() {
            let tokens = tok
                .encode(&req.text, add_special)
                .map_err(Status::internal)?;
            let count = tokens.len() as i32;
            return Ok(Response::new(proto::TokenizeResponse {
                tokens: tokens.iter().map(|&t| t as i32).collect(),
                count,
                max_model_len: self.bridge.context_len(),
                input_text: req.text,
            }));
        }

        // Fallback to Python
        let json_str = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            let text = req.text.clone();
            move || bridge.tokenize_py(&text, add_special)
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Tokenize failed"))?;

        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::TokenizeResponse {
            tokens: v["tokens"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_i64().map(|n| n as i32))
                        .collect()
                })
                .unwrap_or_default(),
            count: v["count"].as_i64().unwrap_or(0) as i32,
            max_model_len: self.bridge.context_len(),
            input_text: req.text,
        }))
    }

    async fn detokenize(
        &self,
        request: Request<proto::DetokenizeRequest>,
    ) -> Result<Response<proto::DetokenizeResponse>, Status> {
        let req = request.into_inner();
        if req.tokens.iter().any(|&token| token < 0) {
            return Err(Status::invalid_argument(
                "Detokenize tokens must be non-negative",
            ));
        }

        // Try Rust-native tokenizer first (no GIL)
        if let Some(tok) = self.bridge.rust_tokenizer() {
            let ids: Vec<u32> = req.tokens.iter().map(|&t| t as u32).collect();
            let text = tok.decode(&ids, true).map_err(Status::internal)?;
            return Ok(Response::new(proto::DetokenizeResponse { text }));
        }

        // Fallback to Python
        let json_str = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            let tokens = req.tokens;
            move || bridge.detokenize_py(tokens)
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Detokenize failed"))?;

        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::DetokenizeResponse {
            text: v["text"].as_str().unwrap_or("").to_string(),
        }))
    }

    // --- SGLang-native RPCs: Info / control ---

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        let healthy = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.health_check()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Health check failed"))?;

        Ok(Response::new(proto::HealthCheckResponse { healthy }))
    }

    async fn get_model_info(
        &self,
        _request: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        let json_info = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.get_model_info()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Failed to get model info"))?;

        Ok(Response::new(proto::GetModelInfoResponse {
            model_path: extract_model_path(&json_info),
            json_info,
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        let json_info = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.get_server_info()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Failed to get server info"))?;
        let mut info: serde_json::Value = serde_json::from_str(&json_info)
            .map_err(|err| Status::internal(format!("Invalid server info: {err}")))?;
        let object = info
            .as_object_mut()
            .ok_or_else(|| Status::internal("SGLang server info must be a JSON object"))?;
        object.insert(
            "grpc_protocol_revision".into(),
            serde_json::Value::String(crate::PROTOCOL_REVISION.into()),
        );
        object.insert(
            "grpc_descriptor_sha256".into(),
            serde_json::Value::String(crate::descriptor_sha256()),
        );
        let json_info = serde_json::to_string(&info)
            .map_err(|err| Status::internal(format!("Encode server info: {err}")))?;

        Ok(Response::new(proto::GetServerInfoResponse { json_info }))
    }

    async fn get_runtime_info(
        &self,
        _request: Request<proto::GetRuntimeInfoRequest>,
    ) -> Result<Response<proto::GetRuntimeInfoResponse>, Status> {
        let bridge = self.bridge.clone();
        let (model_json, server_json) = tokio::task::spawn_blocking(move || {
            Ok::<_, PyErr>((bridge.get_model_info()?, bridge.get_server_info()?))
        })
        .await
        .map_err(|err| Status::internal(format!("Task join error: {err}")))?
        .map_err(|err| pyerr_to_status(err, "Failed to get runtime info"))?;
        let model: serde_json::Value = serde_json::from_str(&model_json)
            .map_err(|err| Status::internal(format!("Invalid model info: {err}")))?;
        let server: serde_json::Value = serde_json::from_str(&server_json)
            .map_err(|err| Status::internal(format!("Invalid server info: {err}")))?;

        let string = |value: &serde_json::Value, key: &str| {
            value
                .get(key)
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        };
        let number = |value: &serde_json::Value, key: &str| {
            value
                .get(key)
                .and_then(|value| {
                    value
                        .as_u64()
                        .or_else(|| value.as_i64().and_then(|v| u64::try_from(v).ok()))
                })
                .unwrap_or_default()
        };
        let model_path = string(&model, "model_path").unwrap_or_default();
        let tokenizer_path = string(&model, "tokenizer_path").unwrap_or_else(|| model_path.clone());
        let served_model_name = server.get("served_model_name").and_then(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .or_else(|| value.as_array()?.first()?.as_str().map(str::to_owned))
        });
        let runtime_kind = match string(&server, "runtime_kind").as_deref() {
            Some("embedding") => proto::RuntimeKind::Embedding,
            Some("image") => proto::RuntimeKind::Image,
            Some("video") => proto::RuntimeKind::Video,
            Some("llm") => proto::RuntimeKind::Llm,
            Some(other) => {
                return Err(Status::failed_precondition(format!(
                    "Unknown runtime_kind reported by SGLang: {other}"
                )));
            }
            None if model
                .get("is_generation")
                .and_then(serde_json::Value::as_bool)
                == Some(false) =>
            {
                proto::RuntimeKind::Embedding
            }
            None => proto::RuntimeKind::Llm,
        };
        let worker_role = match string(&server, "disaggregation_mode").as_deref() {
            Some("prefill") => proto::WorkerRole::Prefill,
            Some("decode") => proto::WorkerRole::Decode,
            _ => proto::WorkerRole::Aggregated,
        };
        let dp_size = number(&server, "dp_size").max(1) as u32;
        let local_dp_size = number(&server, "local_dp_size")
            .max(1)
            .min(u64::from(dp_size)) as u32;
        let local_start_rank = number(&server, "local_dp_rank_start") as u32;
        let bootstrap_port = number(&server, "disaggregation_bootstrap_port");
        let bootstrap = (worker_role == proto::WorkerRole::Prefill && bootstrap_port > 0)
            .then_some(proto::DisaggregatedParams {
                bootstrap_host: string(&server, "host").unwrap_or_else(|| "127.0.0.1".into()),
                bootstrap_port: i32::try_from(bootstrap_port).unwrap_or_default(),
                bootstrap_room: 0,
                prefill_dp_rank: None,
                bootstrap_pair_key: None,
                decode_tp_size: None,
            });

        let mut observability = Vec::new();
        let metrics_port = number(&server, "metrics_http_port").max(number(&server, "port"));
        let kv_config = server.get("kv_events").filter(|value| value.is_object());
        let kv_host = kv_config
            .and_then(|value| string(value, "endpoint_host"))
            .filter(|host| !matches!(host.as_str(), "*" | "0.0.0.0" | "::"))
            .unwrap_or_else(|| "127.0.0.1".into());
        let kv_port_base = kv_config
            .map(|value| number(value, "endpoint_port_base"))
            .unwrap_or_default();
        for dp_rank in 0..dp_size {
            observability.push(proto::ObservabilityEndpoint {
                dp_rank,
                metrics_url: (metrics_port > 0)
                    .then(|| format!("http://127.0.0.1:{metrics_port}/metrics")),
                kv_events_zmq_endpoint: (kv_port_base > 0)
                    .then(|| format!("tcp://{kv_host}:{}", kv_port_base + u64::from(dp_rank))),
                kv_events_topic: kv_config.and_then(|value| string(value, "topic")),
            });
        }

        Ok(Response::new(proto::GetRuntimeInfoResponse {
            model_path,
            tokenizer_path,
            served_model_name,
            runtime_kind: runtime_kind as i32,
            worker_role: worker_role as i32,
            capacity: Some(proto::RuntimeCapacity {
                max_context_length: u64::try_from(self.bridge.context_len()).unwrap_or_default(),
                max_running_requests: number(&server, "max_running_requests"),
                max_total_tokens: number(&server, "max_total_num_tokens"),
                kv_block_size: number(&server, "page_size"),
                total_kv_blocks: number(&server, "total_num_tokens")
                    .checked_div(number(&server, "page_size").max(1))
                    .unwrap_or_default(),
                max_lora_adapters: u32::try_from(number(&server, "max_loras_per_batch"))
                    .unwrap_or_default(),
            }),
            dp_topology: Some(proto::DataParallelTopology {
                start_rank: 0,
                size: dp_size,
                local_start_rank,
                local_size: local_dp_size,
            }),
            bootstrap,
            observability,
            protocol_revision: crate::PROTOCOL_REVISION.into(),
            descriptor_sha256: crate::descriptor_sha256(),
            reasoning_parser: string(&server, "reasoning_parser"),
            tool_call_parser: string(&server, "tool_call_parser"),
            weight_version: string(&model, "weight_version"),
        }))
    }

    async fn list_models(
        &self,
        _request: Request<proto::ListModelsRequest>,
    ) -> Result<Response<proto::ListModelsResponse>, Status> {
        let json_str = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.list_models()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Failed to list models"))?;

        let models_arr: Vec<serde_json::Value> = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse models JSON: {}", e)))?;

        let models = models_arr
            .iter()
            .map(|m| proto::ModelCard {
                id: m["id"].as_str().unwrap_or("").to_string(),
                root: m["root"].as_str().unwrap_or("").to_string(),
                parent: m.get("parent").and_then(|v| v.as_str()).map(String::from),
                max_model_len: m
                    .get("max_model_len")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32),
            })
            .collect();

        Ok(Response::new(proto::ListModelsResponse { models }))
    }

    async fn get_load(
        &self,
        request: Request<proto::GetLoadRequest>,
    ) -> Result<Response<proto::GetLoadResponse>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();
        let receiver = self
            .bridge
            .submit_get_load(&rid, req.dp_rank)
            .map_err(|e| pyerr_to_status(e, "Failed to get load"))?;

        let json_info =
            recv_json_response(&self.bridge, &rid, receiver, self.response_timeout).await?;
        Ok(Response::new(proto::GetLoadResponse { json_info }))
    }

    async fn abort(
        &self,
        request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        let req = request.into_inner();
        if !req.abort_all && req.rid.trim().is_empty() {
            return Err(Status::invalid_argument(
                "Abort requires a non-empty rid unless abort_all is true",
            ));
        }
        if req.abort_all {
            tracing::warn!(
                "Received abort_all over gRPC; this endpoint must only be exposed to trusted clients"
            );
        }
        self.bridge
            .abort(&req.rid, req.abort_all)
            .map_err(|e| pyerr_to_status(e, "Failed to abort"))?;

        Ok(Response::new(proto::AbortResponse { success: true }))
    }

    async fn flush_cache(
        &self,
        _request: Request<proto::FlushCacheRequest>,
    ) -> Result<Response<proto::FlushCacheResponse>, Status> {
        let rid = uuid::Uuid::new_v4().to_string();
        let receiver = self
            .bridge
            .submit_flush_cache(&rid)
            .map_err(|e| pyerr_to_status(e, "Failed to flush cache"))?;

        let json_str =
            recv_json_response(&self.bridge, &rid, receiver, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::FlushCacheResponse {
            success: v["success"].as_bool().unwrap_or(false),
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    async fn pause_generation(
        &self,
        request: Request<proto::PauseGenerationRequest>,
    ) -> Result<Response<proto::PauseGenerationResponse>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();
        let receiver = self
            .bridge
            .submit_pause_generation(&rid, &req.mode)
            .map_err(|e| pyerr_to_status(e, "Failed to pause generation"))?;

        let json_str =
            recv_json_response(&self.bridge, &rid, receiver, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::PauseGenerationResponse {
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    async fn continue_generation(
        &self,
        _request: Request<proto::ContinueGenerationRequest>,
    ) -> Result<Response<proto::ContinueGenerationResponse>, Status> {
        let rid = uuid::Uuid::new_v4().to_string();
        let receiver = self
            .bridge
            .submit_continue_generation(&rid)
            .map_err(|e| pyerr_to_status(e, "Failed to continue generation"))?;

        let json_str =
            recv_json_response(&self.bridge, &rid, receiver, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::ContinueGenerationResponse {
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    // --- OpenAI-compatible RPCs (JSON pass-through) ---

    type ChatCompleteStream = StreamResult<proto::OpenAiStreamChunk>;

    async fn chat_complete(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<Self::ChatCompleteStream>, Status> {
        self.openai_streaming_rpc(request, "submit_openai_chat")
            .await
    }

    type CompleteStream = StreamResult<proto::OpenAiStreamChunk>;

    async fn complete(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<Self::CompleteStream>, Status> {
        self.openai_streaming_rpc(request, "submit_openai_complete")
            .await
    }

    async fn open_ai_embed(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        self.openai_unary_rpc(request, "submit_openai_embed").await
    }

    async fn open_ai_classify(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        self.openai_unary_rpc(request, "submit_openai_classify")
            .await
    }

    async fn score(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        self.openai_unary_rpc(request, "submit_openai_score").await
    }

    async fn rerank(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        self.openai_unary_rpc(request, "submit_openai_rerank").await
    }

    // --- Admin RPCs ---

    async fn load_lo_ra(
        &self,
        request: Request<proto::LoadLoRaRequest>,
    ) -> Result<Response<proto::LoRaUpdateResponse>, Status> {
        let request = request.into_inner();
        let value = self
            .control_json(
                "load_lora",
                serde_json::json!({
                    "name": request.name,
                    "path": request.path,
                    "pinned": request.pinned,
                    "id": request.id,
                }),
            )
            .await?;
        Ok(Response::new(proto::LoRaUpdateResponse {
            success: value
                .get("success")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            error_message: value
                .get("error_message")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            loaded_adapters: lora_adapters(
                value
                    .get("loaded_adapters")
                    .unwrap_or(&serde_json::Value::Null),
            ),
        }))
    }

    async fn unload_lo_ra(
        &self,
        request: Request<proto::UnloadLoRaRequest>,
    ) -> Result<Response<proto::LoRaUpdateResponse>, Status> {
        let request = request.into_inner();
        let value = self
            .control_json(
                "unload_lora",
                serde_json::json!({"name": request.name, "id": request.id}),
            )
            .await?;
        Ok(Response::new(proto::LoRaUpdateResponse {
            success: value
                .get("success")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            error_message: value
                .get("error_message")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            loaded_adapters: lora_adapters(
                value
                    .get("loaded_adapters")
                    .unwrap_or(&serde_json::Value::Null),
            ),
        }))
    }

    async fn list_lo_r_as(
        &self,
        _request: Request<proto::ListLoRAsRequest>,
    ) -> Result<Response<proto::ListLoRAsResponse>, Status> {
        let value = self
            .control_json("list_loras", serde_json::json!({}))
            .await?;
        Ok(Response::new(proto::ListLoRAsResponse {
            adapters: lora_adapters(value.get("adapters").unwrap_or(&serde_json::Value::Null)),
        }))
    }

    async fn release_memory(
        &self,
        request: Request<proto::ReleaseMemoryRequest>,
    ) -> Result<Response<proto::OperationResponse>, Status> {
        let value = self
            .control_json(
                "release_memory",
                serde_json::json!({"tags": request.into_inner().tags}),
            )
            .await?;
        Ok(Response::new(operation_response(&value)))
    }

    async fn resume_memory(
        &self,
        request: Request<proto::ResumeMemoryRequest>,
    ) -> Result<Response<proto::OperationResponse>, Status> {
        let value = self
            .control_json(
                "resume_memory",
                serde_json::json!({"tags": request.into_inner().tags}),
            )
            .await?;
        Ok(Response::new(operation_response(&value)))
    }

    async fn start_profile(
        &self,
        request: Request<proto::StartProfileRequest>,
    ) -> Result<Response<proto::ProfileResponse>, Status> {
        let request = request.into_inner();
        let value = self
            .control_json(
                "start_profile",
                serde_json::json!({
                    "output_dir": request.output_dir,
                    "start_step": request.start_step,
                    "num_steps": request.num_steps,
                    "activities": request.activities,
                    "profile_by_stage": request.profile_by_stage,
                    "with_stack": request.with_stack,
                    "record_shapes": request.record_shapes,
                    "profile_id": request.profile_id,
                    "merge_profiles": request.merge_profiles,
                    "profile_prefix": request.profile_prefix,
                    "profile_stages": request.profile_stages,
                }),
            )
            .await?;
        let response = operation_response(&value);
        Ok(Response::new(proto::ProfileResponse {
            success: response.success,
            message: response.message,
        }))
    }

    async fn stop_profile(
        &self,
        request: Request<proto::StopProfileRequest>,
    ) -> Result<Response<proto::ProfileResponse>, Status> {
        let value = self
            .control_json(
                "stop_profile",
                serde_json::json!({"profile_id": request.into_inner().profile_id}),
            )
            .await?;
        let response = operation_response(&value);
        Ok(Response::new(proto::ProfileResponse {
            success: response.success,
            message: response.message,
        }))
    }

    async fn update_weights_from_disk(
        &self,
        request: Request<proto::UpdateWeightsFromDiskRequest>,
    ) -> Result<Response<proto::UpdateWeightsResponse>, Status> {
        let request = request.into_inner();
        let mut body = behavior_json(request.behavior);
        body.insert("model_path".into(), serde_json::json!(request.model_path));
        body.insert("load_format".into(), serde_json::json!(request.load_format));
        body.insert("is_async".into(), serde_json::json!(request.async_update));
        body.insert("keep_pause".into(), serde_json::json!(request.keep_pause));
        body.insert(
            "recapture_cuda_graph".into(),
            serde_json::json!(request.recapture_cuda_graph),
        );
        body.insert("token_step".into(), serde_json::json!(request.token_step));
        body.insert(
            "manifest".into(),
            request
                .manifest
                .map(prost_struct_to_json)
                .unwrap_or(serde_json::Value::Null),
        );
        let value = self
            .control_json("update_weights_from_disk", serde_json::Value::Object(body))
            .await?;
        Ok(Response::new(update_response(&value)))
    }

    async fn update_weights_from_tensor(
        &self,
        request: Request<proto::UpdateWeightsFromTensorRequest>,
    ) -> Result<Response<proto::UpdateWeightsResponse>, Status> {
        let request = request.into_inner();
        let mut body = behavior_json(request.behavior);
        body.insert(
            "serialized_named_tensors".into(),
            serde_json::json!(request.serialized_named_tensors),
        );
        body.insert("load_format".into(), serde_json::json!(request.load_format));
        body.insert(
            "disable_draft_model".into(),
            serde_json::json!(request.disable_draft_model),
        );
        let value = self
            .control_json(
                "update_weights_from_tensor",
                serde_json::Value::Object(body),
            )
            .await?;
        Ok(Response::new(update_response(&value)))
    }

    async fn update_weights_from_distributed(
        &self,
        request: Request<proto::UpdateWeightsFromDistributedRequest>,
    ) -> Result<Response<proto::UpdateWeightsResponse>, Status> {
        let request = request.into_inner();
        let mut body = behavior_json(request.behavior);
        body.insert(
            "names".into(),
            serde_json::json!(
                request
                    .tensors
                    .iter()
                    .map(|tensor| &tensor.name)
                    .collect::<Vec<_>>()
            ),
        );
        body.insert(
            "dtypes".into(),
            serde_json::json!(
                request
                    .tensors
                    .iter()
                    .map(|tensor| &tensor.dtype)
                    .collect::<Vec<_>>()
            ),
        );
        body.insert(
            "shapes".into(),
            serde_json::json!(
                request
                    .tensors
                    .iter()
                    .map(|tensor| &tensor.shape)
                    .collect::<Vec<_>>()
            ),
        );
        body.insert("group_name".into(), serde_json::json!(request.group_name));
        body.insert("load_format".into(), serde_json::json!(request.load_format));
        let value = self
            .control_json(
                "update_weights_from_distributed",
                serde_json::Value::Object(body),
            )
            .await?;
        Ok(Response::new(update_response(&value)))
    }

    async fn update_weights_from_ipc(
        &self,
        request: Request<proto::UpdateWeightsFromIpcRequest>,
    ) -> Result<Response<proto::UpdateWeightsResponse>, Status> {
        let request = request.into_inner();
        let mut body = behavior_json(request.behavior);
        body.remove("abort_all_requests");
        body.insert("zmq_handles".into(), serde_json::json!(request.zmq_handles));
        let value = self
            .control_json("update_weights_from_ipc", serde_json::Value::Object(body))
            .await?;
        Ok(Response::new(update_response(&value)))
    }

    async fn update_weight_version(
        &self,
        request: Request<proto::UpdateWeightVersionRequest>,
    ) -> Result<Response<proto::UpdateWeightsResponse>, Status> {
        let request = request.into_inner();
        let value = self
            .control_json(
                "update_weight_version",
                serde_json::json!({
                    "new_version": request.new_version,
                    "abort_all_requests": request.abort_all_requests,
                }),
            )
            .await?;
        Ok(Response::new(update_response(&value)))
    }
}

// Helper methods for OpenAI pass-through RPCs.
impl SglangServiceImpl {
    fn enforce_message_size<M: Message>(&self, message: &M) -> Result<(), Status> {
        let encoded = message.encoded_len();
        if encoded > self.max_message_size {
            return Err(Status::resource_exhausted(format!(
                "gRPC request is {encoded} bytes, exceeding the configured {}-byte ceiling; use a media URL or NIXL external-buffer transport for large media and embeddings",
                self.max_message_size
            )));
        }
        Ok(())
    }

    async fn control_json(
        &self,
        method: &str,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, Status> {
        let rid = uuid::Uuid::new_v4().to_string();
        let bytes = serde_json::to_vec(&body)
            .map_err(|err| Status::invalid_argument(format!("Invalid control request: {err}")))?;
        let receiver = self
            .bridge
            .submit_control_json(&rid, method, &bytes)
            .map_err(|err| pyerr_to_status(err, "Failed to submit control request"))?;
        let json = recv_json_response(&self.bridge, &rid, receiver, self.response_timeout).await?;
        serde_json::from_str(&json)
            .map_err(|err| Status::internal(format!("Invalid control response: {err}")))
    }

    async fn openai_streaming_rpc(
        &self,
        request: Request<proto::OpenAiRequest>,
        method_name: &str,
    ) -> Result<Response<StreamResult<proto::OpenAiStreamChunk>>, Status> {
        let req = request.into_inner();
        self.enforce_message_size(&req)?;
        let rid = uuid::Uuid::new_v4().to_string();

        let mut receiver = self
            .bridge
            .submit_openai(&rid, method_name, &req.json_body, &req.trace_headers)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let bridge = self.bridge.clone();
        let rid_clone = rid.clone();
        let response_timeout = self.response_timeout;

        let stream = async_stream::stream! {
            let mut abort_guard = RequestAbortGuard::new(bridge.clone(), rid_clone.clone());
            loop {
                match recv_chunk_with_timeout(&mut receiver, response_timeout, || "Stream chunk timed out".to_string()).await {
                    Ok(Some(ResponseChunk::Data(data))) => {
                        yield Ok(proto::OpenAiStreamChunk {
                            json_chunk: data.json_bytes.unwrap_or_default(),
                            finished: false,
                        });
                    }
                    Ok(Some(ResponseChunk::Finished(data))) => {
                        let bytes = data.json_bytes.unwrap_or_default();
                        abort_guard.disarm();
                        yield Ok(proto::OpenAiStreamChunk {
                            json_chunk: bytes,
                            finished: true,
                        });
                        break;
                    }
                    Ok(Some(ResponseChunk::Error(msg))) => {
                        abort_guard.disarm();
                        yield Err(Status::internal(msg));
                        break;
                    }
                    Ok(None) => {
                        let (status, should_abort) = closed_stream_status(&bridge, &rid_clone);
                        if should_abort {
                            abort_guard.abort_now();
                        } else {
                            abort_guard.disarm();
                        }
                        yield Err(status);
                        break;
                    }
                    Err(status) => {
                        abort_guard.abort_now();
                        yield Err(status);
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    async fn openai_unary_rpc(
        &self,
        request: Request<proto::OpenAiRequest>,
        method_name: &str,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        let req = request.into_inner();
        self.enforce_message_size(&req)?;
        let rid = uuid::Uuid::new_v4().to_string();

        let mut receiver = self
            .bridge
            .submit_openai(&rid, method_name, &req.json_body, &req.trace_headers)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &rid,
            &mut receiver,
            self.response_timeout,
        )
        .await?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                Ok(Response::new(proto::OpenAiResponse {
                    json_body: data.json_bytes.unwrap_or_default(),
                    status_code: openai_status_code(&data.meta_info, 200),
                }))
            }
            ResponseChunk::Error(msg) => {
                let error_json = serde_json::json!({"error": {"message": msg}});
                Ok(Response::new(proto::OpenAiResponse {
                    json_body: error_json.to_string().into_bytes(),
                    status_code: 500,
                }))
            }
        }
    }
}

/// Receive a single JSON response from the bridge channel.
async fn recv_json_response(
    bridge: &Arc<PyBridge>,
    rid: &str,
    mut receiver: Receiver<ResponseChunk>,
    response_timeout: Duration,
) -> Result<String, Status> {
    let chunk =
        recv_terminal_chunk_for_request(bridge, rid, &mut receiver, response_timeout).await?;

    match chunk {
        ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
            let bytes = data.json_bytes.unwrap_or_default();
            String::from_utf8(bytes)
                .map_err(|e| Status::internal(format!("Invalid UTF-8 in response: {}", e)))
        }
        ResponseChunk::Error(msg) => Err(Status::internal(msg)),
    }
}

/// Start the Tonic gRPC server on the given address.
//
// TODO(grpc-auth): this listener is currently unauthenticated. Before exposing
// it in any default deploy path, gate it with the same API-key / admin-key
// checks the HTTP server applies (see issue tracking gRPC auth parity).
pub async fn run_grpc_server(
    listener: std::net::TcpListener,
    bridge: Arc<PyBridge>,
    shutdown: Arc<Notify>,
    response_timeout: Duration,
    max_message_size: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = listener.local_addr()?;
    let listener = tokio::net::TcpListener::from_std(listener)?;
    let service = SglangServiceImpl {
        bridge,
        response_timeout,
        max_message_size,
    };

    // Decode with a small headroom so the service can return a useful
    // RESOURCE_EXHAUSTED message instead of tonic's generic codec error.
    let transport_decode_limit = max_message_size.saturating_add(8 * 1024 * 1024);
    let svc = proto::sglang_service_server::SglangServiceServer::new(service)
        .max_decoding_message_size(transport_decode_limit)
        .max_encoding_message_size(max_message_size);

    tracing::info!("gRPC server listening on {}", addr);

    tonic::transport::Server::builder()
        .add_service(svc)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
            shutdown.notified().await;
            tracing::info!("gRPC server shutting down");
        })
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests;
