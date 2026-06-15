// SPDX-License-Identifier: Apache-2.0
//
// OpenEngine v1 gRPC server for SGLang.
//
// Bridges the vendor-neutral OpenEngine contract to the SGLang scheduler via the
// same `PyBridge` the native `SglangService` uses (`submit_request`,
// `get_model_info`, `get_server_info`, `submit_get_load`, `health_check`,
// `abort`). The Dynamo SGLang sidecar is the OpenEngine *client*; this is the
// *server* that runs inside the SGLang engine process.
//
// Discovery, not flags: engine identity, role, parallelism, and capacity are
// reported by `GetEngineInfo` / `GetModelInfo`, parsed from the scheduler's
// `get_server_info` / `get_model_info` JSON. The only transport input is the
// bind host/port (see `start_openengine_server` in `lib.rs`).

pub mod convert;

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use pyo3::PyErr;
use pyo3::Python;
use pyo3::exceptions::{PyTypeError, PyValueError};
use tokio::sync::mpsc::Receiver;
use tokio::time::timeout;
use tokio_stream::Stream;
use tonic::{Request, Response, Status};

use crate::bridge::{PyBridge, ResponseChunk, TerminalError};
use crate::openengine_proto as pb;
use convert::{Role, Usage};

type StreamResult<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

pub struct OpenEngineServiceImpl {
    pub bridge: Arc<PyBridge>,
    pub response_timeout: Duration,
    /// Disaggregation role discovered once at startup (see `start_openengine_server`).
    pub role: Role,
}

// ---------------------------------------------------------------------------
// Small shared helpers (mirrors the private ones in `server.rs`)
// ---------------------------------------------------------------------------

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

/// Best-effort abort propagated to Python without blocking the Tokio worker.
fn spawn_abort(bridge: Arc<PyBridge>, rid: String) {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        let _ = handle.spawn_blocking(move || {
            let _ = bridge.abort(&rid, false);
        });
    }
}

/// Aborts the in-flight request on drop unless disarmed. A dropped response
/// stream means the client (the sidecar) stopped consuming.
struct RequestAbortGuard {
    bridge: Arc<PyBridge>,
    rid: String,
    armed: bool,
}

impl RequestAbortGuard {
    fn new(bridge: Arc<PyBridge>, rid: String) -> Self {
        Self { bridge, rid, armed: true }
    }
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for RequestAbortGuard {
    fn drop(&mut self) {
        if self.armed {
            spawn_abort(self.bridge.clone(), self.rid.clone());
        }
    }
}

fn now_unix_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn json_u64(v: &serde_json::Value, key: &str) -> u64 {
    v.get(key).and_then(|x| x.as_u64()).unwrap_or(0)
}

fn json_str(v: &serde_json::Value, key: &str) -> String {
    v.get(key).and_then(|x| x.as_str()).unwrap_or("").to_string()
}

/// Parse a `tcp://host:port` ZMQ endpoint into a routable `KvEndpoint`.
fn parse_zmq_endpoint(endpoint: &str) -> Option<pb::KvEndpoint> {
    let rest = endpoint.split("://").nth(1).unwrap_or(endpoint);
    let (host, port) = rest.rsplit_once(':')?;
    let port: u32 = port.parse().ok()?;
    Some(pb::KvEndpoint {
        host: host.to_string(),
        port,
        protocol: "tcp".to_string(),
    })
}

/// Build a `pb::GenerateResponse` carrying a single token event.
fn token_response(request_id: &str, token_ids: Vec<u32>, text: String) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
            token_ids,
            text,
            logprobs: Vec::new(),
        })),
        usage: None,
    }
}

/// Build the terminal `pb::GenerateResponse` for a role.
///
/// Aggregated / decode terminate with `Finished`. Prefill emits `PrefillReady`
/// carrying the SGLang bootstrap triple so the decode peer can connect; that
/// path is wired with disaggregated serving and currently falls back to
/// `Finished` until the bootstrap room is surfaced here.
fn terminal_response(
    request_id: &str,
    reason: pb::FinishReason,
    usage: Usage,
) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(pb::generate_response::Event::Finished(pb::GenerationFinished {
            reason: reason as i32,
            message: String::new(),
        })),
        usage: Some(pb::Usage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total(),
        }),
    }
}

impl OpenEngineServiceImpl {
    /// Fetch + parse the scheduler's server-info JSON (best effort).
    fn server_info_json(&self) -> Result<serde_json::Value, Status> {
        let raw = self
            .bridge
            .get_server_info()
            .map_err(|e| pyerr_to_status(e, "get_server_info"))?;
        serde_json::from_str(&raw)
            .map_err(|e| Status::internal(format!("parse server_info: {e}")))
    }

    fn model_info_json(&self) -> Result<serde_json::Value, Status> {
        let raw = self
            .bridge
            .get_model_info()
            .map_err(|e| pyerr_to_status(e, "get_model_info"))?;
        serde_json::from_str(&raw)
            .map_err(|e| Status::internal(format!("parse model_info: {e}")))
    }
}

#[tonic::async_trait]
impl pb::open_engine_server::OpenEngine for OpenEngineServiceImpl {
    type GenerateStream = StreamResult<pb::GenerateResponse>;
    type DrainStream = StreamResult<pb::DrainResponse>;
    type SubscribeKvEventsStream = StreamResult<pb::KvEventBatch>;
    type SubscribeRuntimeEventsStream = StreamResult<pb::RuntimeEvent>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        let rid = if req.request_id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            req.request_id.clone()
        };
        let req_dict = convert::build_generate_dict(&rid, &req, self.role);

        let mut receiver = self
            .bridge
            .submit_request(&rid, "generate", req_dict)
            .map_err(|e| pyerr_to_status(e, "submit_request"))?;

        let bridge = self.bridge.clone();
        let rid_clone = rid.clone();
        let response_timeout = self.response_timeout;
        let role = self.role;

        let stream = async_stream::stream! {
            let mut abort_guard = RequestAbortGuard::new(bridge.clone(), rid_clone.clone());
            loop {
                let recv = timeout(response_timeout, receiver.recv()).await;
                match recv {
                    Ok(Some(ResponseChunk::Data(data))) => {
                        let toks: Vec<u32> = data
                            .output_ids
                            .unwrap_or_default()
                            .into_iter()
                            .map(|t| t as u32)
                            .collect();
                        yield Ok(token_response(&rid_clone, toks, data.text.unwrap_or_default()));
                    }
                    Ok(Some(ResponseChunk::Finished(data))) => {
                        abort_guard.disarm();
                        // Prefill role: the request only populates the KV cache and
                        // hands it off; suppress generated tokens and emit a terminal
                        // PrefillReady carrying the SGLang bootstrap triple so the
                        // decode peer can connect.
                        if role == Role::Prefill {
                            let session = convert::disagg_params_from_meta(&data.meta_info)
                                .map(|d| convert::prefill_kv_session(&rid_clone, &d));
                            yield Ok(pb::GenerateResponse {
                                request_id: rid_clone.clone(),
                                event: Some(pb::generate_response::Event::PrefillReady(
                                    pb::PrefillReady { kv_session: session },
                                )),
                                usage: Some(pb::Usage {
                                    prompt_tokens: convert::usage_from_meta(&data.meta_info)
                                        .prompt_tokens,
                                    completion_tokens: 0,
                                    total_tokens: 0,
                                }),
                            });
                            break;
                        }
                        // Aggregated / decode: the terminal chunk may carry the final
                        // new token(s); emit them before the terminal so none drop.
                        let final_toks: Vec<u32> = data
                            .output_ids
                            .clone()
                            .unwrap_or_default()
                            .into_iter()
                            .map(|t| t as u32)
                            .collect();
                        if !final_toks.is_empty() {
                            yield Ok(token_response(&rid_clone, final_toks, String::new()));
                        }
                        let usage = convert::usage_from_meta(&data.meta_info);
                        let reason = convert::finish_reason_from_meta(&data.meta_info);
                        yield Ok(terminal_response(&rid_clone, reason, usage));
                        break;
                    }
                    Ok(Some(ResponseChunk::Error(msg))) => {
                        abort_guard.disarm();
                        yield Ok(pb::GenerateResponse {
                            request_id: rid_clone.clone(),
                            event: Some(pb::generate_response::Event::Error(pb::EngineError {
                                code: pb::ErrorCode::Internal as i32,
                                message: msg,
                                retry_hint: String::new(),
                            })),
                            usage: None,
                        });
                        break;
                    }
                    Ok(None) => {
                        // Channel closed before a terminal event.
                        let status = match bridge.take_terminal_error(&rid_clone) {
                            Some(err) => {
                                abort_guard.disarm();
                                terminal_error_status(err)
                            }
                            None => Status::internal(
                                "response stream closed before a terminal event",
                            ),
                        };
                        yield Err(status);
                        break;
                    }
                    Err(_) => {
                        yield Err(Status::deadline_exceeded(format!(
                            "request timed out after {}s",
                            response_timeout.as_secs()
                        )));
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_engine_info(
        &self,
        _request: Request<pb::GetEngineInfoRequest>,
    ) -> Result<Response<pb::EngineInfo>, Status> {
        let info = self.server_info_json()?;
        let parallelism = pb::ParallelismInfo {
            tensor_parallel_size: json_u64(&info, "tp_size") as u32,
            pipeline_parallel_size: json_u64(&info, "pp_size").max(1) as u32,
            data_parallel_size: json_u64(&info, "dp_size").max(1) as u32,
            data_parallel_rank: 0,
            data_parallel_start_rank: 0,
        };
        let role = match self.role {
            Role::Prefill => pb::EngineRole::Prefill,
            Role::Decode => pb::EngineRole::Decode,
            Role::Aggregated => pb::EngineRole::Aggregated,
        };
        // Prefill advertises its bootstrap host/port via the KV connector so the
        // Dynamo sidecar's EngineConfig drives the frontend PrefillRouter.
        let bootstrap_host = json_str(&info, "bootstrap_host");
        let bootstrap_port = json_u64(&info, "bootstrap_port") as u32;
        let kv_connector = if !bootstrap_host.is_empty() && bootstrap_port != 0 {
            Some(pb::KvConnectorInfo {
                enabled: true,
                transfer_backend: json_str(&info, "disaggregation_transfer_backend"),
                local_endpoints: vec![pb::KvEndpoint {
                    host: bootstrap_host,
                    port: bootstrap_port,
                    protocol: "nixl".to_string(),
                }],
                supported_protocols: Vec::new(),
                supports_remote_prefill: true,
                supports_decode_pull: false,
                supports_abort_cleanup: true,
                supports_drain: true,
                schema_version: 0,
            })
        } else {
            None
        };
        Ok(Response::new(pb::EngineInfo {
            engine_name: "sglang".to_string(),
            engine_version: json_str(&info, "version"),
            api_version: "openengine.v1".to_string(),
            role: role as i32,
            instance_id: json_str(&info, "instance_id"),
            supported_models: Vec::new(),
            parallelism: Some(parallelism),
            kv_connector,
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        let info = self.model_info_json()?;
        let context_len = self.bridge.context_len().max(0) as u32;
        let model_id = json_str(&info, "model_path");
        let served = json_str(&info, "served_model_name");
        Ok(Response::new(pb::ModelInfo {
            model_id: model_id.clone(),
            served_model_name: if served.is_empty() { model_id } else { served },
            served_model_aliases: Vec::new(),
            max_context_length: context_len,
            max_output_tokens: 0,
            kv_block_size: json_u64(&info, "page_size") as u32,
            total_kv_blocks: json_u64(&info, "total_kv_blocks"),
            max_running_requests: json_u64(&info, "max_running_requests"),
            max_batched_tokens: json_u64(&info, "max_prefill_tokens"),
            tokenizer_modes: Vec::new(),
            supports_text_input: true,
            supports_token_ids_input: true,
            supports_logprobs: true,
            supports_guided_decoding: true,
            supports_lora: false,
            supports_multimodal: json_u64(&info, "is_multimodal") != 0,
        }))
    }

    async fn get_load(
        &self,
        request: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::LoadInfo>, Status> {
        let _ = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();
        let mut receiver = self
            .bridge
            .submit_get_load(&rid, None)
            .map_err(|e| pyerr_to_status(e, "get_load"))?;
        let json = recv_json(&self.bridge, &rid, &mut receiver, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json).unwrap_or(serde_json::Value::Null);
        Ok(Response::new(pb::LoadInfo {
            instance_id: String::new(),
            timestamp_unix_nanos: now_unix_nanos(),
            running_requests: json_u64(&v, "num_running_reqs") as u32,
            queued_requests: json_u64(&v, "num_waiting_reqs") as u32,
            active_kv_sessions: 0,
            used_kv_blocks: json_u64(&v, "num_used_tokens"),
            total_kv_blocks: json_u64(&v, "max_total_num_tokens"),
            running_tokens: 0,
            waiting_tokens: 0,
            prefill_batch_size: 0,
            decode_batch_size: 0,
            ranks: Vec::new(),
            attributes: HashMap::new(),
        }))
    }

    async fn health(
        &self,
        _request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        let healthy = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.health_check()
        })
        .await
        .map_err(|e| Status::internal(format!("join: {e}")))?
        .map_err(|e| pyerr_to_status(e, "health_check"))?;

        let state = if healthy {
            pb::HealthState::Ready
        } else {
            pb::HealthState::NotReady
        };
        Ok(Response::new(pb::HealthResponse {
            state: state as i32,
            checks: vec![pb::HealthCheck {
                name: "scheduler".to_string(),
                state: state as i32,
                message: String::new(),
            }],
        }))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let req = request.into_inner();
        // Idempotent on the engine side; unknown ids are a no-op.
        self.bridge
            .abort(&req.request_id, req.abort_all)
            .map_err(|e| pyerr_to_status(e, "abort"))?;
        Ok(Response::new(pb::AbortResponse {
            status: pb::AbortStatus::Aborted as i32,
            message: String::new(),
        }))
    }

    async fn drain(
        &self,
        _request: Request<pb::DrainRequest>,
    ) -> Result<Response<Self::DrainStream>, Status> {
        // Minimal drain: report completion. Prefill poll-until-idle (so in-flight
        // KV transfers finish before teardown) is a refinement.
        let stream = async_stream::stream! {
            yield Ok(pb::DrainResponse {
                state: pb::DrainState::Complete as i32,
                in_flight_requests: 0,
                open_kv_sessions: 0,
                message: String::new(),
            });
        };
        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_kv_connector_info(
        &self,
        _request: Request<pb::GetKvConnectorInfoRequest>,
    ) -> Result<Response<pb::KvConnectorInfo>, Status> {
        Ok(Response::new(pb::KvConnectorInfo {
            enabled: false,
            transfer_backend: String::new(),
            local_endpoints: Vec::new(),
            supported_protocols: Vec::new(),
            supports_remote_prefill: false,
            supports_decode_pull: false,
            supports_abort_cleanup: false,
            supports_drain: false,
            schema_version: 0,
        }))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        // KV-aware routing: surface the SGLang scheduler's ZMQ publisher(s),
        // derived from --kv-events-config (see RuntimeHandle._kv_event_sources).
        // The KV router subscribes directly and parses SGLang's msgpack events.
        let info = self.server_info_json()?;
        let mut sources = Vec::new();
        if let Some(arr) = info.get("kv_event_sources").and_then(|v| v.as_array()) {
            for s in arr {
                let endpoint = json_str(s, "endpoint");
                if endpoint.is_empty() {
                    continue;
                }
                sources.push(pb::KvEventSource {
                    transport: "zmq".to_string(),
                    endpoint: endpoint.clone(),
                    topic: json_str(s, "topic"),
                    replay_endpoint: String::new(),
                    data_parallel_rank: s
                        .get("dp_rank")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32,
                    encoding: json_str(s, "encoding"),
                    schema_version: 0,
                    buffer_steps: 0,
                    hwm: 0,
                    max_queue_size: 0,
                    endpoint_addr: parse_zmq_endpoint(&endpoint),
                });
            }
        }
        Ok(Response::new(pb::GetKvEventSourcesResponse { sources }))
    }

    async fn subscribe_kv_events(
        &self,
        _request: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented(
            "SubscribeKvEvents is not implemented; use GetKvEventSources and subscribe to ZMQ directly",
        ))
    }

    async fn subscribe_runtime_events(
        &self,
        _request: Request<pb::SubscribeRuntimeEventsRequest>,
    ) -> Result<Response<Self::SubscribeRuntimeEventsStream>, Status> {
        Err(Status::unimplemented("SubscribeRuntimeEvents is not implemented"))
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

/// Receive a single terminal JSON chunk for unary control RPCs (e.g. get_load).
async fn recv_json(
    bridge: &Arc<PyBridge>,
    rid: &str,
    receiver: &mut Receiver<ResponseChunk>,
    response_timeout: Duration,
) -> Result<String, Status> {
    let mut abort_guard = RequestAbortGuard::new(bridge.clone(), rid.to_string());
    match timeout(response_timeout, receiver.recv()).await {
        Ok(Some(ResponseChunk::Finished(data))) | Ok(Some(ResponseChunk::Data(data))) => {
            abort_guard.disarm();
            let bytes = data.json_bytes.unwrap_or_default();
            String::from_utf8(bytes).map_err(|e| Status::internal(format!("utf8: {e}")))
        }
        Ok(Some(ResponseChunk::Error(msg))) => {
            abort_guard.disarm();
            Err(Status::internal(msg))
        }
        Ok(None) => Err(Status::internal("control stream closed before a response")),
        Err(_) => Err(Status::deadline_exceeded("control RPC timed out")),
    }
}

/// Serve the OpenEngine service on the given listener until `shutdown` fires.
pub async fn run_openengine_server(
    listener: std::net::TcpListener,
    bridge: Arc<PyBridge>,
    role: Role,
    shutdown: Arc<tokio::sync::Notify>,
    response_timeout: Duration,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = listener.local_addr()?;
    let listener = tokio::net::TcpListener::from_std(listener)?;
    let service = OpenEngineServiceImpl {
        bridge,
        response_timeout,
        role,
    };

    let max_message_size = crate::server::DEFAULT_GRPC_MAX_MESSAGE_SIZE;
    let svc = pb::open_engine_server::OpenEngineServer::new(service)
        .max_decoding_message_size(max_message_size)
        .max_encoding_message_size(max_message_size);

    tracing::info!("OpenEngine server listening on {}", addr);

    tonic::transport::Server::builder()
        .add_service(svc)
        .serve_with_incoming_shutdown(
            tokio_stream::wrappers::TcpListenerStream::new(listener),
            async move {
                shutdown.notified().await;
                tracing::info!("OpenEngine server shutting down");
            },
        )
        .await?;

    Ok(())
}

/// Discover the engine's disaggregation role from the scheduler's server-info
/// JSON. `disaggregation_mode` is `null`/`prefill`/`decode` in SGLang.
pub fn discover_role(bridge: &PyBridge) -> Role {
    let raw = match bridge.get_server_info() {
        Ok(s) => s,
        Err(_) => return Role::Aggregated,
    };
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap_or(serde_json::Value::Null);
    match v.get("disaggregation_mode").and_then(|x| x.as_str()) {
        Some("prefill") => Role::Prefill,
        Some("decode") => Role::Decode,
        _ => Role::Aggregated,
    }
}
