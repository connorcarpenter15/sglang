"""Python-side bridge between the Rust gRPC server and TokenizerManager.

The RuntimeHandle exposes synchronous methods that Rust can call via PyO3
(with a brief GIL acquisition). Response chunks are pushed into Rust-side
channels via callback objects while all async work stays on the
TokenizerManager's event loop.
"""

import asyncio
import base64
import dataclasses
import json
import logging
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import ValidationError

from sglang.srt.utils.msgspec_utils import msgspec_to_builtins

logger = logging.getLogger(__name__)


class _BadOpenAIRequest(ValueError):
    pass


class _CaseInsensitiveHeaders:
    __slots__ = ("_data",)

    def __init__(self, headers: Optional[Dict[str, str]] = None):
        self._data = {k.lower(): v for k, v in (headers or {}).items()}

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self._data.get(name.lower(), default)


class _GrpcRequest:
    """Small FastAPI Request shim used by OpenAIServing* and TokenizerManager."""

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ):
        self.headers = _CaseInsensitiveHeaders(headers)
        self.state = SimpleNamespace()
        self._is_disconnected_fn = is_disconnected_fn

    async def is_disconnected(self) -> bool:
        if self._is_disconnected_fn is None:
            return False
        return bool(self._is_disconnected_fn())


class RuntimeHandle:
    """Thin Python handle that the Rust gRPC server calls into.

    Provides synchronous ``submit_*``, ``abort``, and info methods.
    Each submit method receives a ``chunk_callback`` (a Rust-side PyO3 object)
    that it invokes with ``(chunk_dict, finished, error)`` for each response
    chunk produced by TokenizerManager.
    """

    def __init__(
        self,
        tokenizer_manager,
        template_manager,
        server_args,
        scheduler_info: Optional[Dict] = None,
    ):
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self.server_args = server_args
        self.scheduler_info = scheduler_info or {}

        self._openai_serving_classes = None
        self._grpc_nixl_connector = None

        self.tokenizer_manager.auto_create_handle_loop()
        self._event_loop = self.tokenizer_manager.event_loop

    @property
    def _tm_loop(self):
        """Return the TokenizerManager loop used by communicator RPCs."""
        return self._event_loop

    def _safe_callback(self, chunk_callback, payload, **kwargs):
        """Invoke a Rust callback and return its ChunkSendStatus, if any."""
        try:
            return chunk_callback(payload, **kwargs)
        except Exception as e:
            logger.warning("gRPC chunk_callback failed: %s", e)
            return None

    def _send_native_error(self, chunk_callback, message: str):
        # ChunkCallback extracts the PyDict arg before reading error=.
        return self._safe_callback(chunk_callback, {}, finished=True, error=message)

    _BACKPRESSURE_TIMEOUT_S = 300.0

    @staticmethod
    def _is_pending_status(status) -> bool:
        return status is not None and status == type(status).Pending

    @staticmethod
    def _is_closed_status(status) -> bool:
        return status is not None and status == type(status).Closed

    def _abort_request_id(self, rid) -> None:
        if isinstance(rid, list):
            for single_rid in rid:
                self.tokenizer_manager.abort_request(rid=single_rid)
        else:
            self.tokenizer_manager.abort_request(rid=rid)

    async def _send_with_backpressure(
        self,
        chunk_callback,
        ready_event: Optional[asyncio.Event],
        payload,
        *,
        timeout_abort_rid=None,
        **kwargs,
    ) -> bool:
        status = self._safe_callback(chunk_callback, payload, **kwargs)
        if status is None or self._is_closed_status(status):
            return False
        if not self._is_pending_status(status):
            return True

        if kwargs.get("finished"):
            return True
        if ready_event is None:
            return True

        try:
            await asyncio.wait_for(
                ready_event.wait(), timeout=self._BACKPRESSURE_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            if timeout_abort_rid is not None:
                self._abort_request_id(timeout_abort_rid)
                logger.warning(
                    "gRPC chunk backpressure wait timed out after %ss; aborted request",
                    self._BACKPRESSURE_TIMEOUT_S,
                )
            else:
                logger.warning(
                    "gRPC chunk backpressure wait timed out after %ss; closing stream",
                    self._BACKPRESSURE_TIMEOUT_S,
                )
            return False
        ready_event.clear()
        return True

    def _install_on_ready(self, chunk_callback) -> Optional[asyncio.Event]:
        set_on_ready = getattr(chunk_callback, "set_on_ready", None)
        if set_on_ready is None:
            return None
        ready_event = asyncio.Event()
        loop = self._tm_loop

        def _on_ready() -> None:
            loop.call_soon_threadsafe(ready_event.set)

        try:
            set_on_ready(_on_ready)
        except Exception as e:
            logger.warning("gRPC set_on_ready failed: %s", e)
            raise
        return ready_event

    @staticmethod
    def _uninstall_on_ready(chunk_callback) -> None:
        clear = getattr(chunk_callback, "clear_on_ready", None)
        if clear is None:
            return
        try:
            clear()
        except Exception as e:
            logger.warning("gRPC clear_on_ready failed: %s", e)

    def _submit_on_tm_loop(self, coro: Awaitable) -> None:
        future = asyncio.run_coroutine_threadsafe(coro, self._tm_loop)
        future.add_done_callback(self._log_unhandled_future_exception)

    @staticmethod
    def _log_unhandled_future_exception(future) -> None:
        try:
            future.result()
        except Exception as e:
            logger.error(
                "gRPC scheduled coroutine raised unhandled exception: %s",
                e,
                exc_info=True,
            )

    def _submit_json_unary(
        self,
        op_name: str,
        payload_coro_factory: Callable[[], Awaitable[Any]],
        chunk_callback,
        *,
        error_payload_fn: Optional[Callable[[Exception], Any]] = None,
    ) -> None:
        error_fn = error_payload_fn or (lambda e: {"error": {"message": str(e)}})

        async def _run() -> None:
            try:
                payload = await payload_coro_factory()
                self._safe_callback(
                    chunk_callback,
                    json.dumps(payload, default=str).encode("utf-8"),
                    finished=True,
                )
            except Exception as e:
                logger.error("gRPC %s error: %s", op_name, e)
                self._safe_callback(
                    chunk_callback,
                    json.dumps(error_fn(e), default=str).encode("utf-8"),
                    finished=True,
                    error=str(e),
                )

        self._submit_on_tm_loop(_run())

    def _get_openai_serving(self):
        """Lazily initialize OpenAI serving classes."""
        if self._openai_serving_classes is not None:
            return self._openai_serving_classes

        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sglang.srt.entrypoints.openai.serving_classify import (
            OpenAIServingClassify,
        )
        from sglang.srt.entrypoints.openai.serving_completions import (
            OpenAIServingCompletion,
        )
        from sglang.srt.entrypoints.openai.serving_embedding import (
            OpenAIServingEmbedding,
        )
        from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank
        from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore

        self._openai_serving_classes = {
            "chat": OpenAIServingChat(self.tokenizer_manager, self.template_manager),
            "completion": OpenAIServingCompletion(
                self.tokenizer_manager, self.template_manager
            ),
            "embedding": OpenAIServingEmbedding(
                self.tokenizer_manager, self.template_manager
            ),
            "classify": OpenAIServingClassify(
                self.tokenizer_manager, self.template_manager
            ),
            "score": OpenAIServingScore(self.tokenizer_manager),
            "rerank": OpenAIServingRerank(
                self.tokenizer_manager, self.template_manager
            ),
        }
        return self._openai_serving_classes

    def submit_request(
        self,
        *,
        req_type: str,
        req_dict: dict,
        chunk_callback,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ):
        mock_request = (
            _GrpcRequest(is_disconnected_fn=is_disconnected_fn)
            if is_disconnected_fn is not None
            else None
        )
        if req_type == "generate":
            stream = req_dict.get("stream", False)
            self._submit_on_tm_loop(
                self._prepare_and_run_generate(
                    req_dict, chunk_callback, stream, mock_request
                )
            )
        elif req_type == "embed":
            self._submit_on_tm_loop(
                self._prepare_and_run_embed(req_dict, chunk_callback, mock_request)
            )
        else:
            raise ValueError(
                f"Unknown req_type: {req_type!r} (expected 'generate' or 'embed')"
            )

    @staticmethod
    def _tensor_dtype(dtype: int):
        import numpy as np

        dtypes = {
            1: np.uint8,
            2: np.int32,
            3: np.int64,
            4: np.float16,
            5: np.dtype("bfloat16") if hasattr(np, "bfloat16") else np.float16,
            6: np.float32,
            7: np.float64,
        }
        if dtype not in dtypes:
            raise ValueError(f"Unsupported gRPC tensor dtype: {dtype}")
        return dtypes[dtype]

    @staticmethod
    def _torch_tensor_dtype(dtype: int):
        import torch

        dtypes = {
            1: torch.uint8,
            2: torch.int32,
            3: torch.int64,
            4: torch.float16,
            5: torch.bfloat16,
            6: torch.float32,
            7: torch.float64,
        }
        if dtype not in dtypes:
            raise ValueError(f"Unsupported gRPC tensor dtype: {dtype}")
        return dtypes[dtype]

    async def _materialize_grpc_tensor(self, tensor: dict):
        import io

        import numpy as np

        storage = tensor.get("storage") or {}
        shape = tuple(int(dim) for dim in tensor.get("shape") or [])
        if storage.get("kind") == "serialized":
            if storage.get("format") != "pytorch":
                raise ValueError(
                    f"Unsupported serialized tensor format: {storage.get('format')!r}"
                )
            import torch

            value = torch.load(
                io.BytesIO(bytes(storage.get("data") or [])),
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    "Serialized prompt embeddings did not contain a tensor"
                )
            if shape and tuple(value.shape) != shape:
                raise ValueError(
                    f"Serialized tensor shape {tuple(value.shape)} does not match {shape}"
                )
            return value.detach().cpu().numpy()

        dtype = self._tensor_dtype(int(tensor.get("dtype", 0)))
        if storage.get("kind") == "inline":
            array = np.frombuffer(bytes(storage.get("data") or []), dtype=dtype)
            return array.reshape(shape) if shape else array
        if storage.get("kind") != "nixl":
            raise ValueError("gRPC tensor is missing inline or NIXL storage")

        try:
            import torch
            from dynamo import nixl_connect
            from dynamo.nixl_connect import (
                OperationKind,
                RdmaMetadata,
                SerializedDescriptor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "NIXL external buffers require Dynamo's nixl_connect runtime"
            ) from exc

        if self._grpc_nixl_connector is None:
            self._grpc_nixl_connector = nixl_connect.Connector()
            await self._grpc_nixl_connector.initialize()
        metadata = json.loads(bytes(storage.get("metadata") or []).decode("utf-8"))
        descriptor = json.loads(bytes(storage.get("descriptor") or []).decode("utf-8"))
        remote_device = (
            "cpu"
            if descriptor.get("mem_type", "dram").lower() == "dram"
            else f"cuda:{descriptor.get('device_id', 0)}"
        )
        rdma_metadata = RdmaMetadata(
            descriptors=[
                SerializedDescriptor(
                    device=remote_device,
                    ptr=descriptor["addr"],
                    size=descriptor["size"],
                )
            ],
            nixl_metadata=metadata,
            notification_key=f"sglang-grpc-{id(tensor)}",
            operation_kind=int(OperationKind.READ),
        )
        torch_dtype = self._torch_tensor_dtype(int(tensor.get("dtype", 0)))
        target = torch.empty(shape, dtype=torch_dtype)
        local_descriptor = nixl_connect.Descriptor(target)
        operation = await self._grpc_nixl_connector.begin_read(
            rdma_metadata, local_descriptor
        )
        await operation.wait_for_completion()
        # Keep the torch dtype intact (notably bfloat16, which NumPy cannot
        # represent portably). GenerateReqInput accepts tensor-like values and
        # prompt embeddings are converted with `.tolist()` by the caller.
        return target

    async def _materialize_grpc_request(self, req_dict: dict) -> dict:
        req_dict = dict(req_dict)
        tensor = req_dict.pop("grpc_input_embeds", None)
        if tensor is not None:
            if isinstance(tensor, list):
                req_dict["input_embeds"] = [
                    (await self._materialize_grpc_tensor(value)).tolist()
                    for value in tensor
                ]
            else:
                req_dict["input_embeds"] = (
                    await self._materialize_grpc_tensor(tensor)
                ).tolist()

        mm_inputs = req_dict.pop("grpc_multimodal_inputs", [])
        hashes_by_modality = {1: [], 2: [], 3: []}
        by_modality = {1: [], 2: [], 3: []}
        for item in mm_inputs:
            source = item.get("source") or {}
            kind = source.get("kind")
            if kind == "url":
                value = source.get("value")
            elif kind == "inline":
                mime = item.get("mime_type") or "application/octet-stream"
                encoded = base64.b64encode(bytes(source.get("value") or [])).decode()
                value = f"data:{mime};base64,{encoded}"
            elif kind in ("tensor", "external"):
                value = await self._materialize_grpc_tensor(source.get("value"))
            else:
                raise ValueError("Multimodal input is missing its source")
            modality = int(item.get("modality", 0))
            by_modality[modality].append(value)
            if item.get("routing_hash"):
                hashes_by_modality[modality].append(item["routing_hash"])
        for field, modality in (
            ("image_data", 1),
            ("video_data", 2),
            ("audio_data", 3),
        ):
            values = by_modality[modality]
            if values:
                req_dict[field] = values[0] if len(values) == 1 else values
        mm_hashes = [
            value for modality in (1, 2, 3) for value in hashes_by_modality[modality]
        ]
        if mm_hashes:
            req_dict["mm_hashes"] = mm_hashes

        processor_options = req_dict.pop("grpc_multimodal_processor_options", {})
        allowed = getattr(
            __import__(
                "sglang.srt.managers.io_struct", fromlist=["GenerateReqInput"]
            ).GenerateReqInput,
            "__annotations__",
            {},
        )
        for key, value in processor_options.items():
            if key in allowed and key not in req_dict:
                req_dict[key] = value
        req_dict.pop("grpc_stop_visibility", None)
        return req_dict

    async def _prepare_and_run_generate(
        self, req_dict, chunk_callback, stream: bool, request
    ):
        from sglang.srt.managers.io_struct import GenerateReqInput

        try:
            stop_visibility = req_dict.get("grpc_stop_visibility")
            req_dict = await self._materialize_grpc_request(req_dict)
            obj = GenerateReqInput(**req_dict)
        except Exception as exc:
            self._send_native_error(chunk_callback, str(exc))
            return
        await self._run_generate(
            obj, chunk_callback, stream, request, stop_visibility=stop_visibility
        )

    async def _prepare_and_run_embed(self, req_dict, chunk_callback, request):
        from sglang.srt.managers.io_struct import EmbeddingReqInput

        try:
            req_dict = await self._materialize_grpc_request(req_dict)
            obj = EmbeddingReqInput(**req_dict)
        except Exception as exc:
            self._send_native_error(chunk_callback, str(exc))
            return
        await self._run_embed(obj, chunk_callback, request)

    @staticmethod
    def _trim_output_token_metadata(chunk: dict, count: int) -> None:
        if count <= 0:
            return
        output_ids = chunk.get("output_ids")
        if isinstance(output_ids, list) and len(output_ids) >= count:
            chunk["output_ids"] = output_ids[:-count]
        meta_info = chunk.get("meta_info")
        if not isinstance(meta_info, dict):
            return
        for key in ("output_token_logprobs", "output_top_logprobs"):
            values = meta_info.get(key)
            if isinstance(values, list) and len(values) >= count:
                meta_info[key] = values[:-count]
        output_logprobs = meta_info.get("output_token_logprobs")
        if isinstance(output_logprobs, list):
            meta_info["output_token_logprobs_length"] = len(output_logprobs)

    def _decoded_token_suffix_length(
        self, output_ids: list[int], expected_text: str
    ) -> int:
        """Return the emitted-token suffix that decodes to hidden stop text.

        Encoding a stop string in isolation is not reliable: guided decoding can
        emit a different, context-dependent tokenization. Decode suffixes of the
        actual output instead so text and token metadata share one holdback.
        """

        if not output_ids or not expected_text:
            return 0
        tokenizer = getattr(self.tokenizer_manager, "tokenizer", None)
        decode = getattr(tokenizer, "decode", None)
        if decode is None:
            return 0
        max_suffix_tokens = min(len(output_ids), len(expected_text.encode("utf-8")))
        for count in range(1, max_suffix_tokens + 1):
            try:
                decoded = decode(
                    output_ids[-count:],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                decoded = decode(output_ids[-count:], skip_special_tokens=False)
            except Exception as exc:
                logger.debug(
                    "Could not decode gRPC output suffix while hiding a stop: %s",
                    exc,
                )
                return 0
            if decoded == expected_text:
                return count
        return 0

    def _hold_back_hidden_stop_text(
        self, chunk: dict, visibility: Optional[dict]
    ) -> None:
        if not visibility:
            return
        hidden_stops = [
            item.get("value")
            for item in visibility.get("strings", [])
            if item.get("value") and not item.get("include_in_output")
        ]
        text = chunk.get("text")
        held_text = ""
        if isinstance(text, str):
            holdback = max(
                (
                    prefix_length
                    for stop in hidden_stops
                    for prefix_length in range(1, len(stop) + 1)
                    if text.endswith(stop[:prefix_length])
                ),
                default=0,
            )
            if holdback:
                held_text = text[-holdback:]
                chunk["text"] = text[:-holdback]

        output_ids = chunk.get("output_ids")
        if not isinstance(output_ids, list) or not held_text:
            return
        self._trim_output_token_metadata(
            chunk, self._decoded_token_suffix_length(output_ids, held_text)
        )

    def _apply_stop_visibility(self, chunk: dict, visibility: Optional[dict]) -> None:
        if not visibility:
            return
        finish = chunk.get("meta_info", {}).get("finish_reason")
        if not isinstance(finish, dict) or finish.get("type") != "stop":
            return
        matched = finish.get("matched")
        configured_token_match = isinstance(matched, int) and any(
            item.get("token_id") == matched for item in visibility.get("tokens", [])
        )
        output_ids = chunk.get("output_ids", [])
        if (
            isinstance(matched, int)
            and not configured_token_match
            and isinstance(output_ids, list)
            and output_ids[-1:] == [matched]
        ):
            # Guided decoding terminates with its internal grammar EOS token.
            # It is a control marker, not generated output, and would otherwise
            # obscure a string stop immediately before it.
            self._trim_output_token_metadata(chunk, 1)
        if not configured_token_match:
            # Guided decoding may finish on the same suffix as a configured
            # stop condition while SGLang reports either no match or its
            # grammar EOS token. Recover the typed stop reason from the
            # terminal output so per-stop visibility still applies.
            text = chunk.get("text", "")
            string_matches = [
                item
                for item in visibility.get("strings", [])
                if item.get("value") and text.endswith(item["value"])
            ]
            if string_matches:
                matched = max(string_matches, key=lambda item: len(item["value"]))[
                    "value"
                ]
                output_ids = chunk.get("output_ids", [])
                if isinstance(output_ids, list):
                    # SGLang's finish metadata and the appended grammar marker
                    # can use different internal token IDs. Locate the real
                    # string-stop suffix through a short trailing control-token
                    # suffix instead of depending on either internal ID.
                    for control_tokens in range(1, min(4, len(output_ids)) + 1):
                        candidate_ids = output_ids[:-control_tokens]
                        if self._decoded_token_suffix_length(candidate_ids, matched):
                            self._trim_output_token_metadata(chunk, control_tokens)
                            break
            else:
                output_ids = chunk.get("output_ids", [])
                if output_ids:
                    matched_token = next(
                        (
                            item.get("token_id")
                            for item in visibility.get("tokens", [])
                            if item.get("token_id") == output_ids[-1]
                        ),
                        None,
                    )
                    if matched_token is not None:
                        matched = matched_token
            if matched is not None:
                finish["matched"] = matched
        if isinstance(matched, int):
            visible = any(
                item.get("token_id") == matched and item.get("include_in_output")
                for item in visibility.get("tokens", [])
            )
            if not visible and chunk.get("output_ids", [])[-1:] == [matched]:
                RuntimeHandle._trim_output_token_metadata(chunk, 1)
        elif isinstance(matched, str):
            visible = any(
                item.get("value") == matched and item.get("include_in_output")
                for item in visibility.get("strings", [])
            )
            if not visible and chunk.get("text", "").endswith(matched):
                chunk["text"] = chunk["text"][: -len(matched)]
            output_ids = chunk.get("output_ids", [])
            if not visible and isinstance(output_ids, list):
                self._trim_output_token_metadata(
                    chunk, self._decoded_token_suffix_length(output_ids, matched)
                )

    async def _run_generate(
        self,
        obj,
        chunk_callback,
        stream: bool,
        request,
        stop_visibility: Optional[dict] = None,
    ):
        ready_event = None
        try:
            ready_event = self._install_on_ready(chunk_callback) if stream else None
            gen = self.tokenizer_manager.generate_request(obj, request=request)
            if stream:
                sampling_params = getattr(obj, "sampling_params", None) or {}
                expected_choices = max(1, int(sampling_params.get("n", 1)))
                terminal_choices = set()
                async for chunk in gen:
                    choice_finished = (
                        chunk.get("meta_info", {}).get("finish_reason") is not None
                    )
                    request_finished = False
                    if choice_finished:
                        self._apply_stop_visibility(chunk, stop_visibility)
                        choice_index = int(chunk.get("index") or 0)
                        if choice_index in terminal_choices:
                            self._abort_request_id(obj.rid)
                            self._send_native_error(
                                chunk_callback,
                                f"duplicate terminal for choice {choice_index}",
                            )
                            return
                        terminal_choices.add(choice_index)
                        request_finished = len(terminal_choices) == expected_choices
                    else:
                        self._hold_back_hidden_stop_text(chunk, stop_visibility)
                    keep_going = await self._send_with_backpressure(
                        chunk_callback,
                        ready_event,
                        chunk,
                        finished=request_finished,
                        timeout_abort_rid=obj.rid,
                    )
                    if request_finished or not keep_going:
                        return
                # Defensive: generator exited without a finish_reason chunk.
                missing = sorted(set(range(expected_choices)) - terminal_choices)
                self._send_native_error(
                    chunk_callback,
                    f"SGLang stream ended without terminal choices: {missing}",
                )
            else:
                result = await gen.__anext__()
                results = result if isinstance(result, list) else [result]
                for index, item in enumerate(results):
                    self._apply_stop_visibility(item, stop_visibility)
                    self._safe_callback(
                        chunk_callback,
                        item,
                        finished=index == len(results) - 1,
                    )
        except StopAsyncIteration:
            self._safe_callback(chunk_callback, {}, finished=True)
        except Exception as e:
            logger.error("gRPC generate error for rid=%s: %s", obj.rid, e)
            self._send_native_error(chunk_callback, str(e))
        finally:
            if stream:
                self._uninstall_on_ready(chunk_callback)

    async def _run_embed(self, obj, chunk_callback, request):
        try:
            gen = self.tokenizer_manager.generate_request(obj, request=request)
            result = await gen.__anext__()
            if isinstance(result, list):
                result = {
                    "embedding": [item.get("embedding", []) for item in result],
                    "meta_info": {
                        "prompt_tokens": sum(
                            item.get("meta_info", {}).get("prompt_tokens", 0)
                            for item in result
                        )
                    },
                }
            self._safe_callback(chunk_callback, result, finished=True)
        except StopAsyncIteration:
            self._safe_callback(chunk_callback, {}, finished=True)
        except Exception as e:
            logger.error("gRPC embed error for rid=%s: %s", obj.rid, e)
            self._send_native_error(chunk_callback, str(e))

    # Bounded so a stuck TM loop can't deadlock the gRPC handler thread that
    # called abort. abort_request only enqueues a message on the ZMQ socket,
    # so a few seconds is generous; if we time out, log and drop — the client
    # will retry or give up.
    _ABORT_TIMEOUT_S = 5.0

    def abort(self, rid: str = "", abort_all: bool = False):
        """Abort a request by request ID or abort all active requests."""
        loop = self._tm_loop

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is loop:
            self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)
            return

        future = asyncio.run_coroutine_threadsafe(
            self._abort_async(rid, abort_all),
            loop,
        )
        try:
            future.result(timeout=self._ABORT_TIMEOUT_S)
        except TimeoutError:
            future.cancel()
            logger.error(
                "gRPC abort timed out after %ss (rid=%r, abort_all=%s); "
                "tokenizer_manager loop appears stuck",
                self._ABORT_TIMEOUT_S,
                rid,
                abort_all,
            )

    async def _abort_async(self, rid: str, abort_all: bool) -> None:
        self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)

    def get_model_info(self) -> str:
        model_config = self.tokenizer_manager.model_config
        result = {
            "model_path": self.tokenizer_manager.model_path,
            "tokenizer_path": self.server_args.tokenizer_path,
            "is_generation": self.tokenizer_manager.is_generation,
            "weight_version": self.server_args.weight_version,
            "model_type": getattr(model_config.hf_config, "model_type", None),
            "architectures": getattr(model_config.hf_config, "architectures", None),
        }
        return json.dumps(result, default=str)

    def get_server_info(self) -> str:
        result: Dict[str, Any] = dataclasses.asdict(self.server_args)
        result.update(self.scheduler_info)
        describe_kv_events = getattr(
            self.server_args, "describe_kv_events_publisher", None
        )
        if describe_kv_events is not None:
            result["kv_events"] = describe_kv_events()
        return json.dumps(msgspec_to_builtins(result), default=str)

    def health_check(self) -> bool:
        from sglang.srt.managers.tokenizer_manager import ServerStatus

        if self.tokenizer_manager.gracefully_exit:
            return False
        return self.tokenizer_manager.server_status not in (
            ServerStatus.Starting,
            ServerStatus.UnHealthy,
        )

    def tokenize(self, text: str, add_special_tokens: bool = True) -> str:
        tokenizer = self.tokenizer_manager.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        result = {
            "tokens": tokens,
            "count": len(tokens),
            "max_model_len": self.tokenizer_manager.model_config.context_len,
            "input_text": text,
        }
        return json.dumps(result)

    def detokenize(self, tokens: List[int]) -> str:
        tokenizer = self.tokenizer_manager.tokenizer
        text = tokenizer.decode(tokens)
        return json.dumps({"text": text})

    def list_models(self) -> str:
        served_model_name = self.tokenizer_manager.served_model_name
        models = [
            {
                "id": served_model_name,
                "root": served_model_name,
                "max_model_len": self.tokenizer_manager.model_config.context_len,
            }
        ]
        if self.server_args.enable_lora and hasattr(
            self.tokenizer_manager, "lora_registry"
        ):
            lora_registry = self.tokenizer_manager.lora_registry
            for _, lora_ref in lora_registry.get_all_adapters().items():
                models.append(
                    {
                        "id": lora_ref.lora_name,
                        "root": lora_ref.lora_path,
                        "parent": served_model_name,
                    }
                )
        return json.dumps(models)

    def get_load(self, chunk_callback, dp_rank: Optional[int] = None) -> None:
        async def _payload():
            result = await self.tokenizer_manager.get_loads(dp_rank=dp_rank)
            return [r.to_dict() for r in result]

        self._submit_json_unary("get_load", _payload, chunk_callback)

    def flush_cache(self, chunk_callback) -> None:
        async def _payload():
            ret = await self.tokenizer_manager.flush_cache()
            return {"success": ret.success, "message": "Cache flushed."}

        self._submit_json_unary(
            "flush_cache",
            _payload,
            chunk_callback,
            error_payload_fn=lambda e: {"success": False, "message": str(e)},
        )

    def pause_generation(self, mode: str, chunk_callback) -> None:
        async def _payload():
            from sglang.srt.managers.io_struct import PauseGenerationReqInput

            await self.tokenizer_manager.pause_generation(
                PauseGenerationReqInput(mode=mode)
            )
            return {"message": f"Generation paused (mode={mode})."}

        self._submit_json_unary("pause_generation", _payload, chunk_callback)

    def continue_generation(self, chunk_callback) -> None:
        async def _payload():
            from sglang.srt.managers.io_struct import ContinueGenerationReqInput

            await self.tokenizer_manager.continue_generation(
                ContinueGenerationReqInput()
            )
            return {"message": "Generation continued."}

        self._submit_json_unary("continue_generation", _payload, chunk_callback)

    def start_profile(self, output_dir: Optional[str], chunk_callback) -> None:
        async def _payload():
            from sglang.srt.managers.io_struct import ProfileReq

            req = ProfileReq(output_dir=output_dir) if output_dir else ProfileReq()
            await self.tokenizer_manager.start_profile(req)
            return {"message": "Profiling started."}

        self._submit_json_unary("start_profile", _payload, chunk_callback)

    def stop_profile(self, chunk_callback) -> None:
        async def _payload():
            await self.tokenizer_manager.stop_profile()
            return {"message": "Profiling stopped."}

        self._submit_json_unary("stop_profile", _payload, chunk_callback)

    def update_weights_from_disk(
        self, model_path: str, load_format: Optional[str], chunk_callback
    ) -> None:
        async def _payload():
            from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput

            obj = UpdateWeightFromDiskReqInput(
                model_path=model_path, load_format=load_format
            )
            (
                success,
                message,
                num_paused,
            ) = await self.tokenizer_manager.update_weights_from_disk(obj, request=None)
            return {
                "success": success,
                "message": message,
                "num_paused_requests": num_paused,
            }

        self._submit_json_unary(
            "update_weights",
            _payload,
            chunk_callback,
            error_payload_fn=lambda e: {"success": False, "message": str(e)},
        )

    def submit_control(self, method: str, json_body: bytes, chunk_callback) -> None:
        """Dispatch a typed gRPC admin RPC onto TokenizerManager's control API."""
        payload = json.loads(json_body or b"{}")

        async def _payload():
            from sglang.srt.managers.io_struct import (
                LoadLoRAAdapterReqInput,
                ProfileReq,
                ReleaseMemoryOccupationReqInput,
                ResumeMemoryOccupationReqInput,
                UnloadLoRAAdapterReqInput,
                UpdateWeightFromDiskReqInput,
                UpdateWeightsFromDistributedReqInput,
                UpdateWeightsFromIPCReqInput,
                UpdateWeightsFromTensorReqInput,
            )

            tm = self.tokenizer_manager
            if method == "load_lora":
                result = await tm.load_lora_adapter(
                    LoadLoRAAdapterReqInput(
                        lora_name=payload["name"],
                        lora_path=payload["path"],
                        pinned=payload.get("pinned", False),
                        lora_id=payload.get("id"),
                    )
                )
                return msgspec_to_builtins(result)
            if method == "unload_lora":
                result = await tm.unload_lora_adapter(
                    UnloadLoRAAdapterReqInput(
                        lora_name=payload["name"], lora_id=payload.get("id")
                    )
                )
                return msgspec_to_builtins(result)
            if method == "list_loras":
                adapters = []
                for name, ref in tm.lora_registry.get_all_adapters().items():
                    adapters.append(
                        {
                            "name": getattr(ref, "lora_name", name),
                            "path": getattr(ref, "lora_path", ""),
                            "id": getattr(ref, "lora_id", None),
                            "pinned": bool(getattr(ref, "pinned", False)),
                        }
                    )
                return {"adapters": adapters}
            if method == "release_memory":
                await tm.release_memory_occupation(
                    ReleaseMemoryOccupationReqInput(tags=payload.get("tags") or None)
                )
                return {"success": True, "message": "Memory released."}
            if method == "resume_memory":
                await tm.resume_memory_occupation(
                    ResumeMemoryOccupationReqInput(tags=payload.get("tags") or None)
                )
                return {"success": True, "message": "Memory resumed."}
            if method == "start_profile":
                await tm.start_profile(ProfileReq(**payload))
                return {"success": True, "message": "Profiling started."}
            if method == "stop_profile":
                await tm.stop_profile()
                return {"success": True, "message": "Profiling stopped."}
            if method == "update_weights_from_disk":
                success, message, num_paused = await tm.update_weights_from_disk(
                    UpdateWeightFromDiskReqInput(**payload), request=None
                )
                return {
                    "success": success,
                    "message": message,
                    "num_paused_requests": num_paused,
                }
            if method == "update_weights_from_tensor":
                payload["serialized_named_tensors"] = [
                    bytes(value) for value in payload["serialized_named_tensors"]
                ]
                success, message = await tm.update_weights_from_tensor(
                    UpdateWeightsFromTensorReqInput(**payload), request=None
                )
                return {"success": success, "message": message}
            if method == "update_weights_from_distributed":
                success, message = await tm.update_weights_from_distributed(
                    UpdateWeightsFromDistributedReqInput(**payload), request=None
                )
                return {"success": success, "message": message}
            if method == "update_weights_from_ipc":
                success, message = await tm.update_weights_from_ipc(
                    UpdateWeightsFromIPCReqInput(**payload), request=None
                )
                if success and not tm.initial_weights_loaded:
                    tm.initial_weights_loaded = True
                return {"success": success, "message": message}
            if method == "update_weight_version":
                from sglang.srt.managers.io_struct import UpdateWeightVersionReqInput

                req = UpdateWeightVersionReqInput(**payload)
                if req.abort_all_requests:
                    tm.abort_request(abort_all=True)
                tm.server_args.weight_version = req.new_version
                return {
                    "success": True,
                    "message": f"Weight version updated to {req.new_version}",
                    "new_version": req.new_version,
                }
            raise ValueError(f"Unknown gRPC control method: {method}")

        self._submit_json_unary(
            method,
            _payload,
            chunk_callback,
            error_payload_fn=lambda e: {"success": False, "message": str(e)},
        )

    def _submit_openai(
        self,
        serving_key: str,
        streaming: bool,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]],
        is_disconnected_fn: Optional[Callable[[], bool]],
    ) -> None:
        self._submit_on_tm_loop(
            self._run_openai_request(
                serving_key,
                json_body,
                chunk_callback,
                streaming=streaming,
                trace_headers=trace_headers,
                is_disconnected_fn=is_disconnected_fn,
            )
        )

    def submit_openai_chat(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._submit_openai(
            "chat", True, json_body, chunk_callback, trace_headers, is_disconnected_fn
        )

    def submit_openai_complete(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._submit_openai(
            "completion",
            True,
            json_body,
            chunk_callback,
            trace_headers,
            is_disconnected_fn,
        )

    def submit_openai_embed(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._submit_openai(
            "embedding",
            False,
            json_body,
            chunk_callback,
            trace_headers,
            is_disconnected_fn,
        )

    def submit_openai_classify(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._submit_openai(
            "classify",
            False,
            json_body,
            chunk_callback,
            trace_headers,
            is_disconnected_fn,
        )

    def submit_openai_score(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._submit_openai(
            "score", False, json_body, chunk_callback, trace_headers, is_disconnected_fn
        )

    def submit_openai_rerank(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._submit_openai(
            "rerank",
            False,
            json_body,
            chunk_callback,
            trace_headers,
            is_disconnected_fn,
        )

    def _get_openai_request_class(self, serving_key: str):
        """Return the Pydantic request class for a given serving key."""
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionRequest,
            ClassifyRequest,
            CompletionRequest,
            EmbeddingRequest,
            ScoringRequest,
            V1RerankReqInput,
        )

        return {
            "chat": ChatCompletionRequest,
            "completion": CompletionRequest,
            "embedding": EmbeddingRequest,
            "classify": ClassifyRequest,
            "score": ScoringRequest,
            "rerank": V1RerankReqInput,
        }[serving_key]

    async def _run_openai_request(
        self,
        serving_key: str,
        json_body: bytes,
        chunk_callback,
        streaming: bool,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ):
        try:
            serving = self._get_openai_serving()[serving_key]

            try:
                request_dict = json.loads(json_body)
                if not isinstance(request_dict, dict):
                    raise _BadOpenAIRequest(
                        f"Request body must be a JSON object, got {type(request_dict).__name__}"
                    )
                request_cls = self._get_openai_request_class(serving_key)
                request_obj = request_cls(**request_dict)
            except (json.JSONDecodeError, ValidationError, _BadOpenAIRequest) as e:
                error_body = json.dumps(
                    {"error": {"message": str(e), "type": "BadRequest"}}
                ).encode("utf-8")
                if streaming:
                    self._safe_callback(
                        chunk_callback, error_body, finished=True, error=str(e)
                    )
                else:
                    self._safe_callback(
                        chunk_callback, error_body, finished=True, status_code=400
                    )
                return

            mock_request = _GrpcRequest(
                headers=trace_headers,
                is_disconnected_fn=is_disconnected_fn,
            )

            result = await serving.handle_request(request_obj, mock_request)

            if hasattr(result, "body_iterator"):
                ready_event = self._install_on_ready(chunk_callback)
                data_buf: List[str] = []
                stream_closed = False

                async def _flush_event() -> bool:
                    """Flush buffered SSE data lines as one chunk. Returns False if Rust closed."""
                    if not data_buf:
                        return True
                    body = "\n".join(data_buf)
                    data_buf.clear()
                    if body == "[DONE]" or not body:
                        return True
                    return await self._send_with_backpressure(
                        chunk_callback,
                        ready_event,
                        body.encode("utf-8"),
                        finished=False,
                    )

                try:
                    async for raw_chunk in result.body_iterator:
                        if isinstance(raw_chunk, bytes):
                            raw_chunk = raw_chunk.decode("utf-8", errors="replace")
                        for line in raw_chunk.split("\n"):
                            line = line.rstrip("\r")
                            if not line:
                                if not await _flush_event():
                                    stream_closed = True
                                    break
                            elif line.startswith(":"):
                                continue  # SSE comment / heartbeat
                            elif line.startswith("data:"):
                                value = line[5:]
                                if value.startswith(" "):
                                    value = value[1:]
                                data_buf.append(value)
                            # event:, id:, retry:, unknown fields: ignored
                        if stream_closed:
                            break

                    if not stream_closed:
                        await _flush_event()
                        self._safe_callback(chunk_callback, b"", finished=True)
                finally:
                    self._uninstall_on_ready(chunk_callback)
            else:
                if hasattr(result, "model_dump"):
                    resp_bytes = json.dumps(result.model_dump()).encode("utf-8")
                elif hasattr(result, "body"):
                    resp_bytes = result.body
                elif isinstance(result, (dict, list)):
                    resp_bytes = json.dumps(result).encode("utf-8")
                else:
                    resp_bytes = str(result).encode("utf-8")
                status_code = int(
                    getattr(result, "status_code", None)
                    or getattr(result, "code", None)
                    or 200
                )
                self._safe_callback(
                    chunk_callback,
                    resp_bytes,
                    finished=True,
                    status_code=status_code,
                )

        except Exception as e:
            logger.error("gRPC OpenAI %s error: %s", serving_key, e)
            error_body = json.dumps({"error": {"message": str(e)}}).encode("utf-8")
            if streaming:
                self._safe_callback(
                    chunk_callback, error_body, finished=True, error=str(e)
                )
            else:
                self._safe_callback(
                    chunk_callback,
                    error_body,
                    finished=True,
                    status_code=int(getattr(e, "status_code", 500)),
                )
