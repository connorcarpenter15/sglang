"""Canonical OpenEngine services backed by SGLang's live TokenizerManager."""

import asyncio
import copy
import json
import logging
import math
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any

import grpc
from openengine import MINIMUM_CLIENT_REVISION, SCHEMA_REVISION
from openengine.v1 import (
    error_pb2,
    generation_pb2,
    input_pb2,
    kv_pb2,
    lifecycle_pb2,
    lora_pb2,
    model_pb2,
    observability_pb2,
    openengine_pb2_grpc,
    server_pb2,
)

from sglang.version import __version__ as sglang_version

from ._schema_pin import OPENENGINE_COMMIT
from .admission import DrainingError, ProcessAdmission
from .converters import HANDOFF_PROFILE, convert_generate
from .lora_registry import LoraRegistry

logger = logging.getLogger(__name__)


def _role(disaggregation_mode: str) -> int:
    if disaggregation_mode == "null":
        return server_pb2.ENGINE_ROLE_AGGREGATED
    if disaggregation_mode == "prefill":
        return server_pb2.ENGINE_ROLE_PREFILL
    if disaggregation_mode == "decode":
        return server_pb2.ENGINE_ROLE_DECODE
    raise ValueError(f"Unsupported SGLang disaggregation mode {disaggregation_mode!r}")


def _request_metadata(context: grpc.aio.ServicerContext) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in context.invocation_metadata():
        try:
            key, value = item
        except (TypeError, ValueError):
            key, value = item.key, item.value
        key = str(key).lower()
        if key.startswith("openengine-") and key in metadata:
            raise ValueError(f"Duplicate reserved gRPC metadata key {key!r}")
        if isinstance(value, bytes):
            try:
                value = value.decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"gRPC metadata value for {key!r} must be ASCII"
                ) from error
        metadata[key] = str(value)

    for key, minimum, maximum in (
        ("openengine-target-dp-rank", 0, (1 << 32) - 1),
        ("openengine-priority", -(1 << 31), (1 << 31) - 1),
    ):
        if key not in metadata:
            continue
        value = metadata[key]
        digits = value[1:] if value.startswith("-") and minimum < 0 else value
        if not digits.isdecimal():
            raise ValueError(f"gRPC metadata {key!r} must be a base-10 integer")
        parsed = int(value, 10)
        if not minimum <= parsed <= maximum:
            raise ValueError(f"gRPC metadata {key!r} must be in [{minimum}, {maximum}]")
    return metadata


def _logprob_parts(value: Any, fallback_token_id: int) -> tuple[float | None, int, str]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        logprob = None if value[0] is None else float(value[0])
        token_id = int(value[1])
        text = "" if len(value) < 3 or value[2] is None else str(value[2])
        return logprob, token_id, text
    return None, fallback_token_id, ""


def _token_info(
    token_id: int, logprob: Any = None, candidates: Any = None
) -> generation_pb2.TokenInfo:
    value, actual_id, text = _logprob_parts(logprob, token_id)
    info = generation_pb2.TokenInfo(token_id=actual_id, token=text)
    if value is not None:
        info.logprob = value
    for candidate in candidates or []:
        candidate_value, candidate_id, candidate_text = _logprob_parts(
            candidate, token_id
        )
        if candidate_value is None:
            continue
        info.candidates.append(
            generation_pb2.LogProb(
                token_id=candidate_id,
                logprob=candidate_value,
                token=candidate_text,
            )
        )
    return info


def _merge_candidates(*groups: Any) -> list[Any]:
    """Merge top-N and explicitly requested candidates without duplicates."""
    merged = []
    seen: set[int] = set()
    for group in groups:
        for candidate in group or []:
            _, token_id, _ = _logprob_parts(candidate, 0)
            if token_id not in seen:
                merged.append(candidate)
                seen.add(token_id)
    return merged


def _tail(values: Any, size: int) -> list[Any]:
    values = values or []
    return values[-size:] if size and len(values) > size else values


def _usage(meta: dict[str, Any]) -> generation_pb2.Usage:
    prompt = int(meta.get("prompt_tokens", 0) or 0)
    completion = int(meta.get("completion_tokens", 0) or 0)
    usage = generation_pb2.Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )
    cached = meta.get("cached_tokens")
    if cached is not None:
        usage.cached_prompt_tokens = max(0, int(cached))
    return usage


def _finish_reason(finish: dict[str, Any]) -> tuple[int, str]:
    kind = finish.get("type", "stop")
    if kind in ("length", "max_new_tokens"):
        return generation_pb2.FINISH_REASON_LENGTH, str(finish.get("message", ""))
    if kind == "abort":
        return generation_pb2.FINISH_REASON_CANCELLED, str(
            finish.get("message", "Request aborted")
        )
    return generation_pb2.FINISH_REASON_STOP, str(finish.get("message", ""))


def _finished_output(
    index: int, finish: dict[str, Any]
) -> generation_pb2.GenerationFinished:
    reason, message = _finish_reason(finish)
    result = generation_pb2.GenerationFinished(
        output_index=index,
        reason=reason,
        message=message,
    )
    matched = finish.get("matched")
    if isinstance(matched, int):
        result.stop_match.stop_token_id = matched
    elif isinstance(matched, str):
        result.stop_match.stop_text = matched
    return result


class OpenEngineServicer(
    openengine_pb2_grpc.InferenceServicer,
    openengine_pb2_grpc.ControlServicer,
):
    """One wire-neutral service facade over an existing SGLang runtime."""

    def __init__(
        self,
        runtime_handle,
        admission: ProcessAdmission,
        advertised_host: str,
        instance_id: str | None = None,
    ) -> None:
        self.runtime = runtime_handle
        self.tm = runtime_handle.tokenizer_manager
        self.args = runtime_handle.server_args
        self.scheduler_info = runtime_handle.scheduler_info
        self.admission = admission
        self.advertised_host = advertised_host
        self.instance_id = instance_id or str(uuid.uuid4())
        self.role = _role(self.args.disaggregation_mode)
        self.served_model_name = self.args.served_model_name
        self.model_id = self.args.model_path
        self.model_aliases = {self.served_model_name, self.model_id}
        self.loras = LoraRegistry(self.tm)
        self._engine_request_ids: dict[str, tuple[str, ...]] = {}

    async def close(self) -> None:
        await self.loras.close()

    def _split_engine_requests(self, request_id: str, converted, count: int):
        if count == 1:
            return [converted.request]
        group_id = uuid.uuid4().hex
        requests = []
        for index in range(count):
            child = copy.deepcopy(converted.request)
            child.rid = f"{request_id}.openengine.{group_id}.{index}"
            child.sampling_params = {**child.sampling_params, "n": 1}
            requests.append(child)
        return requests

    async def _generate_engine_outputs(self, requests):
        generators = [
            self.tm.generate_request(value, request=None) for value in requests
        ]
        if len(generators) == 1:
            async for output in generators[0]:
                yield output
            return

        tasks = {
            asyncio.create_task(anext(generator)): (index, generator)
            for index, generator in enumerate(generators)
        }
        try:
            while tasks:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    index, generator = tasks.pop(task)
                    try:
                        output = task.result()
                    except StopAsyncIteration:
                        continue
                    output = dict(output)
                    output["index"] = index
                    yield output
                    tasks[asyncio.create_task(anext(generator))] = (index, generator)
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(
                *(generator.aclose() for generator in generators),
                return_exceptions=True,
            )

    def _abort_wire_request(self, request_id: str) -> None:
        for engine_request_id in self._engine_request_ids.get(
            request_id, (request_id,)
        ):
            self.runtime.abort(rid=engine_request_id)

    def _validate_media_request(self, request: generation_pb2.GenerateRequest) -> None:
        supported, _ = self._modalities()
        supported_set = set(supported)
        for item in request.media:
            if item.modality not in supported_set:
                raise ValueError(
                    f"This SGLang model does not support media modality {item.modality}"
                )
            if (
                self.role
                in (server_pb2.ENGINE_ROLE_PREFILL, server_pb2.ENGINE_ROLE_DECODE)
                and item.modality == input_pb2.MODALITY_AUDIO
            ):
                raise ValueError("SGLang audio prefill/decode is unsupported")

    def _log_handoff(
        self,
        request: generation_pb2.GenerateRequest,
        phase: str,
        usage: generation_pb2.Usage | None = None,
    ) -> None:
        if self.role == server_pb2.ENGINE_ROLE_AGGREGATED:
            return
        session = request.kv.session
        if not session.session_id:
            return
        endpoint = session.bootstrap.endpoint
        record = {
            "phase": phase,
            "role": server_pb2.EngineRole.Name(self.role),
            "request_id": request.request_id,
            "session_id": session.session_id,
            "handoff_profile": session.handoff_profile,
            "bootstrap": {
                "host": endpoint.host,
                "port": endpoint.port,
                "protocol": endpoint.protocol,
                # Preserve uint64 precision in JSON logs.
                "room_id": str(session.bootstrap.room_id),
            },
        }
        if usage is not None:
            record["usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
        logger.info("OpenEngine handoff %s", json.dumps(record, sort_keys=True))

    def _log_lora_selection(
        self, request: generation_pb2.GenerateRequest, phase: str
    ) -> None:
        if not request.lora_name:
            return
        logger.info(
            "OpenEngine LoRA selection %s",
            json.dumps(
                {
                    "phase": phase,
                    "role": server_pb2.EngineRole.Name(self.role),
                    "request_id": request.request_id,
                    "session_id": request.kv.session.session_id,
                    "lora_name": request.lora_name,
                },
                sort_keys=True,
            ),
        )

    async def Generate(
        self,
        request: generation_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[generation_pb2.GenerateResponse, None]:
        admitted = False
        engine_started = False
        completed = False
        selected_lora = ""
        session_id = ""
        try:
            metadata = _request_metadata(context)
            self._validate_media_request(request)
            converted = convert_generate(
                request,
                role=self.role,
                served_model_name=self.served_model_name,
                model_aliases=self.model_aliases,
                metadata=metadata,
                lora_path=("logical" if request.lora_name else None),
            )
            session_id = converted.session_id
            await self.admission.admit(request.request_id, session_id)
            admitted = True
            self._log_handoff(request, "admitted")

            if request.lora_name:
                await self.loras.acquire(request.lora_name)
                selected_lora = request.lora_name
                self._log_lora_selection(request, "selected")

            expected_finishes = int(converted.request.sampling_params.get("n", 1) or 1)
            engine_requests = self._split_engine_requests(
                request.request_id, converted, expected_finishes
            )
            self._engine_request_ids[request.request_id] = tuple(
                value.rid for value in engine_requests
            )
            generator = self._generate_engine_outputs(engine_requests)
            engine_started = True
            seen_ids: dict[int, list[int]] = defaultdict(list)
            seen_text: dict[int, str] = defaultdict(str)
            prompt_emitted = False
            finish_count = 0
            final_usage: dict[int, generation_pb2.Usage] = {}

            async for output in generator:
                meta = output.get("meta_info") or {}
                index = int(output.get("index", 0))
                finish = meta.get("finish_reason")
                if isinstance(finish, dict) and finish.get("type") == "abort":
                    code = (
                        error_pb2.ERROR_CODE_INVALID_ARGUMENT
                        if finish.get("status_code") == 400
                        else error_pb2.ERROR_CODE_CANCELLED
                    )
                    yield generation_pb2.GenerateResponse(
                        request_id=request.request_id,
                        error=error_pb2.EngineError(
                            code=code,
                            message=str(
                                finish.get("message", "SGLang request aborted")
                            ),
                            retryable=finish.get("status_code") == 503,
                        ),
                        usage=_usage(meta),
                    )
                    return

                if (
                    not prompt_emitted
                    and request.response.HasField("return_prompt_logprobs")
                    and request.response.return_prompt_logprobs
                    and meta.get("input_token_logprobs") is not None
                ):
                    prompt_values = meta.get("input_token_logprobs") or []
                    prompt_candidates = meta.get("input_top_logprobs") or []
                    prompt_requested = meta.get("input_token_ids_logprobs") or []
                    prompt = generation_pb2.PromptOutput()
                    for offset, value in enumerate(prompt_values):
                        _, token_id, _ = _logprob_parts(value, 0)
                        top_candidates = (
                            prompt_candidates[offset]
                            if offset < len(prompt_candidates)
                            else None
                        )
                        requested_candidates = (
                            prompt_requested[offset]
                            if offset < len(prompt_requested)
                            else None
                        )
                        prompt.tokens.append(
                            _token_info(
                                token_id,
                                value,
                                _merge_candidates(top_candidates, requested_candidates),
                            )
                        )
                    yield generation_pb2.GenerateResponse(
                        request_id=request.request_id, prompt=prompt
                    )
                    prompt_emitted = True

                incoming_ids = [int(value) for value in output.get("output_ids", [])]
                previous_ids = seen_ids[index]
                if (
                    len(incoming_ids) >= len(previous_ids)
                    and incoming_ids[: len(previous_ids)] == previous_ids
                ):
                    delta_ids = incoming_ids[len(previous_ids) :]
                    seen_ids[index] = incoming_ids
                else:
                    delta_ids = incoming_ids
                    previous_ids.extend(incoming_ids)

                incoming_text = output.get("text") or ""
                previous_text = seen_text[index]
                if incoming_text.startswith(previous_text):
                    delta_text = incoming_text[len(previous_text) :]
                    seen_text[index] = incoming_text
                else:
                    delta_text = incoming_text
                    seen_text[index] += incoming_text

                if delta_ids or delta_text:
                    logprobs = _tail(meta.get("output_token_logprobs"), len(delta_ids))
                    top_logprobs = _tail(
                        meta.get("output_top_logprobs"), len(delta_ids)
                    )
                    requested_logprobs = _tail(
                        meta.get("output_token_ids_logprobs"), len(delta_ids)
                    )
                    token_output = generation_pb2.TokenOutput(
                        output_index=index, text=delta_text
                    )
                    for offset, token_id in enumerate(delta_ids):
                        value = logprobs[offset] if offset < len(logprobs) else None
                        top_candidates = (
                            top_logprobs[offset] if offset < len(top_logprobs) else None
                        )
                        requested_candidates = (
                            requested_logprobs[offset]
                            if offset < len(requested_logprobs)
                            else None
                        )
                        token_output.tokens.append(
                            _token_info(
                                token_id,
                                value,
                                _merge_candidates(top_candidates, requested_candidates),
                            )
                        )
                    if not converted.is_prefill:
                        yield generation_pb2.GenerateResponse(
                            request_id=request.request_id, token=token_output
                        )

                if finish is not None:
                    final_usage[index] = _usage(meta)
                    finish_count += 1
                    if converted.is_prefill:
                        response = generation_pb2.GenerateResponse(
                            request_id=request.request_id,
                            prefill_ready=generation_pb2.PrefillReady(
                                kv_session=request.kv.session
                            ),
                            usage=final_usage[index],
                        )
                    else:
                        response = generation_pb2.GenerateResponse(
                            request_id=request.request_id,
                            finished=_finished_output(index, finish),
                        )
                        if finish_count == expected_finishes:
                            prompt_tokens = max(
                                (value.prompt_tokens for value in final_usage.values()),
                                default=0,
                            )
                            completion_tokens = sum(
                                value.completion_tokens
                                for value in final_usage.values()
                            )
                            response.usage.CopyFrom(
                                generation_pb2.Usage(
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    total_tokens=prompt_tokens + completion_tokens,
                                )
                            )
                    terminal = converted.is_prefill or finish_count == expected_finishes
                    if terminal:
                        completed = True
                        self._log_handoff(request, "complete", response.usage)
                        self._log_lora_selection(request, "complete")
                    yield response

            if not completed:
                yield generation_pb2.GenerateResponse(
                    request_id=request.request_id,
                    error=error_pb2.EngineError(
                        code=error_pb2.ERROR_CODE_INTERNAL,
                        message="SGLang response stream ended without a terminal event",
                    ),
                )
        except DrainingError as error:
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(error))
        except LookupError as error:
            if admitted:
                yield generation_pb2.GenerateResponse(
                    request_id=request.request_id,
                    error=error_pb2.EngineError(
                        code=error_pb2.ERROR_CODE_INVALID_ARGUMENT,
                        message=str(error),
                    ),
                )
            else:
                await context.abort(grpc.StatusCode.NOT_FOUND, str(error))
        except ValueError as error:
            if admitted:
                yield generation_pb2.GenerateResponse(
                    request_id=request.request_id,
                    error=error_pb2.EngineError(
                        code=error_pb2.ERROR_CODE_INVALID_ARGUMENT,
                        message=str(error),
                    ),
                )
            else:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("OpenEngine Generate failed for %s", request.request_id)
            if admitted:
                yield generation_pb2.GenerateResponse(
                    request_id=request.request_id,
                    error=error_pb2.EngineError(
                        code=error_pb2.ERROR_CODE_INTERNAL,
                        message=str(error),
                    ),
                )
            else:
                await context.abort(grpc.StatusCode.INTERNAL, str(error))
        finally:
            if engine_started and not completed:
                self._abort_wire_request(request.request_id)
            if admitted and not completed:
                self._log_handoff(request, "aborted")
            if selected_lora and not completed:
                self._log_lora_selection(request, "aborted")
            self._engine_request_ids.pop(request.request_id, None)
            if selected_lora:
                await self.loras.release(selected_lora)
            if admitted:
                await self.admission.finish(request.request_id, session_id)

    async def GetServerInfo(self, request, context) -> server_pb2.ServerInfo:
        connector = self._connector_info()
        page_size = int(self.args.page_size or 0)
        max_total_tokens = int(self.scheduler_info.get("max_total_num_tokens", 0) or 0)
        capacity = server_pb2.DeploymentCapacity()
        if page_size:
            capacity.kv_block_size = page_size
            if max_total_tokens:
                capacity.total_kv_blocks = max_total_tokens // page_size
        max_running = self.args.max_running_requests or self.scheduler_info.get(
            "max_running_requests"
        )
        if max_running:
            capacity.max_running_requests = int(max_running)
        if self.args.max_prefill_tokens:
            capacity.max_batched_tokens = int(self.args.max_prefill_tokens)
        if self.args.enable_lora:
            max_loras = self.args.max_loaded_loras or self.args.max_loras_per_batch
            if max_loras:
                capacity.max_loras = int(max_loras)
        parallelism = server_pb2.ParallelismInfo(
            tensor_parallel_size=self.args.tp_size,
            pipeline_parallel_size=self.args.pp_size,
            data_parallel_size=self.args.dp_size,
        )
        info = server_pb2.ServerInfo(
            engine_name="sglang",
            engine_version=sglang_version,
            engine_role=self.role,
            instance_id=self.instance_id,
            supported_models=[self.served_model_name],
            parallelism=parallelism,
            kv_connector=connector,
            schema_revision=SCHEMA_REVISION,
            minimum_client_revision=MINIMUM_CLIENT_REVISION,
            schema_release=OPENENGINE_COMMIT,
            capacity=capacity,
        )
        info.extra.update(
            {
                "grammar_backend": self.args.grammar_backend,
                "disaggregation_transfer_backend": self.args.disaggregation_transfer_backend,
                "schedule_policy": self.args.schedule_policy,
            }
        )
        return info

    def _modalities(self) -> tuple[list[int], int | None]:
        processor = getattr(self.tm, "mm_processor", None)
        tokens = getattr(processor, "mm_tokens", None)
        modalities = []
        image_id = getattr(tokens, "image_token_id", None)
        if image_id is not None:
            modalities.append(input_pb2.MODALITY_IMAGE)
        if getattr(tokens, "video_token_id", None) is not None:
            modalities.append(input_pb2.MODALITY_VIDEO)
        if getattr(tokens, "audio_token_id", None) is not None:
            modalities.append(input_pb2.MODALITY_AUDIO)
        return modalities, image_id

    async def GetModelInfo(self, request, context) -> model_pb2.ModelInfo:
        if request.model and request.model not in self.model_aliases:
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Model {request.model!r} is not served by this SGLang runtime",
            )
        modalities, image_id = self._modalities()
        aggregate = modalities if self.role == server_pb2.ENGINE_ROLE_AGGREGATED else []
        pd_modalities = (
            [
                value
                for value in modalities
                if value in (input_pb2.MODALITY_IMAGE, input_pb2.MODALITY_VIDEO)
            ]
            if self.role
            in (server_pb2.ENGINE_ROLE_PREFILL, server_pb2.ENGINE_ROLE_DECODE)
            else []
        )
        mm = model_pb2.MultimodalCapabilities(
            aggregate_modalities=aggregate,
            prefill_decode_modalities=pd_modalities,
            source_types=[
                input_pb2.MEDIA_SOURCE_TYPE_URL,
                input_pb2.MEDIA_SOURCE_TYPE_DATA_URI,
                input_pb2.MEDIA_SOURCE_TYPE_RAW_BYTES,
            ],
            supports_per_request_media_options=True,
        )
        if image_id is not None and 0 <= int(image_id) <= (1 << 32) - 1:
            mm.routing_image_token_id = int(image_id)

        logprobs = model_pb2.LogprobCapabilities(
            supported=True,
            candidate_selection_modes=[
                model_pb2.CANDIDATE_TOKEN_SELECTION_MODE_TOP_N,
                model_pb2.CANDIDATE_TOKEN_SELECTION_MODE_TOKEN_IDS,
            ],
        )
        guided = model_pb2.GuidedDecodingCapabilities(
            supported=True,
            modes=[
                model_pb2.GUIDED_DECODING_MODE_JSON_SCHEMA,
                model_pb2.GUIDED_DECODING_MODE_REGEX,
                model_pb2.GUIDED_DECODING_MODE_EBNF_GRAMMAR,
                model_pb2.GUIDED_DECODING_MODE_STRUCTURAL_TAG,
                model_pb2.GUIDED_DECODING_MODE_CHOICE,
                model_pb2.GUIDED_DECODING_MODE_JSON_OBJECT,
            ],
        )
        generation = model_pb2.GenerationCapabilities(
            prompt_logprobs=logprobs,
            output_logprobs=logprobs,
            guided_decoding=guided,
            supports_priority=bool(self.args.enable_priority_scheduling),
            supports_stop_in_output=True,
            supports_cache_salt=True,
            supports_prefix_cache_bypass=False,
        )
        generation.max_num_sequences = int(self.args.max_running_requests or 1)
        aliases = sorted(self.model_aliases.difference({self.served_model_name}))
        return model_pb2.ModelInfo(
            model_id=self.model_id,
            served_model_name=self.served_model_name,
            served_model_aliases=aliases,
            max_context_length=self.tm.model_config.context_len,
            max_output_tokens=self.tm.model_config.context_len,
            tokenizer_modes=[self.args.tokenizer_mode],
            tokenizer=model_pb2.TokenizerInfo(
                source=self.args.tokenizer_path,
                mode=self.args.tokenizer_mode,
            ),
            supports_text_input=not self.args.skip_tokenizer_init,
            supports_token_ids_input=True,
            generation=generation,
            supports_lora=bool(self.args.enable_lora and self.args.dp_size == 1),
            supports_multimodal=bool(modalities),
            reasoning_parser=self.args.reasoning_parser or "",
            tool_call_parser=self.args.tool_call_parser or "",
            multimodal_capabilities=mm,
        )

    async def GetLoad(self, request, context) -> observability_pb2.LoadInfo:
        snapshots = await self.tm.get_loads(dp_rank=None)
        values = [value.to_dict() for value in snapshots]
        page_size = max(1, int(self.args.page_size or 1))
        _, sessions = await self.admission.snapshot()
        timestamp = (
            max((value.get("timestamp", 0) for value in values), default=0)
            or time.time()
        )
        info = observability_pb2.LoadInfo(
            instance_id=self.instance_id,
            timestamp_unix_nanos=int(timestamp * 1_000_000_000),
            running_requests=sum(
                int(value.get("num_running_reqs", 0)) for value in values
            ),
            queued_requests=sum(
                int(value.get("num_waiting_reqs", 0)) for value in values
            ),
            active_kv_sessions=sessions,
            used_kv_blocks=sum(
                math.ceil(int(value.get("num_used_tokens", 0)) / page_size)
                for value in values
            ),
            total_kv_blocks=sum(
                int(value.get("max_total_num_tokens", 0)) // page_size
                for value in values
            ),
            running_tokens=sum(
                int(value.get("num_used_tokens", 0)) for value in values
            ),
            waiting_tokens=sum(
                int(value.get("num_waiting_uncached_tokens", 0)) for value in values
            ),
        )
        if request.include_per_rank:
            for value in values:
                info.ranks.append(
                    observability_pb2.RankLoadInfo(
                        data_parallel_rank=int(value.get("dp_rank", 0)),
                        running_requests=int(value.get("num_running_reqs", 0)),
                        queued_requests=int(value.get("num_waiting_reqs", 0)),
                        used_kv_blocks=math.ceil(
                            int(value.get("num_used_tokens", 0)) / page_size
                        ),
                        total_kv_blocks=int(value.get("max_total_num_tokens", 0))
                        // page_size,
                    )
                )
        return info

    async def Health(self, request, context) -> lifecycle_pb2.HealthResponse:
        healthy = bool(self.runtime.health_check())
        if self.admission.draining:
            state = lifecycle_pb2.HEALTH_STATE_DRAINING
        else:
            state = (
                lifecycle_pb2.HEALTH_STATE_READY
                if healthy
                else lifecycle_pb2.HEALTH_STATE_NOT_READY
            )
        checks = [
            lifecycle_pb2.HealthCheck(name="grpc", state=state),
            lifecycle_pb2.HealthCheck(
                name="scheduler",
                state=(
                    lifecycle_pb2.HEALTH_STATE_READY
                    if healthy
                    else lifecycle_pb2.HEALTH_STATE_NOT_READY
                ),
            ),
            lifecycle_pb2.HealthCheck(name="model", state=state),
        ]
        if request.include_inference_probe:
            checks.append(
                lifecycle_pb2.HealthCheck(
                    name="inference_probe",
                    state=lifecycle_pb2.HEALTH_STATE_DEGRADED,
                    message="SGLang OpenEngine health probes are non-generating",
                )
            )
            if state == lifecycle_pb2.HEALTH_STATE_READY:
                state = lifecycle_pb2.HEALTH_STATE_DEGRADED
        return lifecycle_pb2.HealthResponse(state=state, checks=checks)

    async def Abort(self, request, context) -> lifecycle_pb2.AbortResponse:
        target = request.WhichOneof("target")
        if target == "request_id":
            active = set(await self.admission.active_request_ids())
            if request.request_id not in active:
                return lifecycle_pb2.AbortResponse(
                    status=lifecycle_pb2.ABORT_STATUS_ALREADY_FINISHED,
                    message="Request is no longer active",
                )
            self._abort_wire_request(request.request_id)
        elif target == "kv_session":
            request_ids = await self.admission.request_ids_for_session(
                request.kv_session.session_id
            )
            if not request_ids:
                return lifecycle_pb2.AbortResponse(
                    status=lifecycle_pb2.ABORT_STATUS_ALREADY_FINISHED,
                    message="KV session is no longer active",
                )
            for request_id in request_ids:
                self._abort_wire_request(request_id)
        elif target == "all_requests":
            active = await self.admission.active_request_ids()
            if not active:
                return lifecycle_pb2.AbortResponse(
                    status=lifecycle_pb2.ABORT_STATUS_ALREADY_FINISHED,
                    message="No OpenEngine requests are active",
                )
            self.runtime.abort(abort_all=True)
        else:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "Abort target is required"
            )
        return lifecycle_pb2.AbortResponse(
            status=lifecycle_pb2.ABORT_STATUS_ABORTED, message="Abort requested"
        )

    async def Drain(
        self, request, context
    ) -> AsyncGenerator[lifecycle_pb2.DrainResponse, None]:
        if request.stop_accepting_new_requests:
            await self.admission.start_drain()
        in_flight, sessions = await self.admission.snapshot()
        yield lifecycle_pb2.DrainResponse(
            state=lifecycle_pb2.DRAIN_STATE_STARTED,
            in_flight_requests=in_flight,
            open_kv_sessions=sessions,
        )
        deadline = (
            time.monotonic() + request.deadline_ms / 1000
            if request.HasField("deadline_ms")
            else None
        )
        while True:
            in_flight, sessions = await self.admission.snapshot()
            if in_flight == 0:
                yield lifecycle_pb2.DrainResponse(
                    state=lifecycle_pb2.DRAIN_STATE_COMPLETE,
                    in_flight_requests=0,
                    open_kv_sessions=sessions,
                )
                return
            if deadline is not None and time.monotonic() >= deadline:
                if request.abort_after_deadline:
                    self.runtime.abort(abort_all=True)
                    if await self.admission.wait_empty(timeout=5.0):
                        _, sessions = await self.admission.snapshot()
                        yield lifecycle_pb2.DrainResponse(
                            state=lifecycle_pb2.DRAIN_STATE_COMPLETE,
                            in_flight_requests=0,
                            open_kv_sessions=sessions,
                            message="Deadline reached; active requests were aborted",
                        )
                        return
                yield lifecycle_pb2.DrainResponse(
                    error=error_pb2.EngineError(
                        code=error_pb2.ERROR_CODE_INTERNAL,
                        message="Drain deadline expired with active requests",
                    ),
                    in_flight_requests=in_flight,
                    open_kv_sessions=sessions,
                )
                return
            yield lifecycle_pb2.DrainResponse(
                state=lifecycle_pb2.DRAIN_STATE_IN_PROGRESS,
                in_flight_requests=in_flight,
                open_kv_sessions=sessions,
            )
            wait = 0.25
            if deadline is not None:
                wait = min(wait, max(0.0, deadline - time.monotonic()))
            await self.admission.wait_empty(timeout=wait)

    def _require_lora(self) -> None:
        if not self.args.enable_lora:
            raise ValueError("SGLang was not launched with --enable-lora")
        if self.args.dp_size != 1:
            raise ValueError("SGLang dynamic LoRA is supported only with DP=1")

    async def LoadLora(self, request, context) -> lora_pb2.LoadLoraResponse:
        try:
            self._require_lora()
            adapter, already_loaded = await self.loras.load(request.adapter)
        except ValueError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        return lora_pb2.LoadLoraResponse(adapter=adapter, already_loaded=already_loaded)

    async def UnloadLora(self, request, context) -> lora_pb2.UnloadLoraResponse:
        try:
            self._require_lora()
            adapter = await self.loras.unload(request.lora_name)
        except KeyError:
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"LoRA {request.lora_name!r} is not logically loaded",
            )
        except ValueError as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        return lora_pb2.UnloadLoraResponse(adapter=adapter)

    async def ListLoras(self, request, context) -> lora_pb2.ListLorasResponse:
        return lora_pb2.ListLorasResponse(adapters=await self.loras.list())

    def _connector_info(self) -> kv_pb2.KvConnectorInfo:
        enabled = self.role in (
            server_pb2.ENGINE_ROLE_PREFILL,
            server_pb2.ENGINE_ROLE_DECODE,
        )
        connector = kv_pb2.KvConnectorInfo(
            enabled=enabled,
            transfer_backend=(
                str(self.args.disaggregation_transfer_backend) if enabled else ""
            ),
            supported_protocols=["tcp"] if enabled else [],
            supports_remote_prefill=enabled,
            supports_decode_pull=enabled,
            supports_abort_cleanup=enabled,
            supports_drain=enabled,
            schema_version=1,
            handoff_profile=HANDOFF_PROFILE if enabled else "",
            supports_client_bootstrap=enabled,
        )
        if self.role == server_pb2.ENGINE_ROLE_PREFILL:
            connector.local_endpoints.append(
                kv_pb2.KvEndpoint(
                    host=self.advertised_host,
                    port=self.args.disaggregation_bootstrap_port,
                    protocol="tcp",
                )
            )
        return connector

    async def GetKvConnectorInfo(self, request, context) -> kv_pb2.KvConnectorInfo:
        return self._connector_info()

    async def GetKvEventSources(
        self, request, context
    ) -> kv_pb2.GetKvEventSourcesResponse:
        descriptor = self.args.describe_kv_events_publisher()
        if descriptor is None:
            return kv_pb2.GetKvEventSourcesResponse()
        dp_size = int(descriptor["dp_size"])
        ranks = list(request.data_parallel_ranks) or list(range(dp_size))
        if len(set(ranks)) != len(ranks):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "data_parallel_ranks must not contain duplicates",
            )
        response = kv_pb2.GetKvEventSourcesResponse()
        host = descriptor["endpoint_host"]
        if host in ("*", "0.0.0.0", "::", "[::]"):
            host = self.advertised_host
        for rank in ranks:
            if rank >= dp_size:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"data parallel rank {rank} is outside [0, {dp_size})",
                )
            port = int(descriptor["endpoint_port_base"]) + rank
            if port > 65535:
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"KV event port for rank {rank} exceeds 65535",
                )
            response.sources.append(
                kv_pb2.KvEventSource(
                    transport=str(descriptor["publisher"]),
                    endpoint_addr=kv_pb2.KvEndpoint(
                        host=host, port=port, protocol="tcp"
                    ),
                    topic=str(descriptor.get("topic", "")),
                    data_parallel_rank=rank,
                    encoding="msgpack",
                    schema_version=1,
                )
            )
        return response

    async def SubscribeKvEvents(self, request, context):
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "SGLang advertises direct per-rank ZMQ/msgpack KV event sources",
        )
        if False:
            yield kv_pb2.SubscribeKvEventsResponse()


__all__ = ["OpenEngineServicer"]
