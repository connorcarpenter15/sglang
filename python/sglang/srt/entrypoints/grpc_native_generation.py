"""Native generate/embed request adapter for the Rust gRPC bridge."""

import asyncio
import logging
from typing import Optional

from pydantic import ValidationError

logger = logging.getLogger(__name__)


class NativeGenerationAdapter:
    """Generation-specific RuntimeHandle behavior kept out of the control adapter."""

    @staticmethod
    def _generation_error_meta(error: Exception) -> dict:
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            status_code = (
                400 if isinstance(error, (ValidationError, ValueError)) else 500
            )
        detail = getattr(error, "detail", None)
        return {
            "finish_reason": {
                "type": "error",
                "status_code": int(status_code),
                "message": str(detail if detail is not None else error),
            }
        }

    async def _send_generation_error(
        self,
        chunk_callback,
        ready_event: Optional[asyncio.Event],
        *,
        error: Optional[Exception] = None,
        meta_info: Optional[dict] = None,
        timeout_abort_rid,
        timeout_abort_lifecycle_id,
    ) -> None:
        if meta_info is None:
            if error is None:
                raise ValueError("structured generation error requires error metadata")
            meta_info = self._generation_error_meta(error)
        keep_going = await self._send_with_backpressure(
            chunk_callback,
            ready_event,
            {
                "output_ids": [],
                "delta_output_ids": [],
                "meta_info": meta_info,
            },
            finished=True,
            timeout_abort_rid=timeout_abort_rid,
            timeout_abort_lifecycle_id=timeout_abort_lifecycle_id,
        )
        if not keep_going:
            self._abort_request_id(
                timeout_abort_rid,
                timeout_abort_lifecycle_id,
            )

    @staticmethod
    def _scheduler_error_meta(chunk: dict) -> Optional[dict]:
        meta_info = chunk.get("meta_info") or {}
        finish_reason = meta_info.get("finish_reason")
        if isinstance(finish_reason, dict) and finish_reason.get("type") in (
            "abort",
            "error",
        ):
            return {"finish_reason": finish_reason}
        return None

    async def _run_generate(
        self,
        obj,
        chunk_callback,
        stream: bool,
        request,
        *,
        structured_errors: bool = False,
        lifecycle_id=None,
    ):
        ready_event = None
        gen = None
        sampling_params = getattr(obj, "sampling_params", None)
        if isinstance(sampling_params, dict):
            parallel_sample_num = sampling_params.get("n", 1)
        elif isinstance(sampling_params, list) and sampling_params:
            parallel_sample_num = sampling_params[0].get("n", 1)
        else:
            parallel_sample_num = getattr(obj, "parallel_sample_num", 1)
        expected_choices = max(1, int(parallel_sample_num))
        terminal_choices = set()
        try:
            ready_event = self._install_on_ready(chunk_callback)
            generate_kwargs = {
                "request": request,
                "request_lifecycle_id": lifecycle_id,
            }
            if structured_errors:
                generate_kwargs["yield_scheduler_errors"] = True
            gen = self.tokenizer_manager.generate_request(obj, **generate_kwargs)
            if stream:
                incremental = bool(
                    getattr(
                        getattr(self.tokenizer_manager, "server_args", None),
                        "incremental_streaming_output",
                        False,
                    )
                )
                output_counts = {index: 0 for index in range(expected_choices)}
                async for chunk in gen:
                    if (
                        structured_errors
                        and (error_meta := self._scheduler_error_meta(chunk))
                        is not None
                    ):
                        await self._send_generation_error(
                            chunk_callback,
                            ready_event,
                            meta_info=error_meta,
                            timeout_abort_rid=obj.rid,
                            timeout_abort_lifecycle_id=lifecycle_id,
                        )
                        return
                    output_index = int(chunk.get("index") or 0)
                    if not 0 <= output_index < expected_choices:
                        self._abort_request_id(obj.rid, lifecycle_id)
                        self._send_native_error(
                            chunk_callback,
                            f"output index {output_index} is outside 0..{expected_choices}",
                        )
                        return
                    if output_index in terminal_choices:
                        self._abort_request_id(obj.rid, lifecycle_id)
                        self._send_native_error(
                            chunk_callback,
                            f"data after terminal for output {output_index}",
                        )
                        return
                    choice_finished = (
                        chunk.get("meta_info", {}).get("finish_reason") is not None
                    )
                    if choice_finished:
                        terminal_choices.add(output_index)
                    finished = len(terminal_choices) == expected_choices
                    callback_chunk = dict(chunk)
                    callback_chunk["index"] = output_index
                    output_ids = chunk.get("output_ids") or []
                    if incremental:
                        callback_chunk["delta_output_ids"] = output_ids
                    else:
                        previous_count = output_counts[output_index]
                        delta_start = (
                            previous_count if previous_count <= len(output_ids) else 0
                        )
                        callback_chunk["delta_output_ids"] = output_ids[delta_start:]
                        output_counts[output_index] = len(output_ids)
                    keep_going = await self._send_with_backpressure(
                        chunk_callback,
                        ready_event,
                        callback_chunk,
                        finished=finished,
                        timeout_abort_rid=obj.rid,
                        timeout_abort_lifecycle_id=lifecycle_id,
                    )
                    if finished or not keep_going:
                        return
                # Defensive: generator exited without a finish_reason chunk.
                missing = sorted(set(range(expected_choices)) - terminal_choices)
                error = RuntimeError(
                    f"SGLang stream ended without terminal choices: {missing}"
                )
                if structured_errors:
                    await self._send_generation_error(
                        chunk_callback,
                        ready_event,
                        error=error,
                        timeout_abort_rid=obj.rid,
                        timeout_abort_lifecycle_id=lifecycle_id,
                    )
                else:
                    self._send_native_error(chunk_callback, str(error))
            else:
                result = await gen.__anext__()
                chunks = result if isinstance(result, list) else [result]
                if structured_errors:
                    for chunk in chunks:
                        if (
                            error_meta := self._scheduler_error_meta(chunk)
                        ) is not None:
                            await self._send_generation_error(
                                chunk_callback,
                                ready_event,
                                meta_info=error_meta,
                                timeout_abort_rid=obj.rid,
                                timeout_abort_lifecycle_id=lifecycle_id,
                            )
                            return
                if len(chunks) != expected_choices:
                    error = RuntimeError(
                        f"SGLang returned {len(chunks)} choices; expected {expected_choices}"
                    )
                    if structured_errors:
                        await self._send_generation_error(
                            chunk_callback,
                            ready_event,
                            error=error,
                            timeout_abort_rid=obj.rid,
                            timeout_abort_lifecycle_id=lifecycle_id,
                        )
                    else:
                        self._send_native_error(chunk_callback, str(error))
                    return
                for index, chunk in enumerate(chunks):
                    callback_chunk = dict(chunk)
                    callback_chunk.setdefault("index", index)
                    output_ids = list(chunk.get("output_ids") or [])
                    callback_chunk["output_ids"] = output_ids
                    callback_chunk["delta_output_ids"] = output_ids
                    terminal_choices.add(index)
                    keep_going = await self._send_with_backpressure(
                        chunk_callback,
                        ready_event,
                        callback_chunk,
                        finished=index == len(chunks) - 1,
                        timeout_abort_rid=obj.rid,
                        timeout_abort_lifecycle_id=lifecycle_id,
                    )
                    if not keep_going:
                        return
        except StopAsyncIteration:
            error = RuntimeError("SGLang returned no generation result")
            if structured_errors:
                await self._send_generation_error(
                    chunk_callback,
                    ready_event,
                    error=error,
                    timeout_abort_rid=obj.rid,
                    timeout_abort_lifecycle_id=lifecycle_id,
                )
            else:
                self._send_native_error(chunk_callback, str(error))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("gRPC generate error for rid=%s: %s", obj.rid, e)
            if structured_errors:
                await self._send_generation_error(
                    chunk_callback,
                    ready_event,
                    error=e,
                    timeout_abort_rid=obj.rid,
                    timeout_abort_lifecycle_id=lifecycle_id,
                )
            else:
                self._send_native_error(chunk_callback, str(e))
        finally:
            if gen is not None:
                await gen.aclose()
            self._uninstall_on_ready(chunk_callback)

    async def _run_embed(self, obj, chunk_callback, request, *, lifecycle_id=None):
        try:
            gen = self.tokenizer_manager.generate_request(
                obj,
                request=request,
                request_lifecycle_id=lifecycle_id,
            )
            result = await gen.__anext__()
            self._safe_callback(chunk_callback, result, finished=True)
        except StopAsyncIteration:
            self._safe_callback(chunk_callback, {}, finished=True)
        except Exception as e:
            logger.error("gRPC embed error for rid=%s: %s", obj.rid, e)
            self._send_native_error(chunk_callback, str(e))
