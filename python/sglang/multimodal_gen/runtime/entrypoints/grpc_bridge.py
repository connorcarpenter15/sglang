"""Native gRPC adapter for SGLang's image and video generation runtime."""

import asyncio
import json
import logging
import threading
from concurrent.futures import Future
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class MediaRuntimeHandle:
    """Bridge Rust callbacks onto the diffusion server's asyncio loop.

    The ASGI transport deliberately reuses the public OpenAI-shaped handlers.
    This keeps validation, output persistence, and model-specific defaults
    identical between HTTP and native gRPC without another network hop.
    """

    def __init__(self, app, server_args, event_loop: asyncio.AbstractEventLoop):
        self.app = app
        self.server_args = server_args
        self._event_loop = event_loop
        self._tasks: Dict[str, Future] = {}
        self._tasks_lock = threading.Lock()
        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://sglang-grpc.local",
            trust_env=False,
        )

    @property
    def _is_image_runtime(self) -> bool:
        return self.server_args.pipeline_config.task_type.is_image_gen()

    def submit_media_generate(
        self,
        *,
        rid: str,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self._generate(rid, json_body, trace_headers or {}, chunk_callback),
            self._event_loop,
        )
        with self._tasks_lock:
            self._tasks[rid] = future

        def _discard(completed: Future) -> None:
            with self._tasks_lock:
                if self._tasks.get(rid) is completed:
                    self._tasks.pop(rid, None)

        future.add_done_callback(_discard)

    async def _generate(self, rid, json_body, trace_headers, chunk_callback):
        try:
            payload = json.loads(json_body)
            endpoint = (
                "/v1/images/generations" if self._is_image_runtime else "/v1/videos"
            )
            response = await self._client.post(
                endpoint,
                json=payload,
                headers=trace_headers,
            )
            response_body: Any = response.content
            status_code = response.status_code

            if not self._is_image_runtime and response.is_success:
                job = response.json()
                job_id = job.get("id")
                while job_id and job.get("status") in {"queued", "in_progress"}:
                    await asyncio.sleep(0.25)
                    poll = await self._client.get(f"/v1/videos/{job_id}")
                    status_code = poll.status_code
                    response_body = poll.content
                    if not poll.is_success:
                        break
                    job = poll.json()
                if job.get("status") == "failed":
                    status_code = 500

            chunk_callback(
                response_body,
                finished=True,
                status_code=status_code,
            )
        except asyncio.CancelledError:
            chunk_callback(b"", finished=True, error=f"Media request {rid} cancelled")
            raise
        except Exception as exc:
            logger.exception("Native gRPC media request %s failed", rid)
            chunk_callback(b"", finished=True, error=str(exc))

    def abort(self, rid: str = "", abort_all: bool = False) -> None:
        with self._tasks_lock:
            if abort_all:
                futures = list(self._tasks.values())
            else:
                future = self._tasks.get(rid)
                futures = [future] if future is not None else []
        for future in futures:
            future.cancel()

    def get_model_info(self) -> str:
        task_type = self.server_args.pipeline_config.task_type
        return json.dumps(
            {
                "model_path": self.server_args.model_path,
                "is_generation": True,
                "model_type": "diffusion",
                "task_type": task_type.name,
                "is_image_gen": task_type.is_image_gen(),
            }
        )

    def get_server_info(self) -> str:
        return json.dumps(
            {
                "model_path": self.server_args.model_path,
                "served_model_name": self.server_args.model_id
                or self.server_args.model_path,
                "runtime_kind": "image" if self._is_image_runtime else "video",
                "tp_size": self.server_args.tp_size,
                "dp_size": self.server_args.dp_size,
            }
        )

    def health_check(self) -> bool:
        warmup_done = getattr(self.app.state, "server_warmup_done", None)
        return (
            not self._client.is_closed
            and warmup_done is not None
            and warmup_done.is_set()
        )

    async def aclose(self) -> None:
        with self._tasks_lock:
            futures = list(self._tasks.values())
        for future in futures:
            future.cancel()
        if futures:
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
                return_exceptions=True,
            )
        await self._client.aclose()


def start_native_grpc_server(server_args, runtime_handle):
    try:
        from sglang.srt.grpc import _core as grpc_native
    except ImportError as exc:
        raise RuntimeError(
            "Native gRPC extension (sglang.srt.grpc._core) is required when "
            "--grpc-port is set for multimodal generation"
        ) from exc

    handle = grpc_native.start_server(
        host=server_args.host,
        port=server_args.grpc_port,
        runtime_handle=runtime_handle,
        worker_threads=server_args.grpc_worker_threads,
        response_timeout_secs=server_args.grpc_response_timeout_secs,
        require_tokenizer=False,
        max_message_size=server_args.grpc_max_message_size,
    )
    logger.info(
        "Native media gRPC server started on %s:%s",
        server_args.host,
        server_args.grpc_port,
    )
    return handle
