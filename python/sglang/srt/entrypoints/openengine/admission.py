"""Process-wide request admission shared by HTTP and OpenEngine."""

import asyncio
import json
from collections import Counter


class DrainingError(RuntimeError):
    """Raised when a new request reaches a draining runtime."""


class ProcessAdmission:
    """Track requests without taking ownership of the underlying engine."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._requests: set[str] = set()
        self._request_sessions: dict[str, str] = {}
        self._external_requests = 0
        self._sessions: Counter[str] = Counter()
        self._draining = False

    @property
    def draining(self) -> bool:
        return self._draining

    async def admit(self, request_id: str, session_id: str = "") -> None:
        async with self._condition:
            if self._draining:
                raise DrainingError("SGLang is draining")
            if request_id in self._requests:
                raise ValueError(f"request_id {request_id!r} is already active")
            self._requests.add(request_id)
            self._request_sessions[request_id] = session_id
            if session_id:
                self._sessions[session_id] += 1

    async def finish(self, request_id: str, session_id: str = "") -> None:
        async with self._condition:
            self._requests.discard(request_id)
            self._request_sessions.pop(request_id, None)
            if session_id and self._sessions[session_id] > 0:
                self._sessions[session_id] -= 1
                if self._sessions[session_id] == 0:
                    del self._sessions[session_id]
            self._condition.notify_all()

    async def admit_external(self) -> None:
        async with self._condition:
            if self._draining:
                raise DrainingError("SGLang is draining")
            self._external_requests += 1

    async def finish_external(self) -> None:
        async with self._condition:
            if self._external_requests > 0:
                self._external_requests -= 1
            self._condition.notify_all()

    async def start_drain(self) -> None:
        async with self._condition:
            self._draining = True
            self._condition.notify_all()

    async def snapshot(self) -> tuple[int, int]:
        async with self._condition:
            return len(self._requests) + self._external_requests, len(self._sessions)

    async def active_request_ids(self) -> tuple[str, ...]:
        async with self._condition:
            return tuple(self._requests)

    async def request_ids_for_session(self, session_id: str) -> tuple[str, ...]:
        async with self._condition:
            return tuple(
                request_id
                for request_id, value in self._request_sessions.items()
                if value == session_id
            )

    async def wait_empty(self, timeout: float | None = None) -> bool:
        async def _wait() -> None:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: not self._requests and self._external_requests == 0
                )

        try:
            if timeout is None:
                await _wait()
            else:
                await asyncio.wait_for(_wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return True


_INFERENCE_PATHS = frozenset(
    {
        "/generate",
        "/encode",
        "/classify",
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/classify",
        "/v1/rerank",
        "/v1/score",
        "/v1/responses",
        "/v1/audio/transcriptions",
        "/api/chat",
        "/api/generate",
        "/v1/messages",
        "/invocations",
        "/vertex_generate",
    }
)


class HttpAdmissionMiddleware:
    """Make an OpenEngine drain apply to the sibling HTTP inference surface."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") not in ("POST", "PUT")
            or scope.get("path") not in _INFERENCE_PATHS
        ):
            await self.app(scope, receive, send)
            return

        fastapi_app = scope.get("app")
        admission = getattr(
            getattr(fastapi_app, "state", None), "openengine_admission", None
        )
        if admission is None:
            await self.app(scope, receive, send)
            return

        try:
            await admission.admit_external()
        except DrainingError as error:
            body = json.dumps(
                {
                    "error": {
                        "message": str(error),
                        "type": "service_unavailable",
                        "code": 503,
                        "retryable": True,
                    }
                }
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        finished = False

        async def tracked_send(message) -> None:
            nonlocal finished
            await send(message)
            if (
                not finished
                and message.get("type") == "http.response.body"
                and not message.get("more_body", False)
            ):
                finished = True
                await admission.finish_external()

        try:
            await self.app(scope, receive, tracked_send)
        finally:
            if not finished:
                await admission.finish_external()


__all__ = ["DrainingError", "HttpAdmissionMiddleware", "ProcessAdmission"]
