"""Lifecycle wrapper for SGLang's optional OpenEngine sibling server."""

import asyncio
import logging
import os

import grpc
from openengine.v1 import openengine_pb2_grpc

from sglang.srt.utils.network import get_local_ip_auto

from .admission import ProcessAdmission
from .servicer import OpenEngineServicer

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 256 * 1024 * 1024
_WILDCARD_HOSTS = {"*", "0.0.0.0", "::", "[::]"}


def resolve_advertised_host(runtime_handle, bind_host: str) -> str:
    """Resolve a connectable host while keeping loopback local deployments simple."""
    configured = runtime_handle.server_args.openengine_advertise_host
    advertised = (
        configured or os.getenv("OPENENGINE_ADVERTISED_HOST") or os.getenv("POD_IP")
    )
    if advertised is None:
        advertised = (
            get_local_ip_auto(fallback=runtime_handle.server_args.host)
            if bind_host in _WILDCARD_HOSTS
            else bind_host
        )
    if advertised in _WILDCARD_HOSTS:
        raise RuntimeError(
            "OpenEngine endpoint discovery requires a connectable advertised host; "
            "set --openengine-advertise-host or OPENENGINE_ADVERTISED_HOST"
        )
    return advertised


class OpenEngineServer:
    """A gRPC server that shares, and never shuts down, the live SGLang runtime."""

    def __init__(self, runtime_handle, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.admission = ProcessAdmission()
        self._server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", _MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", _MAX_MESSAGE_LENGTH),
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
            ]
        )
        advertised_host = resolve_advertised_host(runtime_handle, host)
        self.servicer = OpenEngineServicer(
            runtime_handle=runtime_handle,
            admission=self.admission,
            advertised_host=advertised_host,
        )
        openengine_pb2_grpc.add_InferenceServicer_to_server(self.servicer, self._server)
        openengine_pb2_grpc.add_ControlServicer_to_server(self.servicer, self._server)
        bound = self._server.add_insecure_port(f"{host}:{port}")
        if bound == 0:
            raise RuntimeError(f"Failed to bind OpenEngine server to {host}:{port}")
        self.port = bound if port == 0 else port

    async def start(self) -> None:
        await self._server.start()
        logger.info("OpenEngine sibling server started on %s:%d", self.host, self.port)

    async def stop(self, grace: float = 60.0) -> None:
        await self.admission.start_drain()
        started = asyncio.get_running_loop().time()
        await self.admission.wait_empty(timeout=grace)
        elapsed = asyncio.get_running_loop().time() - started
        await self._server.stop(grace=max(0.0, grace - elapsed))
        elapsed = asyncio.get_running_loop().time() - started
        await self.servicer.close(timeout=max(0.0, grace - elapsed))
        logger.info("OpenEngine sibling server stopped")


__all__ = ["OpenEngineServer", "resolve_advertised_host"]
