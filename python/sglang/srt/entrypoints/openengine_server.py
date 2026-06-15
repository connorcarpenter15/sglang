# SPDX-License-Identifier: Apache-2.0
"""
OpenEngine v1 gRPC server entrypoint for SGLang (Dynamo sidecar mode).

Brings up the SGLang engine (scheduler + TokenizerManager), wraps it in a
minimal ``RuntimeHandle`` implementing the contract the Rust ``_core`` server
bridge expects, and mounts the OpenEngine service on ``--openengine-port``.

The Rust side (``sglang.srt.grpc._core.start_openengine_server``) drives this
handle from its own Tokio runtime: it calls ``submit_request`` (non-blocking,
kicks off generation and streams chunks back through ``chunk_callback``) plus a
handful of synchronous info/control methods. We run the engine's asyncio loop
in a background thread so concurrent ``generate`` streams are served off the
single TokenizerManager, mirroring how the HTTP/gRPC servers use it.
"""

import asyncio
import json
import logging
import threading
import time

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import GenerateReqInput

logger = logging.getLogger(__name__)

# Keys we forward from the Rust-built request dict into GenerateReqInput.
# Anything else is dropped defensively (GenerateReqInput rejects unknown kwargs).
_GENERATE_REQ_KEYS = frozenset(
    {
        "rid",
        "text",
        "input_ids",
        "image_data",
        "sampling_params",
        "stream",
        "routed_dp_rank",
        "bootstrap_host",
        "bootstrap_port",
        "bootstrap_room",
        "return_logprob",
        "top_logprobs_num",
        "logprob_start_len",
    }
)


class RuntimeHandle:
    """Adapts an ``sgl.Engine`` to the method surface the Rust ``_core`` bridge
    calls on its ``runtime_handle`` (see rust/sglang-grpc/src/bridge.rs)."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.tokenizer_manager = engine.tokenizer_manager
        self.server_args = engine.server_args
        # Disaggregation: prefill workers advertise a bootstrap host/port and mint
        # a per-request room; the decode peer connects to it. Resolved once here.
        self.role = getattr(self.server_args, "disaggregation_mode", "null") or "null"
        self.bootstrap_host = None
        self.bootstrap_port = None
        if self.role == "prefill":
            port = getattr(self.server_args, "disaggregation_bootstrap_port", None)
            if port:
                try:
                    from sglang.srt.utils.network import get_local_ip_auto

                    self.bootstrap_host = get_local_ip_auto()
                except Exception:  # noqa: BLE001
                    self.bootstrap_host = "127.0.0.1"
                self.bootstrap_port = int(port)
        # Run the engine loop in a background thread so submit_request (called
        # from the Rust runtime) can schedule concurrent generations onto it.
        self.loop = engine.loop
        self._thread = threading.Thread(
            target=self._run_loop, name="sglang-openengine-loop", daemon=True
        )
        self._thread.start()
        # Start the TokenizerManager's receive/handle tasks on our loop.
        fut = asyncio.run_coroutine_threadsafe(self._ensure_handle_loop(), self.loop)
        try:
            fut.result(timeout=30)
        except Exception as e:  # noqa: BLE001
            logger.warning("auto_create_handle_loop failed (continuing): %s", e)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _ensure_handle_loop(self):
        fn = getattr(self.tokenizer_manager, "auto_create_handle_loop", None)
        if callable(fn):
            fn()

    # --- generation -------------------------------------------------------

    def submit_request(self, req_type, req_dict, chunk_callback):
        """Non-blocking: schedule generation on the engine loop. Tokens stream
        back through chunk_callback(chunk_dict, finished, error)."""
        asyncio.run_coroutine_threadsafe(
            self._drive(req_dict, chunk_callback), self.loop
        )

    async def _drive(self, req_dict, chunk_callback):
        try:
            kwargs = {k: v for k, v in req_dict.items() if k in _GENERATE_REQ_KEYS}
            # Prefill: mint a per-request bootstrap room and inject the triple so
            # SGLang registers it; the terminal chunk carries it back for PrefillReady.
            disagg = None
            if self.role == "prefill" and self.bootstrap_host and self.bootstrap_port:
                # Bootstrap path: the router already assigned host/port/room
                # (mapped from kv_session into the dict). Completed path: none
                # provided, so mint our own.
                if "bootstrap_room" not in kwargs:
                    import random

                    kwargs["bootstrap_host"] = self.bootstrap_host
                    kwargs["bootstrap_port"] = self.bootstrap_port
                    kwargs["bootstrap_room"] = random.randint(0, 2**63 - 1)
                disagg = {
                    "bootstrap_host": kwargs["bootstrap_host"],
                    "bootstrap_port": kwargs["bootstrap_port"],
                    "bootstrap_room": kwargs["bootstrap_room"],
                    "transfer_backend": getattr(
                        self.server_args, "disaggregation_transfer_backend", "nixl"
                    ),
                }
            obj = GenerateReqInput(**kwargs)
            gen = self.tokenizer_manager.generate_request(obj, None)
            # SGLang streams CUMULATIVE output_ids per chunk; the OpenEngine
            # TokenOutput contract (and the Dynamo sidecar) expect only the NEW
            # tokens. Track how many we've forwarded and emit the suffix.
            sent = 0
            async for out in gen:
                meta = out.get("meta_info", {}) or {}
                finished = bool(meta.get("finish_reason"))
                full_ids = out.get("output_ids", []) or []
                new_ids = full_ids[sent:]
                sent = len(full_ids)
                if finished and disagg is not None:
                    meta = dict(meta)
                    meta["disaggregated_params"] = disagg
                chunk = {
                    "output_ids": new_ids,
                    "text": "",
                    "meta_info": meta,
                }
                chunk_callback(chunk, finished, None)
                if finished:
                    break
        except Exception as e:  # noqa: BLE001
            logger.warning("openengine generate failed: %s", e, exc_info=True)
            try:
                chunk_callback({}, True, repr(e))
            except Exception:  # noqa: BLE001
                pass

    def abort(self, rid, abort_all):
        try:
            self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)
        except Exception as e:  # noqa: BLE001
            logger.debug("abort(%s) failed: %s", rid, e)

    # --- info / control ---------------------------------------------------

    def get_model_info(self) -> str:
        sa = self.server_args
        is_mm = bool(getattr(self.tokenizer_manager, "mm_processor", None))
        page_size = int(getattr(sa, "page_size", 0) or 0)
        # KV capacity for the router: total token budget / page size.
        max_total_tokens = 0
        sched = getattr(self.engine, "_scheduler_init_result", None)
        infos = getattr(sched, "scheduler_infos", None) if sched is not None else None
        if infos:
            try:
                max_total_tokens = int(infos[0].get("max_total_num_tokens", 0) or 0)
            except Exception:  # noqa: BLE001
                max_total_tokens = 0
        total_kv_blocks = (max_total_tokens // page_size) if page_size else 0
        return json.dumps(
            {
                "model_path": sa.model_path,
                "served_model_name": getattr(sa, "served_model_name", None)
                or sa.model_path,
                "is_multimodal": int(is_mm),
                "page_size": page_size,
                "total_kv_blocks": total_kv_blocks,
                "max_running_requests": int(getattr(sa, "max_running_requests", 0) or 0),
                "max_prefill_tokens": int(getattr(sa, "max_prefill_tokens", 0) or 0),
            }
        )

    def get_server_info(self) -> str:
        sa = self.server_args
        return json.dumps(
            {
                "tp_size": getattr(sa, "tp_size", 1),
                "dp_size": getattr(sa, "dp_size", 1) or 1,
                "pp_size": getattr(sa, "pp_size", 1) or 1,
                "disaggregation_mode": getattr(sa, "disaggregation_mode", "null"),
                "max_running_requests": getattr(sa, "max_running_requests", 0) or 0,
                "max_prefill_tokens": getattr(sa, "max_prefill_tokens", 0) or 0,
                "page_size": getattr(sa, "page_size", 0) or 0,
                "version": getattr(__import__("sglang"), "__version__", "unknown"),
                "kv_event_sources": self._kv_event_sources(),
                "bootstrap_host": self.bootstrap_host or "",
                "bootstrap_port": self.bootstrap_port or 0,
                "disaggregation_transfer_backend": getattr(
                    sa, "disaggregation_transfer_backend", "nixl"
                ),
            }
        )

    def _kv_event_sources(self):
        """Derive routable ZMQ KV-event publisher descriptors from SGLang's
        --kv-events-config. The KV router subscribes to these directly and
        parses SGLang's native msgpack KVEventBatch. Empty if not configured."""
        cfg = getattr(self.server_args, "kv_events_config", None)
        if not cfg:
            return []
        try:
            kv = json.loads(cfg)
        except Exception:  # noqa: BLE001
            return []
        ep = kv.get("endpoint")
        if not ep:
            return []
        try:
            from sglang.srt.utils.network import get_local_ip_auto

            ip = get_local_ip_auto()
        except Exception:  # noqa: BLE001
            ip = "127.0.0.1"
        # Replace bind wildcards with a routable host so a remote KV router can dial.
        ep_routable = ep.replace("://*:", f"://{ip}:").replace("://0.0.0.0:", f"://{ip}:")
        return [
            {
                "transport": "zmq",
                "endpoint": ep_routable,
                "topic": kv.get("topic", ""),
                "dp_rank": 0,
                "encoding": "msgpack",
            }
        ]

    def health_check(self) -> bool:
        return True

    def get_load(self, callback, dp_rank):
        # Best-effort point-in-time load. Refined later; for now report idle.
        payload = json.dumps(
            {"num_running_reqs": 0, "num_waiting_reqs": 0, "num_used_tokens": 0}
        ).encode()
        try:
            callback(payload, True, None)
        except Exception as e:  # noqa: BLE001
            logger.debug("get_load callback failed: %s", e)


def serve_openengine(server_args):
    """Launch the SGLang engine + the OpenEngine v1 gRPC server, and block."""
    from sglang.srt.grpc import _core

    host = server_args.openengine_host or server_args.host or "127.0.0.1"
    port = server_args.openengine_port

    logger.info("Starting SGLang engine for OpenEngine sidecar...")
    engine = Engine(server_args=server_args)
    handle = RuntimeHandle(engine)

    logger.info("Mounting OpenEngine v1 gRPC server on %s:%d", host, port)
    server = _core.start_openengine_server(host, port, handle)

    try:
        while server.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down OpenEngine server")
        try:
            server.shutdown()
        finally:
            engine.shutdown()
