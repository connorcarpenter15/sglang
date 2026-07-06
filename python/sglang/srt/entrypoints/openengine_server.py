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
import os
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
        "lora_path",
    }
)


def _resolve_local_ip() -> str:
    """Best-effort routable IP for this node, robust across sglang versions.

    The OpenEngine sidecar overlays onto the container's OWN sglang, whose util
    layout differs from the fork's: ``sglang.srt.utils.network`` does not exist
    in 0.5.6.post2 (the helper lives directly in ``sglang.srt.utils``). Try both
    import paths, then fall back to the standard egress-interface probe. A
    routable host (NOT loopback) is required so a remote decode peer / KV router
    can dial the prefill bootstrap server."""
    import importlib

    for mod_path, fn_name in (
        ("sglang.srt.utils.network", "get_local_ip_auto"),
        ("sglang.srt.utils", "get_local_ip_auto"),
        ("sglang.srt.utils", "get_local_ip_by_remote"),
    ):
        try:
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                ip = fn()
                if ip and ip != "127.0.0.1":
                    return ip
        except Exception:  # noqa: BLE001
            continue
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip:
            return ip
    except Exception:  # noqa: BLE001
        pass
    return "127.0.0.1"


_GENREQ_ACCEPTED = None


def _genreq_accepted_params() -> frozenset:
    """Parameter names accepted by the running sglang's ``GenerateReqInput``.

    The overlay runs against the container's OWN sglang, whose GenerateReqInput
    fields can lag the fork (e.g. 0.5.6.post2 has no ``routed_dp_rank``). Filter
    request kwargs against this set so version-skew fields are dropped instead of
    raising ``unexpected keyword argument``. Cached (fixed per process)."""
    global _GENREQ_ACCEPTED
    if _GENREQ_ACCEPTED is None:
        import inspect

        try:
            _GENREQ_ACCEPTED = frozenset(
                p
                for p in inspect.signature(GenerateReqInput.__init__).parameters
                if p != "self"
            )
        except (ValueError, TypeError):
            _GENREQ_ACCEPTED = frozenset()
    return _GENREQ_ACCEPTED


class RuntimeHandle:
    """Adapts an ``sgl.Engine`` to the method surface the Rust ``_core`` bridge
    calls on its ``runtime_handle`` (see rust/sglang-grpc/src/bridge.rs)."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.tokenizer_manager = engine.tokenizer_manager
        self.server_args = engine.server_args
        # Point-in-time load, refreshed from the scheduler's per-batch ``load``
        # piggyback (see GetLoad). Total token budget is fixed at startup.
        self._running_reqs = 0
        self._waiting_reqs = 0
        self._max_total_tokens = self._compute_max_total_tokens()
        # Disaggregation: prefill workers advertise a bootstrap host/port and mint
        # a per-request room; the decode peer connects to it. Resolved once here.
        self.role = getattr(self.server_args, "disaggregation_mode", "null") or "null"
        self.bootstrap_host = None
        self.bootstrap_port = None
        if self.role == "prefill":
            port = getattr(self.server_args, "disaggregation_bootstrap_port", None)
            if port:
                self.bootstrap_host = _resolve_local_ip()
                self.bootstrap_port = int(port)
        # dp-attention DP-rank assignment fallback (see _drive). The DP
        # controller direct-routes when a rank is set but round-robins (gated on
        # worker status) when it's None, which can stall a disagg-decode receive.
        self._dp_size = getattr(self.server_args, "dp_size", 1) or 1
        self._dp_rr = 0
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
        # Prefill disagg bootstrap warmup. The in-process dynamo.sglang prefill
        # runs warmup_prefill_engine (init_prefill) -- one FAKE_BOOTSTRAP_HOST
        # generate that registers the prefill's bootstrap-server route table and
        # JITs the disagg path. WITHOUT it the decode's /route queries hit
        # KeyError in the prefill bootstrap server (HTTP 500) and every NIXL KV
        # handshake fails ("an unwarmed prefill silently drops production
        # requests"). Replicate it so the sidecar prefill serves disagg decode.
        if self.role == "prefill" and self.bootstrap_port:
            logger.info("openengine prefill disagg warmup starting...")
            wfut = asyncio.run_coroutine_threadsafe(self._warmup_prefill(), self.loop)
            try:
                wfut.result(timeout=900)
                logger.info("openengine prefill disagg warmup complete")
            except Exception as e:  # noqa: BLE001
                logger.warning("openengine prefill warmup failed (continuing): %s", e)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _ensure_handle_loop(self):
        fn = getattr(self.tokenizer_manager, "auto_create_handle_loop", None)
        if callable(fn):
            fn()

    async def _warmup_prefill(self):
        """Drive one FAKE_BOOTSTRAP_HOST request through the prefill disagg path
        (mirrors dynamo.sglang._disagg.warmup_prefill_engine) to register the
        bootstrap route table + JIT the disagg/MoE kernels before real traffic."""
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        gen = await self.engine.async_generate(
            input_ids=[0, 1, 2, 3],
            sampling_params={"temperature": 0.0, "max_new_tokens": 8, "ignore_eos": True},
            stream=True,
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            bootstrap_port=self.bootstrap_port,
            bootstrap_room=999999,
        )
        async for _ in gen:
            pass

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
            # Reconcile fork-newer field names with the running sglang and drop
            # any kwargs its GenerateReqInput doesn't accept (version skew).
            accepted = _genreq_accepted_params()
            if accepted:
                if "routed_dp_rank" in kwargs and "routed_dp_rank" not in accepted:
                    dp_rank = kwargs.pop("routed_dp_rank")
                    if "data_parallel_rank" in accepted:
                        kwargs["data_parallel_rank"] = dp_rank
                kwargs = {k: v for k, v in kwargs.items() if k in accepted}
            # NOTE: do NOT force data_parallel_rank. The in-process dynamo.sglang
            # decode leaves it None and lets the DP controller round-robin assign
            # the rank, which correctly pairs the per-DP-rank NIXL KV receiver with
            # the prefill. Worker-forced direct-routing broke that pairing (non-zero
            # decode ranks failed the NIXL handshake). The cold first decode is slow
            # (DeepEP/wide-EP JIT, ~10-13 min) but completes; don't mistake it for a
            # stall.
            logger.info(
                "openengine _drive: role=%s bootstrap_room=%s bootstrap_host=%s "
                "bootstrap_port=%s data_parallel_rank=%s n_input_ids=%d has_text=%s",
                self.role,
                kwargs.get("bootstrap_room"),
                kwargs.get("bootstrap_host"),
                kwargs.get("bootstrap_port"),
                kwargs.get("data_parallel_rank"),
                len(kwargs.get("input_ids") or []),
                bool(kwargs.get("text")),
            )
            obj = GenerateReqInput(**kwargs)
            gen = self.tokenizer_manager.generate_request(obj, None)
            logger.info("openengine _drive: role=%s submitted, awaiting first chunk", self.role)
            _chunk_count = 0
            # SGLang streams CUMULATIVE output_ids per chunk; the OpenEngine
            # TokenOutput contract (and the Dynamo sidecar) expect only the NEW
            # tokens. Track how many we've forwarded and emit the suffix.
            sent = 0
            async for out in gen:
                _chunk_count += 1
                if _chunk_count == 1:
                    logger.info("openengine _drive: role=%s FIRST CHUNK received", self.role)
                meta = out.get("meta_info", {}) or {}
                # Scheduler piggybacks per-DP-rank load on each chunk; cache the
                # latest so GetLoad can report a real running/waiting count.
                if "num_running_reqs" in meta:
                    self._running_reqs = meta["num_running_reqs"] or 0
                if "num_waiting_reqs" in meta:
                    self._waiting_reqs = meta["num_waiting_reqs"] or 0
                finished = bool(meta.get("finish_reason"))
                full_ids = out.get("output_ids", []) or []
                new_ids = full_ids[sent:]
                # SGLang streams logprobs cumulatively too (output_token_logprobs
                # / output_top_logprobs align 1:1 with output_ids); de-cumulate
                # with the same pre-update `sent` index so each chunk carries only
                # its new tokens' logprobs.
                new_token_lp = (meta.get("output_token_logprobs") or [])[sent:]
                new_top_lp = (meta.get("output_top_logprobs") or [])[sent:]
                sent = len(full_ids)
                # Copy before mutating; the original meta is the engine's.
                meta = dict(meta)
                if finished and disagg is not None:
                    meta["disaggregated_params"] = disagg
                if new_token_lp:
                    meta["oe_token_logprobs"] = new_token_lp
                if new_top_lp:
                    meta["oe_top_logprobs"] = new_top_lp
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

    def _compute_max_total_tokens(self) -> int:
        """Total KV token budget from the scheduler init result (0 if unknown)."""
        sched = getattr(self.engine, "_scheduler_init_result", None)
        infos = getattr(sched, "scheduler_infos", None) if sched is not None else None
        if infos:
            try:
                return int(infos[0].get("max_total_num_tokens", 0) or 0)
            except Exception:  # noqa: BLE001
                return 0
        return 0

    def get_model_info(self) -> str:
        sa = self.server_args
        is_mm = bool(getattr(self.tokenizer_manager, "mm_processor", None))
        page_size = int(getattr(sa, "page_size", 0) or 0)
        # KV capacity for the router: total token budget / page size.
        total_kv_blocks = (self._max_total_tokens // page_size) if page_size else 0
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
                # Response parser names the frontend applies to this model's
                # output (tool-call / reasoning). Configured on the engine via
                # --tool-call-parser / --reasoning-parser; discovered here so the
                # sidecar stays endpoint-only.
                "reasoning_parser": getattr(sa, "reasoning_parser", None) or "",
                "tool_call_parser": getattr(sa, "tool_call_parser", None) or "",
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
        ip = _resolve_local_ip()
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
        # Real point-in-time load. running/waiting come from the scheduler's
        # per-batch ``load`` piggyback cached in _drive; self-correct to idle
        # when the TokenizerManager has no in-flight requests. (used_tokens is
        # not exposed by SGLang on this path, so it stays 0 — full KV-capacity
        # load reporting to the router needs the snapshot-publisher path.)
        try:
            in_flight = len(self.tokenizer_manager.rid_to_state)
        except Exception:  # noqa: BLE001
            in_flight = 0
        if in_flight == 0:
            self._running_reqs = 0
            self._waiting_reqs = 0
        payload = json.dumps(
            {
                "num_running_reqs": self._running_reqs,
                "num_waiting_reqs": self._waiting_reqs,
                "num_used_tokens": 0,
                "max_total_num_tokens": self._max_total_tokens,
            }
        ).encode()
        try:
            callback(payload, True, None)
        except Exception as e:  # noqa: BLE001
            logger.debug("get_load callback failed: %s", e)


def serve_openengine(server_args, openengine_host=None, openengine_port=None):
    """Launch the SGLang engine + the OpenEngine v1 gRPC server, and block.

    `openengine_host`/`openengine_port` may be passed explicitly (the overlay
    deployment runs this module standalone on a stock sglang whose ServerArgs
    has no openengine_* fields); otherwise they fall back to ServerArgs attrs
    (the in-tree fork path, where launch_server sets them)."""
    from sglang.srt.grpc import _core

    host = (
        openengine_host
        or getattr(server_args, "openengine_host", None)
        or server_args.host
        or "127.0.0.1"
    )
    port = (
        openengine_port
        if openengine_port is not None
        else getattr(server_args, "openengine_port", None)
    )

    logger.info("Starting SGLang engine for OpenEngine sidecar...")
    engine = Engine(server_args=server_args)
    handle = RuntimeHandle(engine)

    # Per-stream "no chunk produced" watchdog on the gRPC server. The default
    # (300s) sheds requests that wait too long for their first token; under
    # extreme prefill-queue oversubscription (e.g. conc 4096 disagg) that kills
    # deeply-queued requests the engine would otherwise still serve. Align it
    # with the deployment's SGLang disagg timeouts via this env knob.
    response_timeout_secs = int(
        os.environ.get("SGLANG_OPENENGINE_RESPONSE_TIMEOUT_SECS", "300")
    )

    logger.info(
        "Mounting OpenEngine v1 gRPC server on %s:%d (response_timeout_secs=%d)",
        host,
        port,
        response_timeout_secs,
    )
    server = _core.start_openengine_server(
        host, port, handle, response_timeout_secs=response_timeout_secs
    )

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


def main():
    """Standalone entrypoint for the overlay deployment.

    Invoked as ``python -m sglang.srt.entrypoints.openengine_server <sglang
    flags> --openengine-port P [--openengine-host H]``. The OpenEngine transport
    flags are stripped here (the stock sglang ServerArgs doesn't define them);
    the rest are parsed by sglang's normal CLI path.
    """
    import sys

    from sglang.srt.server_args import prepare_server_args

    argv = sys.argv[1:]
    oe_host = None
    oe_port = None
    rest = []
    it = iter(argv)
    for a in it:
        if a == "--openengine-port":
            oe_port = int(next(it))
        elif a == "--openengine-host":
            oe_host = next(it)
        else:
            rest.append(a)
    if oe_port is None:
        raise SystemExit("openengine_server: --openengine-port is required")
    server_args = prepare_server_args(rest)
    serve_openengine(server_args, openengine_host=oe_host, openengine_port=oe_port)


if __name__ == "__main__":
    main()
