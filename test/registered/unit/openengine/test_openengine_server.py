import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from openengine.v1 import (
    generation_pb2,
    input_pb2,
    kv_pb2,
    lora_pb2,
    model_pb2,
    observability_pb2,
    server_pb2,
)

from sglang.srt.entrypoints.openengine._schema_pin import OPENENGINE_COMMIT
from sglang.srt.entrypoints.openengine.admission import (
    DrainingError,
    ProcessAdmission,
)
from sglang.srt.entrypoints.openengine.converters import (
    HANDOFF_PROFILE,
    MAX_BOOTSTRAP_ROOM,
    convert_generate,
)
from sglang.srt.entrypoints.openengine.lora_registry import LoraRegistry
from sglang.srt.entrypoints.openengine.servicer import OpenEngineServicer
from sglang.srt.entrypoints.openengine.server import resolve_advertised_host


class _Context:
    def __init__(self, metadata=()):
        self._metadata = metadata

    def invocation_metadata(self):
        return self._metadata

    async def abort(self, code, details):
        raise RuntimeError((code, details))


def _request(request_id="request-1"):
    request = generation_pb2.GenerateRequest(
        request_id=request_id,
        model="served",
        prompt="Hello",
    )
    request.stopping.max_tokens = 8
    return request


def test_typed_bootstrap_is_required_and_room_is_signed_i64_safe():
    request = _request()
    request.kv.session.CopyFrom(
        kv_pb2.KvSessionRef(
            session_id="session-1",
            handoff_profile=HANDOFF_PROFILE,
            bootstrap=kv_pb2.KvBootstrap(
                endpoint=kv_pb2.KvEndpoint(host="decode", port=8998, protocol="tcp"),
                room_id=MAX_BOOTSTRAP_ROOM,
            ),
        )
    )
    converted = convert_generate(
        request,
        role=server_pb2.ENGINE_ROLE_PREFILL,
        served_model_name="served",
        model_aliases={"served"},
        metadata={"openengine-target-dp-rank": "0"},
    )
    assert converted.request.bootstrap_host == "decode"
    assert converted.request.bootstrap_port == 8998
    assert converted.request.bootstrap_room == MAX_BOOTSTRAP_ROOM
    assert converted.request.sampling_params["max_new_tokens"] == 1

    request.kv.session.bootstrap.room_id = MAX_BOOTSTRAP_ROOM + 1
    with pytest.raises(ValueError, match="room_id"):
        convert_generate(
            request,
            role=server_pb2.ENGINE_ROLE_DECODE,
            served_model_name="served",
            model_aliases={"served"},
            metadata={},
        )


def test_disaggregation_rejects_parallel_outputs_before_scheduling():
    request = _request()
    request.sampling.num_sequences = 2
    request.kv.session.CopyFrom(
        kv_pb2.KvSessionRef(
            session_id="session-1",
            handoff_profile=HANDOFF_PROFILE,
            bootstrap=kv_pb2.KvBootstrap(
                endpoint=kv_pb2.KvEndpoint(host="prefill", port=8998, protocol="tcp"),
                room_id=1,
            ),
        )
    )
    with pytest.raises(ValueError, match="one output sequence"):
        convert_generate(
            request,
            role=server_pb2.ENGINE_ROLE_PREFILL,
            served_model_name="served",
            model_aliases={"served"},
            metadata={},
        )


@pytest.mark.asyncio
async def test_handoff_evidence_preserves_uint64_room_as_decimal_string(caplog):
    runtime = _Runtime(mode="prefill")
    servicer = OpenEngineServicer(
        runtime,
        ProcessAdmission(),
        advertised_host="127.0.0.1",
        instance_id="instance",
    )
    request = _request()
    request.kv.session.CopyFrom(
        kv_pb2.KvSessionRef(
            session_id="session-1",
            handoff_profile=HANDOFF_PROFILE,
            bootstrap=kv_pb2.KvBootstrap(
                endpoint=kv_pb2.KvEndpoint(host="prefill", port=8998, protocol="tcp"),
                room_id=MAX_BOOTSTRAP_ROOM,
            ),
        )
    )
    with caplog.at_level(logging.INFO):
        responses = [
            response async for response in servicer.Generate(request, _Context())
        ]
    evidence = [
        json.loads(record.message.removeprefix("OpenEngine handoff "))
        for record in caplog.records
        if "OpenEngine handoff" in record.message
    ]
    assert [value["phase"] for value in evidence] == ["admitted", "complete"]
    assert all(value["session_id"] == "session-1" for value in evidence)
    assert all(value["handoff_profile"] == HANDOFF_PROFILE for value in evidence)
    assert all(
        value["bootstrap"]["room_id"] == str(MAX_BOOTSTRAP_ROOM) for value in evidence
    )
    assert [response.WhichOneof("event") for response in responses] == ["prefill_ready"]
    assert runtime.aborts == []
    await servicer.close()


def test_media_order_and_raw_bytes_survive_conversion():
    request = _request()
    request.media.extend(
        [
            input_pb2.MediaItem(
                modality=input_pb2.MODALITY_IMAGE,
                data_uri="data:image/png;base64,AA==",
            ),
            input_pb2.MediaItem(
                modality=input_pb2.MODALITY_VIDEO,
                raw_bytes=b"video",
                mime_type="video/mp4",
            ),
            input_pb2.MediaItem(
                modality=input_pb2.MODALITY_IMAGE,
                url="https://example.test/image.png",
            ),
        ]
    )
    request.media_options.update(
        {
            "image": {"image_max_dynamic_patch": 4},
            "video": {"use_audio_in_video": False},
        }
    )
    converted = convert_generate(
        request,
        role=server_pb2.ENGINE_ROLE_AGGREGATED,
        served_model_name="served",
        model_aliases={"served"},
        metadata={},
    ).request
    assert converted.modalities == ["image", "video", "image"]
    assert converted.image_data == [
        "data:image/png;base64,AA==",
        "https://example.test/image.png",
    ]
    assert converted.video_data == ["data:video/mp4;base64,dmlkZW8="]
    assert converted.image_max_dynamic_patch == 4


@pytest.mark.asyncio
async def test_process_admission_drains_without_dropping_admitted_requests():
    admission = ProcessAdmission()
    await admission.admit("one", "session")
    await admission.start_drain()
    with pytest.raises(DrainingError):
        await admission.admit("two")
    assert not await admission.wait_empty(timeout=0.01)
    await admission.finish("one", "session")
    assert await admission.wait_empty(timeout=0.01)


class _LoraResult:
    def __init__(self, success=True, error_message=""):
        self.success = success
        self.error_message = error_message


class _LoraManager:
    def __init__(self):
        self.loads = []
        self.unloads = []

    async def load_lora_adapter(self, request, _):
        self.loads.append(request.lora_name)
        return _LoraResult()

    async def unload_lora_adapter(self, request, _):
        self.unloads.append(request.lora_name)
        return _LoraResult()


def _adapter(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    return lora_pb2.LoraAdapter(
        lora_id=7, lora_name="adapter", source_path=str(tmp_path)
    )


@pytest.mark.asyncio
async def test_lora_is_lazy_and_unload_preserves_admitted_selection(tmp_path):
    manager = _LoraManager()
    registry = LoraRegistry(manager)
    adapter, already = await registry.load(_adapter(tmp_path))
    assert not already
    assert manager.loads == []

    await registry.acquire(adapter.lora_name)
    assert manager.loads == [adapter.lora_name]
    await registry.unload(adapter.lora_name)
    with pytest.raises(KeyError):
        await registry.acquire(adapter.lora_name)
    assert manager.unloads == []
    await registry.release(adapter.lora_name)
    for _ in range(20):
        if manager.unloads:
            break
        await asyncio.sleep(0)
    assert manager.unloads == [adapter.lora_name]
    await registry.close()


class _Load:
    def to_dict(self):
        return {
            "timestamp": 1.0,
            "dp_rank": 0,
            "num_running_reqs": 0,
            "num_waiting_reqs": 0,
            "num_waiting_uncached_tokens": 0,
            "num_used_tokens": 0,
            "max_total_num_tokens": 1024,
        }


class _TokenizerManager:
    def __init__(self):
        self.model_config = SimpleNamespace(context_len=4096)
        self.mm_processor = None

    async def generate_request(self, obj, request=None):
        yield {
            "text": "Hi",
            "output_ids": [1],
            "meta_info": {"finish_reason": None},
        }
        yield {
            "text": "",
            "output_ids": [],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 2,
                "completion_tokens": 1,
            },
        }

    async def get_loads(self, dp_rank=None):
        return [_Load()]


class _Runtime:
    def __init__(self, *, mode="null"):
        self.tokenizer_manager = _TokenizerManager()
        self.server_args = SimpleNamespace(
            disaggregation_mode=mode,
            host="0.0.0.0",
            openengine_advertise_host=None,
            served_model_name="served",
            model_path="canonical",
            tokenizer_path="canonical",
            tokenizer_mode="auto",
            skip_tokenizer_init=False,
            reasoning_parser=None,
            tool_call_parser=None,
            enable_lora=False,
            dp_size=2,
            tp_size=2,
            pp_size=1,
            page_size=16,
            max_running_requests=32,
            max_prefill_tokens=4096,
            max_loaded_loras=None,
            max_loras_per_batch=1,
            enable_priority_scheduling=True,
            grammar_backend="xgrammar",
            schedule_policy="fcfs",
            disaggregation_transfer_backend="nixl",
            disaggregation_bootstrap_port=8998,
            describe_kv_events_publisher=lambda: {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "",
                "block_size": 16,
                "dp_size": 2,
            },
        )
        self.scheduler_info = {"max_total_num_tokens": 1024}
        self.aborts = []

    def abort(self, **kwargs):
        self.aborts.append(kwargs)

    def health_check(self):
        return True


def test_advertised_host_prefers_injected_pod_ip(monkeypatch):
    runtime = _Runtime(mode="prefill")
    monkeypatch.setenv("POD_IP", "10.42.3.17")
    assert resolve_advertised_host(runtime, "127.0.0.1") == "10.42.3.17"
    monkeypatch.setenv("OPENENGINE_ADVERTISED_HOST", "prefill.example")
    assert resolve_advertised_host(runtime, "0.0.0.0") == "prefill.example"


@pytest.mark.asyncio
async def test_servicer_streams_terminal_usage_and_discovers_per_rank_sources():
    runtime = _Runtime()
    servicer = OpenEngineServicer(
        runtime,
        ProcessAdmission(),
        advertised_host="127.0.0.1",
        instance_id="instance",
    )
    responses = [
        response async for response in servicer.Generate(_request(), _Context())
    ]
    assert [response.WhichOneof("event") for response in responses] == [
        "token",
        "finished",
    ]
    assert responses[-1].usage.total_tokens == 3
    assert runtime.aborts == []

    server_info = await servicer.GetServerInfo(None, _Context())
    assert server_info.schema_revision == 3
    assert server_info.schema_release == OPENENGINE_COMMIT
    model_info = await servicer.GetModelInfo(
        model_pb2.GetModelInfoRequest(model="served"), _Context()
    )
    assert model_info.tokenizer.source == "canonical"
    sources = await servicer.GetKvEventSources(
        kv_pb2.GetKvEventSourcesRequest(), _Context()
    )
    assert [
        (source.data_parallel_rank, source.endpoint_addr.port)
        for source in sources.sources
    ] == [
        (0, 5557),
        (1, 5558),
    ]
    load = await servicer.GetLoad(observability_pb2.GetLoadRequest(), _Context())
    assert load.total_kv_blocks == 64
    await servicer.close()


@pytest.mark.asyncio
async def test_dropped_parallel_stream_aborts_every_engine_request():
    runtime = _Runtime()
    servicer = OpenEngineServicer(
        runtime,
        ProcessAdmission(),
        advertised_host="127.0.0.1",
        instance_id="instance",
    )
    request = _request("parallel")
    request.sampling.num_sequences = 2
    stream = servicer.Generate(request, _Context())
    first = await anext(stream)
    assert first.token.output_index in (0, 1)
    await stream.aclose()

    assert len(runtime.aborts) == 2
    assert all(
        value["rid"].startswith("parallel.openengine.") for value in runtime.aborts
    )
    await servicer.close()
