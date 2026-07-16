import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle


class _SendStatus(Enum):
    Ready = 1
    Pending = 2
    Closed = 3


class _Callback:
    def __init__(self):
        self.calls = []

    def __call__(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        return _SendStatus.Ready


class _Tokenizer:
    _pieces = {1: "alpha", 2: "END", 36: "E", 45: "N", 35: "D", 46: "O"}

    def decode(self, token_ids, **_kwargs):
        return "".join(self._pieces[token_id] for token_id in token_ids)


class _TokenizerManager:
    def __init__(self, chunks, tokenizer=None):
        self.chunks = chunks
        self.tokenizer = tokenizer

    async def generate_request(self, _obj, request=None):
        del request
        for chunk in self.chunks:
            yield chunk


class _Registry:
    def get_all_adapters(self):
        return {
            "adapter": SimpleNamespace(
                lora_name="adapter",
                lora_path="/tmp/adapter",
                lora_id="id",
                pinned=True,
            )
        }


class _ControlTokenizerManager:
    def __init__(self):
        self.calls = []
        self.lora_registry = _Registry()
        self.initial_weights_loaded = False
        self.server_args = SimpleNamespace(weight_version="old")

    async def load_lora_adapter(self, req):
        self.calls.append(("load_lora", req))
        return {"success": True}

    async def unload_lora_adapter(self, req):
        self.calls.append(("unload_lora", req))
        return {"success": True}

    async def release_memory_occupation(self, req):
        self.calls.append(("release_memory", req))

    async def resume_memory_occupation(self, req):
        self.calls.append(("resume_memory", req))

    async def start_profile(self, req):
        self.calls.append(("start_profile", req))

    async def stop_profile(self):
        self.calls.append(("stop_profile", None))

    async def update_weights_from_disk(self, req, request=None):
        self.calls.append(("update_weights_from_disk", req))
        return True, "disk", 3

    async def update_weights_from_tensor(self, req, request=None):
        self.calls.append(("update_weights_from_tensor", req))
        return True, "tensor"

    async def update_weights_from_distributed(self, req, request=None):
        self.calls.append(("update_weights_from_distributed", req))
        return True, "distributed"

    async def update_weights_from_ipc(self, req, request=None):
        self.calls.append(("update_weights_from_ipc", req))
        return True, "ipc"

    def abort_request(self, **kwargs):
        self.calls.append(("abort", kwargs))


@pytest.mark.asyncio
async def test_interleaved_choices_close_only_after_every_terminal():
    chunks = [
        {
            "index": 1,
            "output_ids": [20],
            "meta_info": {"finish_reason": {"type": "stop"}},
        },
        {"index": 0, "output_ids": [10], "meta_info": {"finish_reason": None}},
        {
            "index": 0,
            "output_ids": [10, 11],
            "meta_info": {"finish_reason": {"type": "length"}},
        },
    ]
    handle = RuntimeHandle.__new__(RuntimeHandle)
    handle.tokenizer_manager = _TokenizerManager(chunks)
    callback = _Callback()
    obj = SimpleNamespace(rid="rid", sampling_params={"n": 2})

    await handle._run_generate(obj, callback, True, None)

    assert [call[1]["finished"] for call in callback.calls] == [False, False, True]
    assert [call[0]["index"] for call in callback.calls] == [1, 0, 0]


def test_stop_visibility_is_independent_for_strings_and_tokens():
    handle = RuntimeHandle.__new__(RuntimeHandle)
    handle.tokenizer_manager = _TokenizerManager([], _Tokenizer())
    hidden_string = {
        "text": "answer<stop>",
        "output_ids": [1, 2],
        "meta_info": {"finish_reason": {"type": "stop", "matched": "<stop>"}},
    }
    handle._apply_stop_visibility(
        hidden_string,
        {"strings": [{"value": "<stop>", "include_in_output": False}]},
    )
    assert hidden_string["text"] == "answer"

    visible_token = {
        "text": "answer",
        "output_ids": [1, 2, 99],
        "meta_info": {"finish_reason": {"type": "stop", "matched": 99}},
    }
    handle._apply_stop_visibility(
        visible_token,
        {"tokens": [{"token_id": 99, "include_in_output": True}]},
    )
    assert visible_token["output_ids"] == [1, 2, 99]


def test_stop_visibility_recovers_missing_match_from_terminal_output():
    handle = RuntimeHandle.__new__(RuntimeHandle)
    handle.tokenizer_manager = _TokenizerManager([], _Tokenizer())
    hidden_string = {
        "text": "alphaEND",
        "output_ids": [1, 36, 45, 35, 151645],
        "meta_info": {"finish_reason": {"type": "stop", "matched": 151645}},
    }
    handle._apply_stop_visibility(
        hidden_string,
        {
            "strings": [
                {"value": "END", "include_in_output": False},
                {"value": "D", "include_in_output": True},
            ]
        },
    )
    assert hidden_string["text"] == "alpha"
    assert hidden_string["output_ids"] == [1]
    assert hidden_string["meta_info"]["finish_reason"]["matched"] == "END"


def test_hidden_stop_prefixes_are_held_back_before_terminal():
    handle = RuntimeHandle.__new__(RuntimeHandle)
    handle.tokenizer_manager = _TokenizerManager([], _Tokenizer())
    visibility = {"strings": [{"value": "END", "include_in_output": False}]}
    partial = {
        "text": "alphaEN",
        "output_ids": [1, 36, 45],
        "meta_info": {
            "finish_reason": None,
            "output_token_logprobs": [[-0.1, 1], [-0.2, 36], [-0.3, 45]],
            "output_top_logprobs": [[], [], []],
            "output_token_logprobs_length": 3,
        },
    }
    handle._hold_back_hidden_stop_text(partial, visibility)
    assert partial["text"] == "alpha"
    assert partial["output_ids"] == [1]
    assert partial["meta_info"]["output_token_logprobs"] == [[-0.1, 1]]
    assert partial["meta_info"]["output_top_logprobs"] == [[]]
    assert partial["meta_info"]["output_token_logprobs_length"] == 1

    diverged = {
        "text": "alphaENO",
        "output_ids": [1, 36, 45, 46],
        "meta_info": {"finish_reason": None},
    }
    handle._hold_back_hidden_stop_text(diverged, visibility)
    assert diverged["text"] == "alphaENO"


@dataclass
class _ServerArgs:
    model_path: str = "model"

    def describe_kv_events_publisher(self):
        return {"endpoint_port_base": 5557, "dp_size": 2}


def test_server_info_includes_structured_kv_event_descriptor():
    handle = RuntimeHandle.__new__(RuntimeHandle)
    handle.server_args = _ServerArgs()
    handle.scheduler_info = {}

    assert '"kv_events"' in handle.get_server_info()


@pytest.mark.asyncio
async def test_all_typed_controls_map_to_tokenizer_manager_models():
    manager = _ControlTokenizerManager()
    handle = RuntimeHandle.__new__(RuntimeHandle)
    handle.tokenizer_manager = manager
    handle._event_loop = asyncio.get_running_loop()

    operations = [
        ("load_lora", {"name": "adapter", "path": "/tmp/adapter"}),
        ("unload_lora", {"name": "adapter"}),
        ("list_loras", {}),
        ("release_memory", {"tags": ["weights"]}),
        ("resume_memory", {"tags": ["weights"]}),
        ("start_profile", {"activities": ["CPU"], "num_steps": 1}),
        ("stop_profile", {}),
        ("update_weights_from_disk", {"model_path": "/tmp/model"}),
        (
            "update_weights_from_tensor",
            {"serialized_named_tensors": [[1, 2, 3]]},
        ),
        (
            "update_weights_from_distributed",
            {
                "names": ["weight"],
                "dtypes": ["float16"],
                "shapes": [[1]],
            },
        ),
        ("update_weights_from_ipc", {"zmq_handles": {"gpu": "ipc://w"}}),
        (
            "update_weight_version",
            {"new_version": "v2", "abort_all_requests": True},
        ),
    ]

    responses = {}
    for method, payload in operations:
        callback = _Callback()
        handle.submit_control(method, json.dumps(payload).encode(), callback)
        for _ in range(100):
            if callback.calls:
                break
            await asyncio.sleep(0)
        assert callback.calls, method
        body, kwargs = callback.calls[-1]
        assert kwargs.get("error") is None, (method, kwargs)
        responses[method] = json.loads(body)

    assert responses["list_loras"]["adapters"][0]["name"] == "adapter"
    assert responses["update_weights_from_disk"]["num_paused_requests"] == 3
    assert manager.initial_weights_loaded is True
    assert manager.server_args.weight_version == "v2"
    assert ("abort", {"abort_all": True}) in manager.calls
