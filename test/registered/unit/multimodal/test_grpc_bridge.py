import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.entrypoints.grpc_bridge import (
    MediaRuntimeHandle,
    _normalize_video_response,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _TaskType:
    def __init__(self, image: bool):
        self._image = image
        self.name = "T2I" if image else "T2V"

    def is_image_gen(self):
        return self._image


class _Response:
    def __init__(self, body, status_code=200):
        self.content = json.dumps(body).encode()
        self.status_code = status_code

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def json(self):
        return json.loads(self.content)


class _Client:
    def __init__(self, image: bool):
        self.image = image
        self.is_closed = False
        self.posts = []
        self.polls = []

    async def post(self, endpoint, json, headers):
        self.posts.append((endpoint, json, headers))
        if self.image:
            return _Response({"data": [{"url": "image.png"}]})
        return _Response({"id": "video-1", "model": "wan", "status": "queued"})

    async def get(self, endpoint):
        self.polls.append(endpoint)
        return _Response(
            {
                "id": "video-1",
                "model": "wan",
                "status": "completed",
                "progress": 100,
                "created_at": 123,
                "url": "https://example.test/video.mp4",
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("image", [True, False])
async def test_media_generate_uses_openai_handlers_for_image_and_video(image):
    warmup = asyncio.Event()
    warmup.set()
    handle = MediaRuntimeHandle.__new__(MediaRuntimeHandle)
    handle.app = SimpleNamespace(state=SimpleNamespace(server_warmup_done=warmup))
    handle.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(task_type=_TaskType(image))
    )
    handle._client = _Client(image)
    calls = []

    await handle._generate(
        "rid",
        json.dumps(
            {
                "prompt": "hello",
                "nvext": {"num_inference_steps": 2, "seed": 7},
            }
        ).encode(),
        {"traceparent": "trace"},
        lambda payload, **kwargs: calls.append((payload, kwargs)),
    )

    assert calls[0][1] == {"finished": True, "status_code": 200}
    assert handle._client.posts[0][0] == (
        "/v1/images/generations" if image else "/v1/videos"
    )
    if not image:
        assert handle._client.polls == ["/v1/videos/video-1"]
        body = json.loads(calls[0][0])
        assert body["status"] == "completed"
        assert body["created"] == 123
        assert body["data"] == [
            {
                "output_format": "mp4",
                "url": "https://example.test/video.mp4",
            }
        ]
    posted = handle._client.posts[0][1]
    assert "nvext" not in posted
    assert posted["num_inference_steps"] == 2
    assert posted["seed"] == 7


def test_video_response_embeds_node_local_output(tmp_path):
    output = tmp_path / "result.webm"
    output.write_bytes(b"video-bytes")

    response = _normalize_video_response(
        {
            "id": "video-2",
            "model": "wan",
            "status": "completed",
            "progress": 100,
            "created_at": 456,
            "file_path": str(output),
        },
        {"model": "wan"},
        64 * 1024 * 1024,
    )

    assert response["created"] == 456
    assert response["data"] == [
        {
            "output_format": "webm",
            "b64_json": base64.b64encode(b"video-bytes").decode("ascii"),
        }
    ]


def test_media_health_waits_for_runtime_warmup():
    warmup = asyncio.Event()
    handle = MediaRuntimeHandle.__new__(MediaRuntimeHandle)
    handle.app = SimpleNamespace(state=SimpleNamespace(server_warmup_done=warmup))
    handle._client = SimpleNamespace(is_closed=False)

    assert handle.health_check() is False
    warmup.set()
    assert handle.health_check() is True
