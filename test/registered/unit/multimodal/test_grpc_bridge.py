import asyncio
import json
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.entrypoints.grpc_bridge import MediaRuntimeHandle
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
        return _Response({"id": "video-1", "status": "queued"})

    async def get(self, endpoint):
        self.polls.append(endpoint)
        return _Response(
            {"id": "video-1", "status": "completed", "data": [{"url": "video.mp4"}]}
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
        json.dumps({"prompt": "hello"}).encode(),
        {"traceparent": "trace"},
        lambda payload, **kwargs: calls.append((payload, kwargs)),
    )

    assert calls[0][1] == {"finished": True, "status_code": 200}
    assert handle._client.posts[0][0] == (
        "/v1/images/generations" if image else "/v1/videos"
    )
    if not image:
        assert handle._client.polls == ["/v1/videos/video-1"]
        assert json.loads(calls[0][0])["status"] == "completed"


def test_media_health_waits_for_runtime_warmup():
    warmup = asyncio.Event()
    handle = MediaRuntimeHandle.__new__(MediaRuntimeHandle)
    handle.app = SimpleNamespace(state=SimpleNamespace(server_warmup_done=warmup))
    handle._client = SimpleNamespace(is_closed=False)

    assert handle.health_check() is False
    warmup.set()
    assert handle.health_check() is True
