"""Logical, lazy OpenEngine LoRA lifecycle over SGLang dynamic adapters."""

import asyncio
import json
from collections import Counter
from pathlib import Path

from openengine.v1 import lora_pb2

from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqInput,
)


class LoraRegistry:
    """Bind external identities immediately and load GPU state on first use."""

    def __init__(self, tokenizer_manager) -> None:
        self._tm = tokenizer_manager
        self._condition = asyncio.Condition()
        self._active: dict[str, lora_pb2.LoraAdapter] = {}
        self._identities: dict[str, lora_pb2.LoraAdapter] = {}
        self._names_by_id: dict[int, str] = {}
        self._names_by_path: dict[str, str] = {}
        self._loaded: set[str] = set()
        self._load_tasks: dict[str, asyncio.Task] = {}
        self._unload_tasks: dict[str, asyncio.Task] = {}
        self._inflight: Counter[str] = Counter()

    @staticmethod
    def _validate(adapter: lora_pb2.LoraAdapter) -> lora_pb2.LoraAdapter:
        if not adapter.lora_name:
            raise ValueError("lora_name must not be empty")
        path = Path(adapter.source_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"LoRA source path is not a directory: {path}")
        config_path = path / "adapter_config.json"
        if not config_path.is_file():
            raise ValueError(f"LoRA directory is missing adapter_config.json: {path}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"LoRA adapter_config.json is invalid: {path}") from error
        if not isinstance(config, dict):
            raise ValueError(f"LoRA adapter_config.json must contain an object: {path}")
        if not any(
            (path / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            raise ValueError(f"LoRA directory is missing adapter weights: {path}")
        return lora_pb2.LoraAdapter(
            lora_id=adapter.lora_id,
            lora_name=adapter.lora_name,
            source_path=str(path),
        )

    async def load(
        self, adapter: lora_pb2.LoraAdapter
    ) -> tuple[lora_pb2.LoraAdapter, bool]:
        validated = self._validate(adapter)
        pending_unload = None
        async with self._condition:
            identity = self._identities.get(validated.lora_name)
            if identity is not None:
                if identity != validated:
                    raise ValueError(
                        f"LoRA name {validated.lora_name!r} is bound to different attributes"
                    )
                if validated.lora_name in self._active:
                    return identity, True
                pending_unload = self._unload_tasks.get(validated.lora_name)
            else:
                bound_name = self._names_by_id.get(validated.lora_id)
                if bound_name is not None:
                    raise ValueError(
                        f"LoRA ID {validated.lora_id} is bound to {bound_name!r}"
                    )
                bound_name = self._names_by_path.get(validated.source_path)
                if bound_name is not None:
                    raise ValueError(
                        f"LoRA path {validated.source_path!r} is bound to {bound_name!r}"
                    )
                self._identities[validated.lora_name] = validated
                self._names_by_id[validated.lora_id] = validated.lora_name
                self._names_by_path[validated.source_path] = validated.lora_name
                self._active[validated.lora_name] = validated
                return validated, False

        if pending_unload is not None:
            await pending_unload
        async with self._condition:
            self._active[validated.lora_name] = self._identities[validated.lora_name]
            return self._active[validated.lora_name], False

    async def list(self) -> list[lora_pb2.LoraAdapter]:
        async with self._condition:
            return [self._active[name] for name in sorted(self._active)]

    async def acquire(self, name: str) -> str:
        async with self._condition:
            adapter = self._active.get(name)
            if adapter is None:
                raise KeyError(name)
            self._inflight[name] += 1
            task = self._load_tasks.get(name)
            if name not in self._loaded and task is None:
                task = asyncio.create_task(self._physical_load(adapter))
                self._load_tasks[name] = task
        try:
            if task is not None:
                await task
        except BaseException:
            await self.release(name)
            raise
        return adapter.source_path

    async def release(self, name: str) -> None:
        async with self._condition:
            if self._inflight[name] > 0:
                self._inflight[name] -= 1
                if self._inflight[name] == 0:
                    del self._inflight[name]
            self._condition.notify_all()

    async def unload(self, name: str) -> lora_pb2.LoraAdapter:
        async with self._condition:
            adapter = self._active.pop(name, None)
            if adapter is None:
                raise KeyError(name)
            task = asyncio.create_task(self._physical_unload_when_idle(name))
            self._unload_tasks[name] = task
            task.add_done_callback(self._log_background_failure)
            return adapter

    @staticmethod
    def _log_background_failure(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Lazy LoRA unload failed")

    async def _physical_load(self, adapter: lora_pb2.LoraAdapter) -> None:
        try:
            result = await self._tm.load_lora_adapter(
                LoadLoRAAdapterReqInput(
                    lora_name=adapter.lora_name,
                    lora_path=adapter.source_path,
                ),
                None,
            )
            if not result.success:
                raise RuntimeError(
                    result.error_message
                    or f"SGLang failed to load LoRA {adapter.lora_name!r}"
                )
            async with self._condition:
                self._loaded.add(adapter.lora_name)
        finally:
            async with self._condition:
                self._load_tasks.pop(adapter.lora_name, None)
                self._condition.notify_all()

    async def _physical_unload_when_idle(self, name: str) -> None:
        try:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._inflight[name] == 0 and name not in self._load_tasks
                )
                loaded = name in self._loaded
            if loaded:
                result = await self._tm.unload_lora_adapter(
                    UnloadLoRAAdapterReqInput(lora_name=name), None
                )
                if not result.success:
                    raise RuntimeError(
                        result.error_message or f"SGLang failed to unload LoRA {name!r}"
                    )
                async with self._condition:
                    self._loaded.discard(name)
        finally:
            async with self._condition:
                self._unload_tasks.pop(name, None)
                self._condition.notify_all()

    async def close(self) -> None:
        async with self._condition:
            tasks = tuple(self._load_tasks.values()) + tuple(
                self._unload_tasks.values()
            )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["LoraRegistry"]
