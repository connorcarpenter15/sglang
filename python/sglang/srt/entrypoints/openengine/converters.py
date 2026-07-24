"""Wire conversion between OpenEngine and SGLang's native request objects."""

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

from google.protobuf.json_format import MessageToDict
from openengine.v1 import generation_pb2, server_pb2

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import Modality

HANDOFF_PROFILE = "sglang.bootstrap.v1"
MAX_BOOTSTRAP_ROOM = (1 << 63) - 1

_MODALITIES = {
    generation_pb2.MODALITY_IMAGE: "image",
    generation_pb2.MODALITY_VIDEO: "video",
    generation_pb2.MODALITY_AUDIO: "audio",
}

_MEDIA_OPTION_FIELDS = {
    "image": {
        "max_dynamic_patch",
        "min_dynamic_patch",
        "image_max_dynamic_patch",
        "images_config",
    },
    "video": {"video_max_dynamic_patch", "use_audio_in_video"},
    "audio": set(),
}


@dataclass(frozen=True)
class ConvertedGenerate:
    request: GenerateReqInput
    session_id: str
    is_prefill: bool


def _optional(message: Any, name: str) -> Any | None:
    return getattr(message, name) if message.HasField(name) else None


def _sampling(request: generation_pb2.GenerateRequest, *, prefill: bool) -> dict:
    sampling = request.sampling
    stopping = request.stopping
    params: dict[str, Any] = {}
    for source, target in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("min_p", "min_p"),
        ("frequency_penalty", "frequency_penalty"),
        ("presence_penalty", "presence_penalty"),
        ("repetition_penalty", "repetition_penalty"),
        ("seed", "sampling_seed"),
        ("num_sequences", "n"),
    ):
        value = _optional(sampling, source)
        if value is not None:
            params[target] = (
                int(value) if source in ("seed", "num_sequences") else value
            )

    max_tokens = _optional(stopping, "max_tokens")
    params["max_new_tokens"] = (
        min(max_tokens or 128, 1) if prefill else (max_tokens or 128)
    )
    min_tokens = _optional(stopping, "min_tokens")
    if min_tokens is not None:
        params["min_new_tokens"] = min(min_tokens, params["max_new_tokens"])
    ignore_eos = _optional(stopping, "ignore_eos")
    if ignore_eos is not None:
        params["ignore_eos"] = ignore_eos
    include_stop = _optional(stopping, "include_stop_in_output")
    if include_stop is not None:
        params["no_stop_trim"] = include_stop

    stop_texts: list[str] = []
    stop_ids: set[int] = set()
    for condition in stopping.conditions:
        kind = condition.WhichOneof("condition")
        if kind == "stop_text":
            stop_texts.append(condition.stop_text)
        elif kind == "stop_token_id":
            stop_ids.add(condition.stop_token_id)
    if stop_texts:
        params["stop"] = stop_texts
    if stop_ids:
        params["stop_token_ids"] = stop_ids

    guide = request.guided
    guide_kind = guide.WhichOneof("guide")
    if guide.backend:
        raise ValueError(
            "SGLang selects its grammar backend at server startup; per-request "
            "guided-decoding backend overrides are unsupported"
        )
    if guide_kind == "json_schema":
        params["json_schema"] = guide.json_schema
    elif guide_kind == "regex":
        params["regex"] = guide.regex
    elif guide_kind == "ebnf_grammar":
        params["ebnf"] = guide.ebnf_grammar
    elif guide_kind == "structural_tag":
        params["structural_tag"] = guide.structural_tag
    elif guide_kind == "choice":
        if not guide.choice.choices:
            raise ValueError("guided choice must contain at least one value")
        params["regex"] = (
            "^(?:" + "|".join(re.escape(value) for value in guide.choice.choices) + ")$"
        )
    elif guide_kind == "json_object":
        params["json_schema"] = json.dumps({"type": "object"}, separators=(",", ":"))
    return params


def _candidate_selection(response, field: str) -> tuple[int, list[int] | None]:
    selection = getattr(response, field)
    kind = selection.WhichOneof("selection")
    if kind == "top_n":
        return selection.top_n, None
    if kind == "token_ids":
        return 0, list(selection.token_ids.ids)
    if kind == "all":
        raise ValueError("SGLang does not support returning every logprob candidate")
    return 0, None


def _response_options(request: generation_pb2.GenerateRequest) -> dict[str, Any]:
    response = request.response
    prompt_requested = bool(_optional(response, "return_prompt_logprobs"))
    output_requested = bool(_optional(response, "return_output_logprobs"))
    prompt_top_n, prompt_ids = _candidate_selection(response, "prompt_candidates")
    output_top_n, output_ids = _candidate_selection(response, "output_candidates")
    token_ids = sorted(set((prompt_ids or []) + (output_ids or []))) or None
    requested = prompt_requested or output_requested or bool(token_ids)
    result: dict[str, Any] = {
        "return_logprob": requested,
        "return_text_in_logprobs": requested,
    }
    if requested:
        result["top_logprobs_num"] = max(prompt_top_n, output_top_n)
        result["token_ids_logprob"] = token_ids
        start = _optional(response, "prompt_logprob_start")
        result["logprob_start_len"] = (
            start if prompt_requested and start is not None else -1
        )
    return result


def _media_source(item) -> str:
    source = item.WhichOneof("source")
    if source in ("url", "data_uri"):
        return getattr(item, source)
    if source == "raw_bytes":
        if not item.mime_type:
            raise ValueError("raw media bytes require mime_type")
        encoded = base64.b64encode(item.raw_bytes).decode("ascii")
        return f"data:{item.mime_type};base64,{encoded}"
    raise ValueError("Each media item must carry exactly one source")


def _media(request: generation_pb2.GenerateRequest) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {name: [] for name in _MODALITIES.values()}
    image_hashes: list[str] = []
    all_image_hashes = True
    modalities: list[str] = []
    for item in request.media:
        modality = _MODALITIES.get(item.modality)
        if modality is None:
            raise ValueError(f"Unsupported media modality {item.modality}")
        grouped[modality].append(_media_source(item))
        modalities.append(modality)
        if modality == "image":
            if not item.uuid:
                all_image_hashes = False
            else:
                try:
                    bytes.fromhex(item.uuid)
                except ValueError:
                    all_image_hashes = False
                image_hashes.append(item.uuid)

    result: dict[str, Any] = {}
    for modality, values in grouped.items():
        if values:
            result[f"{modality}_data"] = values
    if modalities:
        result["modalities"] = modalities
    if (
        grouped["image"]
        and all_image_hashes
        and len(image_hashes) == len(grouped["image"])
    ):
        result["mm_hashes"] = image_hashes

    options = (
        MessageToDict(request.media_options, preserving_proto_field_name=True)
        if request.media_options.fields
        else {}
    )
    unknown_modalities = set(options).difference(_MEDIA_OPTION_FIELDS)
    if unknown_modalities:
        raise ValueError(
            f"Unknown media option modalities: {sorted(unknown_modalities)}"
        )
    for modality, values in options.items():
        if not isinstance(values, dict):
            raise ValueError(f"media_options.{modality} must be an object")
        unknown = set(values).difference(_MEDIA_OPTION_FIELDS[modality])
        if unknown:
            raise ValueError(
                f"Unsupported SGLang media options for {modality}: {sorted(unknown)}"
            )
        for name, value in values.items():
            if name in result and result[name] != value:
                raise ValueError(f"Conflicting media option {name!r}")
            result[name] = value
    return result


_NUMBERED_MEDIA_PLACEHOLDER = re.compile(r"<\|(image|video|audio)_([^|]+)\|>")


def _encode_placeholder(tokenizer: Any, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return []
    try:
        return [int(value) for value in encode(text, add_special_tokens=False)]
    except (TypeError, ValueError):
        return []


def _placeholder(
    multimodal_processor: Any, modality: str
) -> tuple[str, tuple[int, ...]] | None:
    marker_for = getattr(multimodal_processor, "get_unexpanded_mm_token", None)
    if marker_for is None:
        return None
    text = marker_for(Modality.from_str(modality))
    if not isinstance(text, str) or not text:
        return None

    tokenizer = getattr(multimodal_processor, "_tokenizer", None)
    token_ids = _encode_placeholder(tokenizer, text)
    if not token_ids:
        return None
    return text, tuple(token_ids)


def _expected_numbered_placeholders(modalities: list[str]) -> list[str]:
    indexes = {modality: 0 for modality in _MODALITIES.values()}
    expected = []
    for modality in modalities:
        indexes[modality] += 1
        expected.append(f"<|{modality}_{indexes[modality]}|>")
    return expected


def _native_text_occurrences(
    text: str, multimodal_processor: Any
) -> list[tuple[int, int, str]]:
    tokens = getattr(multimodal_processor, "mm_tokens", None)
    get_regex = getattr(tokens, "get_combined_regex", None)
    get_modality = getattr(tokens, "get_modality_of_token", None)
    if get_regex is None or get_modality is None:
        raise ValueError("SGLang multimodal processor cannot validate native markers")

    occurrences = []
    for match in get_regex().finditer(text):
        modality = get_modality(match.group(0))
        if not isinstance(modality, Modality):
            raise ValueError("SGLang native media placeholder has unknown modality")
        occurrences.append((match.start(), match.end(), modality.name.lower()))
    return occurrences


def _replacement_placeholders(
    multimodal_processor: Any, modalities: list[str]
) -> dict[str, tuple[str, tuple[int, ...]]]:
    placeholders = {
        modality: _placeholder(multimodal_processor, modality)
        for modality in set(modalities)
    }
    missing_specs = [
        modality
        for modality, placeholder in placeholders.items()
        if placeholder is None
    ]
    if missing_specs:
        raise ValueError(
            "SGLang multimodal processor does not define replacement placeholders for "
            f"{sorted(missing_specs)}"
        )
    return placeholders


def _normalize_numbered_text(
    text: str,
    modalities: list[str],
    multimodal_processor: Any,
) -> str:
    numbered = list(_NUMBERED_MEDIA_PLACEHOLDER.finditer(text))
    native = _native_text_occurrences(text, multimodal_processor)
    if numbered:
        if native:
            raise ValueError(
                "Prompt mixes canonical numbered and SGLang native media placeholders"
            )
        expected = _expected_numbered_placeholders(modalities)
        actual = [match.group(0) for match in numbered]
        if actual != expected:
            raise ValueError(
                "Canonical numbered media placeholders must match request media "
                f"in order: expected {expected}, got {actual}"
            )
        placeholders = _replacement_placeholders(multimodal_processor, modalities)
        parts = []
        offset = 0
        for match, modality in zip(numbered, modalities):
            parts.extend((text[offset : match.start()], placeholders[modality][0]))
            offset = match.end()
        parts.append(text[offset:])
        return "".join(parts)

    native_modalities = [modality for _, _, modality in native]
    if native_modalities:
        if native_modalities != modalities:
            raise ValueError(
                "SGLang native media placeholders must match request media in order"
            )
        return text
    raise ValueError(
        "Multimodal prompt requires exactly one placeholder per media item"
    )


def _decode_token_ids(tokenizer: Any, input_ids: list[int]) -> str | None:
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return None
    try:
        return str(decode(input_ids, skip_special_tokens=False))
    except (TypeError, ValueError):
        return None


def _normalize_numbered_token_ids(
    input_ids: list[int],
    modalities: list[str],
    multimodal_processor: Any,
) -> list[int]:
    tokenizer = getattr(multimodal_processor, "_tokenizer", None)
    decoded = _decode_token_ids(tokenizer, input_ids)
    if decoded is None:
        raise ValueError("SGLang tokenizer could not decode multimodal token input")
    numbered = list(_NUMBERED_MEDIA_PLACEHOLDER.finditer(decoded))
    native = _native_text_occurrences(decoded, multimodal_processor)
    if numbered:
        if native:
            raise ValueError(
                "Prompt mixes canonical numbered and SGLang native media placeholders"
            )
        expected = _expected_numbered_placeholders(modalities)
        actual = [match.group(0) for match in numbered]
        if actual != expected:
            raise ValueError(
                "Canonical numbered media placeholders must match request media "
                f"in order: expected {expected}, got {actual}"
            )
        placeholders = _replacement_placeholders(multimodal_processor, modalities)
        output = []
        offset = 0
        for canonical, modality in zip(expected, modalities):
            pattern = _encode_placeholder(tokenizer, canonical)
            match_offset = next(
                (
                    index
                    for index in range(offset, len(input_ids) - len(pattern) + 1)
                    if input_ids[index : index + len(pattern)] == pattern
                ),
                None,
            )
            if not pattern or match_offset is None:
                raise ValueError(
                    f"Could not normalize tokenized media placeholder {canonical}"
                )
            output.extend(input_ids[offset:match_offset])
            output.extend(placeholders[modality][1])
            offset = match_offset + len(pattern)
        output.extend(input_ids[offset:])
        return output

    native_modalities = [modality for _, _, modality in native]
    if native_modalities:
        if native_modalities != modalities:
            raise ValueError(
                "SGLang native media token placeholders must match request media in order"
            )
        return input_ids
    raise ValueError(
        "Multimodal token input requires exactly one placeholder per media item"
    )


def _normalize_media_placeholders(
    request: generation_pb2.GenerateRequest,
    kwargs: dict[str, Any],
    multimodal_processor: Any,
) -> None:
    if not request.media:
        return

    modalities = [_MODALITIES[item.modality] for item in request.media]

    if "text" in kwargs:
        kwargs["text"] = _normalize_numbered_text(
            kwargs["text"], modalities, multimodal_processor
        )
        return

    kwargs["input_ids"] = _normalize_numbered_token_ids(
        kwargs["input_ids"],
        modalities,
        multimodal_processor,
    )


def _bootstrap(request: generation_pb2.GenerateRequest, role: int) -> tuple[dict, str]:
    if role == server_pb2.ENGINE_ROLE_AGGREGATED:
        if request.kv.HasField("session"):
            raise ValueError("Aggregated SGLang requests must not carry a KV session")
        return {}, ""
    if role not in (server_pb2.ENGINE_ROLE_PREFILL, server_pb2.ENGINE_ROLE_DECODE):
        raise ValueError("Unsupported SGLang OpenEngine role")
    if not request.kv.HasField("session"):
        raise ValueError("Disaggregated SGLang requests require kv.session")
    session = request.kv.session
    if session.handoff_profile != HANDOFF_PROFILE:
        raise ValueError(
            f"SGLang requires handoff_profile {HANDOFF_PROFILE!r}, got "
            f"{session.handoff_profile!r}"
        )
    if not session.HasField("bootstrap"):
        raise ValueError("SGLang disaggregation requires kv.session.bootstrap")
    bootstrap = session.bootstrap
    if not bootstrap.endpoint.host or bootstrap.endpoint.port == 0:
        raise ValueError("SGLang bootstrap requires a host and nonzero port")
    if bootstrap.room_id > MAX_BOOTSTRAP_ROOM:
        raise ValueError(f"SGLang bootstrap room_id must be <= {MAX_BOOTSTRAP_ROOM}")
    values: dict[str, Any] = {
        "bootstrap_host": bootstrap.endpoint.host,
        "bootstrap_port": bootstrap.endpoint.port,
        "bootstrap_room": bootstrap.room_id,
        "session_id": session.session_id or request.request_id,
    }
    if session.dp_rank:
        values["disagg_prefill_dp_rank"] = session.dp_rank
    return values, session.session_id or request.request_id


def convert_generate(
    request: generation_pb2.GenerateRequest,
    *,
    role: int,
    served_model_name: str,
    model_aliases: set[str],
    metadata: dict[str, str],
    lora_path: str | None = None,
    multimodal_processor: Any = None,
) -> ConvertedGenerate:
    if not request.request_id:
        raise ValueError("request_id must not be empty")
    if request.model and request.model not in model_aliases:
        raise LookupError(
            f"Model {request.model!r} is not served by {served_model_name!r}"
        )
    input_kind = request.WhichOneof("input")
    if input_kind not in ("prompt", "token_ids"):
        raise ValueError("Generate requires prompt or token_ids input")
    if request.kv.HasField("bypass_prefix_cache") and request.kv.bypass_prefix_cache:
        raise ValueError("SGLang does not support per-request prefix-cache bypass")

    bootstrap, session_id = _bootstrap(request, role)
    if (
        role != server_pb2.ENGINE_ROLE_AGGREGATED
        and "openengine-target-dp-rank" in metadata
    ):
        target_rank = int(metadata["openengine-target-dp-rank"])
        if target_rank != request.kv.session.dp_rank:
            raise ValueError(
                "openengine-target-dp-rank does not match kv.session.dp_rank"
            )
    prefill = role == server_pb2.ENGINE_ROLE_PREFILL
    sampling = _sampling(request, prefill=prefill)
    if sampling.get("n", 1) < 1:
        raise ValueError("num_sequences must be greater than zero")
    if role != server_pb2.ENGINE_ROLE_AGGREGATED and sampling.get("n", 1) != 1:
        raise ValueError(
            "SGLang disaggregation supports one output sequence per request"
        )

    kwargs: dict[str, Any] = {
        "rid": request.request_id,
        "stream": True,
        "sampling_params": sampling,
        **bootstrap,
        **_response_options(request),
        **_media(request),
    }
    if input_kind == "prompt":
        kwargs["text"] = request.prompt
    else:
        kwargs["input_ids"] = list(request.token_ids.ids)
    if request.media and multimodal_processor is None:
        raise ValueError("SGLang model does not have a multimodal processor")
    if request.media:
        _normalize_media_placeholders(request, kwargs, multimodal_processor)
    if request.kv.HasField("cache_salt"):
        kwargs["extra_key"] = request.kv.cache_salt
    if lora_path:
        # Despite the historical field name, TokenizerManager resolves this
        # value as an adapter *name* in its LoRA registry.
        kwargs["lora_path"] = request.lora_name
    if "openengine-target-dp-rank" in metadata:
        kwargs["routed_dp_rank"] = int(metadata["openengine-target-dp-rank"])
    if "openengine-priority" in metadata:
        kwargs["priority"] = int(metadata["openengine-priority"])
    if "openengine-routing-key" in metadata:
        kwargs["routing_key"] = metadata["openengine-routing-key"]
    trace = {
        name: metadata[name]
        for name in ("traceparent", "tracestate", "baggage")
        if name in metadata
    }
    if trace:
        kwargs["external_trace_header"] = trace
    return ConvertedGenerate(
        request=GenerateReqInput(**kwargs), session_id=session_id, is_prefill=prefill
    )


__all__ = [
    "ConvertedGenerate",
    "HANDOFF_PROFILE",
    "MAX_BOOTSTRAP_ROOM",
    "convert_generate",
]
