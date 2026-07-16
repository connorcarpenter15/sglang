# Native gRPC server

SGLang can expose the typed `sglang.runtime.v1.SglangService` alongside its HTTP
API. The native listener is intended for trusted sidecars and currently has no
HTTP API-key middleware, so do not expose it directly to untrusted networks.

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 30000 \
  --grpc-port 50051
```

The standard SRT runtime supports typed generation, batched embeddings,
classification/tokenization helpers, LoRA lifecycle, memory/profiling controls,
weight updates, runtime discovery, and OpenAI-compatible pass-through methods.
The multimodal-generation runtime exposes `MediaGenerate` for image and video
diffusion when launched with the same `--grpc-port` option. LLM diffusion keeps
using the normal streaming `Generate` method.

Only the endpoint leader starts the public listener in a distributed launch;
followers continue to use SGLang's internal transports. Native runtime discovery
reports the worker role, DP topology, capacity, bootstrap data, metrics and
KV-event endpoints, protocol revision, and descriptor SHA-256.

The default encoded message ceiling is 64 MiB. Configure it with
`--grpc-max-message-size`; oversized media or embeddings should use URLs or NIXL
external-buffer descriptors. `--grpc-response-timeout-secs` controls the maximum
wait between response chunks. The legacy `SGLANG_TONIC_PAYLOAD` environment
override remains available to callers that invoke the extension without the
server argument.

Clients must compile against the exact
`proto/sglang/runtime/v1/sglang.proto` descriptor reported by
`GetRuntimeInfo`. A descriptor or protocol-revision mismatch is a hard startup
compatibility error, not a best-effort mode.
