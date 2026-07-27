"""Python bindings generated from SGLang's pinned OpenEngine schema."""

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
if (
    not (_PACKAGE_ROOT / "_schema_identity.py").is_file()
    or not (_PACKAGE_ROOT / "v1" / "openengine_pb2_grpc.py").is_file()
):
    from sglang.srt.entrypoints.openengine.generate import ensure_openengine_bindings

    ensure_openengine_bindings()
    del ensure_openengine_bindings

del Path, _PACKAGE_ROOT
