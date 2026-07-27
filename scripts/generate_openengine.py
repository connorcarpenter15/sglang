"""Generate SGLang's Python gRPC bindings from a pinned OpenEngine schema."""

import argparse
import importlib.util
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _local_proto_root(source_root: Path, expected: str) -> Path:
    source_root = source_root.resolve()
    actual = _run("git", "rev-parse", "HEAD", cwd=source_root)
    if actual != expected:
        raise RuntimeError(
            f"OpenEngine source is at {actual}, but SGLang pins {expected}"
        )
    dirty = _run("git", "status", "--porcelain", "--", "proto", cwd=source_root)
    if dirty:
        raise RuntimeError("OpenEngine proto sources have uncommitted changes")
    proto_root = source_root / "proto"
    if not (proto_root / "openengine" / "v1" / "openengine.proto").is_file():
        raise RuntimeError(f"OpenEngine proto source is missing under {proto_root}")
    return proto_root


def _export_bsr(module: str, output: Path) -> Path:
    if not re.fullmatch(r"buf\.build/[^/:]+/[^/:]+:[A-Za-z0-9_-]{16,}", module):
        raise RuntimeError(
            "--buf-module must include an immutable BSR module commit, for "
            "example buf.build/openengine/openengine:<commit>"
        )
    if shutil.which("buf") is None:
        raise RuntimeError("buf is required to export an OpenEngine BSR module")
    subprocess.run(["buf", "export", module, "--output", str(output)], check=True)
    if not (output / "openengine" / "v1" / "openengine.proto").is_file():
        raise RuntimeError(f"BSR export {module!r} did not contain OpenEngine v1")
    return output


def _grpc_tools_include() -> Path:
    spec = importlib.util.find_spec("grpc_tools")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(
            "grpcio-tools is required to generate OpenEngine bindings; install "
            "the SGLang openengine extra or add grpcio-tools to the build environment"
        )
    return Path(next(iter(spec.submodule_search_locations))) / "_proto"


def _generate(proto_root: Path, output: Path, schema_release: str) -> None:
    proto_files = sorted(
        str(path.relative_to(proto_root))
        for path in (proto_root / "openengine" / "v1").glob("*.proto")
    )
    if not proto_files:
        raise RuntimeError(f"No OpenEngine proto files found under {proto_root}")

    package = output / "openengine"
    versioned = package / "v1"
    versioned.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").touch()
    (versioned / "__init__.py").touch()
    (package / "_schema_identity.py").write_text(
        f'"""Generated OpenEngine schema identity."""\n\n'
        f"SCHEMA_RELEASE = {schema_release!r}\n",
        encoding="utf-8",
    )
    for pattern in ("*_pb2.py", "*_pb2_grpc.py", "*_pb2.pyi"):
        for generated in versioned.glob(pattern):
            generated.unlink()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{proto_root}",
            f"-I{_grpc_tools_include()}",
            f"--python_out={output}",
            f"--grpc_python_out={output}",
            *proto_files,
        ],
        cwd=proto_root,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Pinned local OpenEngine checkout (default: OPENENGINE_SOURCE_ROOT "
            "or ../openengine-trtllm)"
        ),
    )
    parser.add_argument(
        "--buf-module",
        help=(
            "Immutable BSR module input; may also be supplied through "
            "OPENENGINE_BSR_MODULE"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Python package root (default: <SGLang checkout>/python)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    expected = (root / "OPENENGINE_COMMIT").read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise RuntimeError("OPENENGINE_COMMIT must contain one full lowercase Git SHA")
    packaged = runpy.run_path(
        str(
            root
            / "python"
            / "sglang"
            / "srt"
            / "entrypoints"
            / "openengine"
            / "_schema_pin.py"
        )
    )["OPENENGINE_COMMIT"]
    if packaged != expected:
        raise RuntimeError(
            f"Packaged OpenEngine pin is {packaged}, but OPENENGINE_COMMIT contains "
            f"{expected}"
        )

    module = args.buf_module or os.environ.get("OPENENGINE_BSR_MODULE")
    configured_source = args.source_root or (
        Path(value) if (value := os.environ.get("OPENENGINE_SOURCE_ROOT")) else None
    )
    if module and configured_source:
        raise RuntimeError("Use exactly one of --source-root and --buf-module")

    output = (args.output or root / "python").resolve()
    with tempfile.TemporaryDirectory(prefix="sglang-openengine-") as temporary:
        if module:
            proto_root = _export_bsr(module, Path(temporary))
            identity = module.rsplit(":", 1)[1]
        else:
            source_root = configured_source or root.parent / "openengine-trtllm"
            proto_root = _local_proto_root(source_root, expected)
            identity = expected
        _generate(proto_root, output, identity)

    print(f"Generated OpenEngine Python bindings from {identity}")


if __name__ == "__main__":
    main()
