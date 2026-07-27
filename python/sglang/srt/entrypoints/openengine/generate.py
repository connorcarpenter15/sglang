"""Generate Python gRPC bindings from SGLang's pinned OpenEngine schema."""

import argparse
import importlib
import importlib.util
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_BSR_MODULE_PATTERN = re.compile(r"buf\.build/[^/:]+/[^/:]+:([0-9a-f]{32})")


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[5]
    return candidate if (candidate / "OPENENGINE_COMMIT").is_file() else None


def _schema_pin() -> str:
    values = runpy.run_path(str(Path(__file__).with_name("_schema_pin.py")))
    expected = values["OPENENGINE_COMMIT"]
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise RuntimeError("OPENENGINE_COMMIT must be one full lowercase Git SHA")
    root = _repo_root()
    if root is not None:
        root_pin = (root / "OPENENGINE_COMMIT").read_text(encoding="utf-8").strip()
        if root_pin != expected:
            raise RuntimeError(
                f"Packaged OpenEngine pin is {expected}, but OPENENGINE_COMMIT "
                f"contains {root_pin}"
            )
    return expected


def _default_source_root() -> Path | None:
    root = _repo_root()
    if root is None:
        return None
    candidate = root.parent / "openengine-trtllm"
    return candidate if (candidate / ".git").exists() else None


def _default_output() -> Path:
    root = _repo_root()
    if root is not None:
        return root / "python"
    return Path(__file__).resolve().parents[4]


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


def _export_bsr(module: str, output: Path) -> tuple[Path, str]:
    match = _BSR_MODULE_PATTERN.fullmatch(module)
    if match is None:
        raise RuntimeError(
            "OPENENGINE_BSR_MODULE must end in an immutable 32-character "
            "lowercase hexadecimal BSR commit"
        )
    if shutil.which("buf") is None:
        raise RuntimeError("buf is required to export an OpenEngine BSR module")
    subprocess.run(["buf", "export", module, "--output", str(output)], check=True)
    if not (output / "openengine" / "v1" / "openengine.proto").is_file():
        raise RuntimeError(f"BSR export {module!r} did not contain OpenEngine v1")
    return output, match.group(1)


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


def generation_input_available() -> bool:
    return bool(
        os.environ.get("OPENENGINE_SOURCE_ROOT")
        or os.environ.get("OPENENGINE_BSR_MODULE")
        or _default_source_root()
    )


def generate_openengine(
    *,
    source_root: Path | None = None,
    buf_module: str | None = None,
    output: Path | None = None,
) -> str:
    """Generate bindings and return their immutable schema-release identity."""
    expected = _schema_pin()
    configured_source = source_root or (
        Path(value) if (value := os.environ.get("OPENENGINE_SOURCE_ROOT")) else None
    )
    module = buf_module or os.environ.get("OPENENGINE_BSR_MODULE")
    if module and configured_source:
        raise RuntimeError("Configure exactly one OpenEngine schema source")
    if not module and configured_source is None:
        configured_source = _default_source_root()
    if not module and configured_source is None:
        raise RuntimeError(
            "OpenEngine bindings are absent; set OPENENGINE_SOURCE_ROOT to the "
            "pinned local checkout or OPENENGINE_BSR_MODULE to an immutable module"
        )

    target = (output or _default_output()).resolve()
    with tempfile.TemporaryDirectory(prefix="sglang-openengine-") as temporary:
        if module:
            proto_root, identity = _export_bsr(module, Path(temporary))
        else:
            proto_root = _local_proto_root(configured_source, expected)
            identity = expected
        _generate(proto_root, target, identity)
    importlib.invalidate_caches()
    return identity


def ensure_openengine_bindings() -> None:
    """Generate missing or explicitly mismatched bindings before server imports."""
    source_configured = bool(os.environ.get("OPENENGINE_SOURCE_ROOT"))
    module = os.environ.get("OPENENGINE_BSR_MODULE")
    local_source = _default_source_root()
    desired = None
    if module:
        match = _BSR_MODULE_PATTERN.fullmatch(module)
        if match is None:
            raise RuntimeError(
                "OPENENGINE_BSR_MODULE must end in an immutable 32-character "
                "lowercase hexadecimal BSR commit"
            )
        desired = match.group(1)
    elif source_configured or local_source is not None:
        desired = _schema_pin()

    try:
        identity = importlib.import_module("openengine._schema_identity").SCHEMA_RELEASE
        importlib.import_module("openengine.v1.openengine_pb2_grpc")
        if desired is None or identity == desired:
            return
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass

    generate_openengine()
    for name in tuple(sys.modules):
        if name == "openengine._schema_identity" or name.startswith("openengine.v1."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    importlib.import_module("openengine._schema_identity")
    importlib.import_module("openengine.v1.openengine_pb2_grpc")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--buf-module")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    identity = generate_openengine(
        source_root=args.source_root,
        buf_module=args.buf_module,
        output=args.output,
    )
    print(f"Generated OpenEngine Python bindings from {identity}")


if __name__ == "__main__":
    main()
