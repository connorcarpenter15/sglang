"""Verify a tracked-files-only SGLang checkout builds OpenEngine bindings."""

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Pinned local OpenEngine checkout",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="sglang-clean-build-") as temporary:
        temporary_root = Path(temporary)
        archive = temporary_root / "sglang.tar"
        checkout = temporary_root / "checkout"
        build_lib = temporary_root / "build"
        with archive.open("wb") as output:
            subprocess.run(
                ["git", "archive", "--format=tar", "HEAD"],
                cwd=root,
                stdout=output,
                check=True,
            )
        checkout.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(checkout)

        generated = checkout / "python" / "openengine" / "v1"
        if any(generated.glob("*_pb2.py")):
            raise RuntimeError(
                "Tracked checkout unexpectedly contains generated bindings"
            )

        environment = os.environ.copy()
        environment.update(
            {
                "OPENENGINE_SOURCE_ROOT": str(args.source_root.resolve()),
                "SGLANG_BUILD_RUST_EXTS": "none",
            }
        )
        subprocess.run(
            [
                sys.executable,
                "setup.py",
                "build_py",
                "--build-lib",
                str(build_lib),
            ],
            cwd=checkout / "python",
            env=environment,
            check=True,
        )

        expected = [
            build_lib / "openengine" / "_schema_identity.py",
            build_lib / "openengine" / "v1" / "generation_pb2.py",
            build_lib / "openengine" / "v1" / "openengine_pb2_grpc.py",
        ]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise RuntimeError(f"Clean package build omitted bindings: {missing}")

        environment["PYTHONPATH"] = str(build_lib)
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from openengine._schema_identity import SCHEMA_RELEASE; "
                    "from openengine.v1 import openengine_pb2_grpc; "
                    "assert SCHEMA_RELEASE"
                ),
            ],
            cwd=temporary_root,
            env=environment,
            check=True,
        )

    print("Clean SGLang package build generated OpenEngine bindings")


if __name__ == "__main__":
    main()
