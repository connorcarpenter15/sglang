"""Generate SGLang's Python bindings from its pinned OpenEngine schema."""

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    generate = runpy.run_path(
        str(
            root
            / "python"
            / "sglang"
            / "srt"
            / "entrypoints"
            / "openengine"
            / "generate.py"
        )
    )["main"]
    generate()


if __name__ == "__main__":
    main()
