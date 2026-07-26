#!/usr/bin/env python3
"""Initialize OLMo3 context length before Transformers validates YaRN RoPE."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


RELATIVE_PATH = Path("configs/olmo3.py")
MARKER = (
    "        # OLMO3_ROPE_CONFIG_INIT_FIX: RoPE validation runs in "
    "PretrainedConfig.__init__."
)
ASSIGNMENT = "        self.max_position_embeddings = max_position_embeddings\n"
SUPER_CALL = "        super().__init__(\n"


def patch_source(source: str) -> str:
    """Move the context-length assignment ahead of ``super().__init__``."""
    if source.count(SUPER_CALL) != 1:
        raise RuntimeError("expected exactly one OLMo3 PretrainedConfig init call")

    if MARKER not in source:
        if source.count(ASSIGNMENT) != 1:
            raise RuntimeError(
                "expected exactly one OLMo3 max_position_embeddings assignment"
            )
        super_index = source.index(SUPER_CALL)
        assignment_index = source.index(ASSIGNMENT)
        if assignment_index < super_index:
            raise RuntimeError(
                "OLMo3 context length is already initialized before super without "
                "the expected patch marker"
            )
        source = (
            source[:assignment_index]
            + source[assignment_index + len(ASSIGNMENT) :]
        )
        super_index = source.index(SUPER_CALL)
        source = (
            source[:super_index]
            + MARKER
            + "\n"
            + ASSIGNMENT
            + source[super_index:]
        )

    marker_index = source.index(MARKER)
    assignment_index = source.index(ASSIGNMENT, marker_index)
    super_index = source.index(SUPER_CALL)
    if not marker_index < assignment_index < super_index:
        raise RuntimeError(
            "OLMo3 context length must be initialized before PretrainedConfig"
        )
    if source.count(ASSIGNMENT) != 1:
        raise RuntimeError("OLMo3 context-length assignment was duplicated")
    return source


def patch_venv(venv: Path) -> None:
    roots = list(venv.glob("lib/python*/site-packages/sglang/srt"))
    if len(roots) != 1:
        raise RuntimeError(f"expected one sglang/srt under {venv}, found {roots}")
    path = roots[0] / RELATIVE_PATH
    original = path.read_text()
    patched = patch_source(original)
    if patched != original:
        backup = path.with_suffix(path.suffix + ".pre_olmo3_rope_config_init")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(patched)
        print(f"  patched: {path.relative_to(roots[0])}")
    else:
        print(f"  verified: {path.relative_to(roots[0])}")
    for pyc in path.parent.glob("olmo3*.pyc"):
        pyc.unlink()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <venv_path>")
    patch_venv(Path(sys.argv[1]).resolve())
    print("[patch] OLMo3 RoPE config initialization verified")


if __name__ == "__main__":
    main()
