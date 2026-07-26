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
ROPE_PARAMETERS_CAPTURE = (
    '        checkpoint_rope_parameters = kwargs.get("rope_parameters")\n'
)
ROPE_SCALING_ASSIGNMENT = "        self.rope_scaling = rope_scaling\n"
ROPE_SCALING_PRESERVE = """        # OLMO3_ROPE_PARAMETERS_FIX: preserve Transformers-v5 checkpoint data.
        self.rope_scaling = (
            rope_scaling
            if rope_scaling is not None
            else checkpoint_rope_parameters
        )
"""


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

    if ROPE_SCALING_PRESERVE not in source:
        if source.count(ROPE_PARAMETERS_CAPTURE) == 0:
            if source.count(ASSIGNMENT) != 1:
                raise RuntimeError(
                    "expected one OLMo3 context-length assignment before "
                    "preserving RoPE parameters"
                )
            source = source.replace(
                ASSIGNMENT,
                ASSIGNMENT + ROPE_PARAMETERS_CAPTURE,
                1,
            )
        elif source.count(ROPE_PARAMETERS_CAPTURE) != 1:
            raise RuntimeError("OLMo3 RoPE parameter capture was duplicated")
        if source.count(ROPE_SCALING_ASSIGNMENT) != 1:
            raise RuntimeError("expected one OLMo3 rope_scaling assignment")
        source = source.replace(
            ROPE_SCALING_ASSIGNMENT,
            ROPE_SCALING_PRESERVE,
            1,
        )

    marker_index = source.index(MARKER)
    assignment_index = source.index(ASSIGNMENT, marker_index)
    rope_capture_index = source.index(ROPE_PARAMETERS_CAPTURE, assignment_index)
    super_index = source.index(SUPER_CALL)
    if not marker_index < assignment_index < rope_capture_index < super_index:
        raise RuntimeError(
            "OLMo3 context length and RoPE parameters must be captured before "
            "PretrainedConfig"
        )
    if source.count(ASSIGNMENT) != 1:
        raise RuntimeError("OLMo3 context-length assignment was duplicated")
    if source.count(ROPE_PARAMETERS_CAPTURE) != 1:
        raise RuntimeError("OLMo3 RoPE parameter capture was duplicated")
    if source.count(ROPE_SCALING_PRESERVE) != 1:
        raise RuntimeError("OLMo3 RoPE parameter preservation was not installed")
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
