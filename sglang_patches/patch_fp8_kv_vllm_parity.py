#!/usr/bin/env python3
"""Patch SGLang FA3 FP8-KV query handling to match vLLM's contract.

SGLang normally raw-casts Q to the configured KV dtype and only forwards K/V
descales when checkpoint quantization created them. vLLM always owns explicit
Q/K/V scales, statically quantizes Q, and forwards q_descale/k_descale/v_descale
to FA3. This patch preserves stock behavior unless
``SGLANG_FP8_KV_VLLM_PARITY=1`` is set.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


PARITY_MARKER = "# FP8_KV_VLLM_PARITY: statically quantize Q and forward q_descale."
ENV_NAME = "SGLANG_FP8_KV_VLLM_PARITY"

OS_IMPORT_OLD = "from dataclasses import dataclass\n"
OS_IMPORT_NEW = "from dataclasses import dataclass\nimport os\n"

QUANT_IMPORT_OLD = (
    "from sglang.srt.layers.radix_attention import AttentionType\n"
)
QUANT_IMPORT_NEW = (
    "from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant\n"
    "from sglang.srt.layers.radix_attention import AttentionType\n"
)

DESCALE_DECL_OLD = "        k_descale, v_descale = None, None\n"
DESCALE_DECL_NEW = "        q_descale, k_descale, v_descale = None, None, None\n"

QUERY_CAST_OLD = "            q = q.to(self.kv_cache_dtype)\n"
QUERY_CAST_NEW = f'''            {PARITY_MARKER}
            q_scale = getattr(layer, "q_scale", None)
            if (
                os.environ.get("{ENV_NAME}") == "1"
                and self.fa_impl_ver == 3
                and q_scale is not None
            ):
                q_shape = q.shape
                q, _ = scaled_fp8_quant(
                    q.reshape(-1, q.shape[-1]), q_scale
                )
                q = q.reshape(q_shape)
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                q_descale = q_scale.expand(descale_shape)
            else:
                q = q.to(self.kv_cache_dtype)
'''

K_DESCALE_CALL_RE = re.compile(r"(?m)^(?P<indent>\s*)k_descale=k_descale,\n")


def patch_flashattention_backend_text(text: str) -> str:
    if PARITY_MARKER in text:
        required = (
            "scaled_fp8_quant",
            "q_descale=q_descale",
            ENV_NAME,
        )
        missing = [needle for needle in required if needle not in text]
        if missing:
            raise RuntimeError(
                "FP8-KV parity marker is present but the patch is incomplete: "
                f"{missing}."
            )
        return text

    patched = text
    replacements = (
        (OS_IMPORT_OLD, OS_IMPORT_NEW, 1, "os import"),
        (QUANT_IMPORT_OLD, QUANT_IMPORT_NEW, 1, "FP8 quant import"),
        (DESCALE_DECL_OLD, DESCALE_DECL_NEW, 2, "descale declarations"),
        (QUERY_CAST_OLD, QUERY_CAST_NEW, 2, "FP8 query casts"),
    )
    for old, new, expected, description in replacements:
        count = patched.count(old)
        if count != expected:
            raise RuntimeError(
                f"Could not locate {expected} {description}; found {count}. "
                "The SGLang FA3 backend source changed."
            )
        patched = patched.replace(old, new)

    patched, count = K_DESCALE_CALL_RE.subn(
        lambda match: (
            f'{match.group("indent")}q_descale=q_descale,\n'
            f'{match.group("indent")}k_descale=k_descale,\n'
        ),
        patched,
    )
    if count < 2:
        raise RuntimeError(
            "Could not locate FA3 attention calls that consume K/V descales; "
            f"found {count}."
        )
    return patched


def patch_venv(venv: Path) -> None:
    roots = list(venv.glob("lib/python*/site-packages/sglang/srt"))
    if len(roots) != 1:
        raise RuntimeError(f"Expected one sglang/srt under {venv}, found {roots}")
    path = roots[0] / "layers/attention/flashattention_backend.py"
    original = path.read_text()
    patched = patch_flashattention_backend_text(original)
    if patched != original:
        backup = path.with_suffix(path.suffix + ".pre_fp8_kv_vllm_parity")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(patched)
        print(f"  patched: {path.relative_to(roots[0])}")
    else:
        print(f"  verified: {path.relative_to(roots[0])}")

    for pyc in (roots[0] / "layers/attention").rglob("*.pyc"):
        pyc.unlink()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <venv_path>")
    patch_venv(Path(sys.argv[1]).resolve())
    print("[patch] SGLang FA3 FP8-KV vLLM parity verified")


if __name__ == "__main__":
    main()
