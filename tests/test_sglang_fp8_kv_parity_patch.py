from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PATCHES = REPO / "sglang_patches"
sys.path.insert(0, str(PATCHES))

from patch_fp8_kv_vllm_parity import (  # noqa: E402
    ENV_NAME,
    PARITY_MARKER,
    patch_flashattention_backend_text,
)


class FP8KVVllmParityPatchTests(unittest.TestCase):
    def test_patch_is_complete_and_idempotent_on_current_sglang(self):
        source = (
            REPO.parent / "sglang/python/sglang/srt/layers/attention/flashattention_backend.py"
        )
        if not source.exists():
            self.skipTest("local SGLang checkout is unavailable")

        original = source.read_text()
        patched = patch_flashattention_backend_text(original)
        ast.parse(patched)
        self.assertEqual(patched.count(PARITY_MARKER), 2)
        self.assertEqual(patched.count("q_descale=q_descale"), 12)
        self.assertIn(ENV_NAME, patched)
        self.assertIn("scaled_fp8_quant", patched)
        self.assertEqual(patch_flashattention_backend_text(patched), patched)

    def test_custom_target_and_draft_install_explicit_unit_scales(self):
        for name in ("olmo2_sink_dflash.py", "dflash_sink.py"):
            source = (PATCHES / name).read_text()
            with self.subTest(name=name):
                self.assertIn("_install_vllm_parity_fp8_scales", source)
                self.assertIn("attention.register_buffer", source)
                self.assertIn('(\"q_scale\", \"k_scale\", \"v_scale\")', source)
                self.assertIn("persistent=False", source)

    def test_apply_script_installs_backend_patch(self):
        source = (PATCHES / "apply_patches.sh").read_text()
        self.assertIn('patch_fp8_kv_vllm_parity.py" "$VENV"', source)


if __name__ == "__main__":
    unittest.main()
