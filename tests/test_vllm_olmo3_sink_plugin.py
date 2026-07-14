from __future__ import annotations

import sys
import tomllib
import unittest
from importlib import import_module
from pathlib import Path

import torch
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = REPO_ROOT / "vllm_patches"
sys.path.insert(0, str(PATCH_ROOT / "src"))

plugin = import_module("olmo3_sink_vllm")
model_module = import_module("olmo3_sink_vllm.model")
register = plugin.register
Olmo3SinkForCausalLM = model_module.Olmo3SinkForCausalLM
_rope_parameters_for_layer = model_module._rope_parameters_for_layer
_select_sink_shard = model_module._select_sink_shard


class VllmOlmo3SinkPluginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = AutoConfig.from_pretrained(
            REPO_ROOT.parent / "olmo3sink-32b" / "opd-32b-deploy",
            trust_remote_code=True,
        )

    def test_plugin_entry_point_and_registration(self) -> None:
        metadata = tomllib.loads((PATCH_ROOT / "pyproject.toml").read_text())
        entry_points = metadata["project"]["entry-points"]["vllm.general_plugins"]
        self.assertEqual(entry_points["olmo3_sink"], "olmo3_sink_vllm:register")

        from vllm import ModelRegistry

        register()
        self.assertIn("Olmo3SinkForCausalLM", ModelRegistry.get_supported_archs())
        model_class = ModelRegistry._try_load_model_cls("Olmo3SinkForCausalLM")
        self.assertIs(model_class, Olmo3SinkForCausalLM)

    def test_nested_rope_parameters_are_resolved_per_layer(self) -> None:
        full = _rope_parameters_for_layer(self.config, None)
        sliding = _rope_parameters_for_layer(self.config, self.config.sliding_window)

        self.assertEqual(full["rope_type"], "yarn")
        self.assertEqual(full["factor"], 32.0)
        self.assertEqual(full["attn_factor"], 1.3465735902799727)
        self.assertNotIn("full_attention", full)
        self.assertEqual(
            sliding,
            {"rope_type": "default", "rope_theta": 500000.0},
        )

    def test_sink_sharding_is_contiguous_by_tp_rank(self) -> None:
        weight = torch.arange(40, dtype=torch.bfloat16)
        shard = _select_sink_shard(
            weight,
            total_heads=40,
            local_heads=5,
            tp_rank=3,
        )
        torch.testing.assert_close(shard, weight[15:20])

    def test_invalid_sink_shape_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 40 values"):
            _select_sink_shard(
                torch.zeros(39),
                total_heads=40,
                local_heads=5,
                tp_rank=0,
            )

    def test_target_plugin_has_no_dflash_registration(self) -> None:
        source = (PATCH_ROOT / "src" / "olmo3_sink_vllm" / "__init__.py").read_text()
        self.assertNotIn("DFlash", source)
        self.assertNotIn("speculative", source)


if __name__ == "__main__":
    unittest.main()
