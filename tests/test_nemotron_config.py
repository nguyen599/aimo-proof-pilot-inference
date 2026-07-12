from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from eval_config import active_model, load_config  # noqa: E402
from launch_server import decode_graph_batches  # noqa: E402


class NemotronConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = REPO / "evaluation/configs/nemotron_cascade2.yaml"
        cls.config = load_config(cls.path)

    def test_checked_in_config_is_full_uniform_policy(self):
        search = self.config["search"]
        self.assertEqual(search["proofs_per_round"], 128)
        self.assertEqual(search["verifications_per_proof"], 64)
        self.assertEqual(search["top_proofs"], 32)
        self.assertEqual(search["refinements_per_proof"], 4)
        self.assertEqual(search["analyses_per_refinement"], 8)
        self.assertEqual(search["max_rounds"], 8)

    def test_default_is_bf16_target_only_tp2(self):
        model = active_model(self.config)
        self.assertEqual(model.mode, "bf16")
        self.assertEqual(model.tensor_parallel_size, 2)
        self.assertFalse(model.quantized)
        self.assertFalse(model.dflash)
        self.assertIsNone(model.draft)

    def test_quantization_and_dflash_are_independent(self):
        expected = {
            (False, False): ("bf16", "opd-32b-deploy", None),
            (True, False): ("humming_w4a8", "opd-32b-v33-s200-gptq-w4a16", None),
            (False, True): ("bf16", "opd-32b-deploy", "dflash-32b-draft-v2test-phaseL"),
            (True, True): (
                "humming_w4a8",
                "opd-32b-v33-s200-gptq-w4a16",
                "dflash-32b-draft-v2test-phaseL-int4mlp",
            ),
        }
        for flags, paths in expected.items():
            with self.subTest(flags=flags):
                config = copy.deepcopy(self.config)
                config["model"]["quantized"], config["model"]["dflash"] = flags
                model = active_model(config)
                self.assertEqual(model.mode, paths[0])
                self.assertEqual(model.target.name, paths[1])
                self.assertEqual(model.draft.name if model.draft else None, paths[2])

    def test_decode_graphs_cover_configured_ceiling(self):
        batches = decode_graph_batches(48)
        self.assertEqual(batches[:16], list(range(1, 17)))
        self.assertEqual(batches[-1], 48)
        self.assertNotIn(64, batches)

    def test_launcher_has_one_config_interface(self):
        launcher = (REPO / "serve_opd32b.sh").read_text()
        self.assertIn("nemotron_cascade2.yaml", launcher)
        self.assertIn("launch_server.py", launcher)
        self.assertNotIn("MODEL_MODE", launcher)
        self.assertNotIn("DFLASH=", launcher)


if __name__ == "__main__":
    unittest.main()
