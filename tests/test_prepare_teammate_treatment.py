import tempfile
import unittest
from pathlib import Path

import yaml

from evaluation.prepare_teammate_treatment import build_config, write_config


class PrepareTeammateTreatmentTests(unittest.TestCase):
    def test_builds_four_round_cumulative_tournament_treatment(self) -> None:
        config = build_config(
            model_path=Path("/models/ours"),
            draft_path=Path("/models/draft"),
            tensor_parallel_size=2,
            data_parallel_size=2,
            server_port=31000,
            proofs_per_round=64,
            verifications_per_proof=8,
            top_proofs=16,
            refine_parents=4,
            reviews_per_parent=3,
            max_rounds=4,
            max_running_requests=32,
            search_concurrency=64,
        )
        self.assertEqual(config["models"]["bf16_target"], "/models/ours")
        self.assertEqual(config["model"]["tensor_parallel_size"], 2)
        self.assertEqual(config["model"]["data_parallel_size"], 2)
        search = config["search"]
        self.assertEqual(search["proofs_per_round"], 64)
        self.assertEqual(search["verifications_per_proof"], 8)
        self.assertEqual(search["top_proofs"], 16)
        self.assertEqual(search["refine_parents"], 4)
        self.assertEqual(search["reviews_per_refine_parent"], 3)
        self.assertEqual(search["max_rounds"], 4)
        self.assertEqual(search["early_stop_threshold"], 1.0)
        self.assertTrue(search["llm_selector"])
        self.assertTrue(search["selection_tournament"])
        self.assertEqual(search["selection_tournament_rounds"], 64)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "config.yaml"
            write_config(output, config)
            self.assertEqual(yaml.safe_load(output.read_text()), config)

    def test_rejects_invalid_refinement_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "reviews_per_parent"):
            build_config(
                model_path=Path("/models/ours"),
                draft_path=Path("/models/draft"),
                tensor_parallel_size=2,
                data_parallel_size=2,
                server_port=31000,
                proofs_per_round=64,
                verifications_per_proof=2,
                top_proofs=16,
                refine_parents=4,
                reviews_per_parent=3,
                max_rounds=4,
                max_running_requests=32,
                search_concurrency=64,
            )


if __name__ == "__main__":
    unittest.main()
