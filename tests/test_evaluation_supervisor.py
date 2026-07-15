from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class EvaluationSupervisorTests(unittest.TestCase):
    def test_service_requires_explicit_config_without_gpu_override(self):
        wrapper = (REPO / "evaluation/supervisor/opd32b-eval.sh").read_text()
        self.assertNotIn("CUDA_VISIBLE_DEVICES", wrapper)
        self.assertIn("CONFIG is required", wrapper)
        self.assertIn("--config", wrapper)
        for hidden_override in ("MODEL_MODE", "DFLASH=", "MAXREQ=", "CTX="):
            self.assertNotIn(hidden_override, wrapper)

    def test_supervisor_owns_the_full_process_group(self):
        config = (REPO / "evaluation/supervisor/opd32b-eval.conf").read_text()
        self.assertIn("autostart=false", config)
        self.assertIn("autorestart=unexpected", config)
        self.assertIn("stopasgroup=true", config)
        self.assertIn("killasgroup=true", config)
        self.assertIn("stdout_logfile=/dev/stdout", config)


if __name__ == "__main__":
    unittest.main()
