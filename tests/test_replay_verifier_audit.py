from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation import replay_verifier_audit as replay


def test_load_cases_aligns_problem_ids_and_selected_proofs(tmp_path):
    input_path = tmp_path / "problems.parquet"
    proofs_path = tmp_path / "records.jsonl"
    pd.DataFrame(
        {
            "problem_idx": ["1", "2"],
            "problem": ["First problem.", "Second problem."],
        }
    ).to_parquet(input_path)
    proofs_path.write_text(
        "\n".join(
            [
                json.dumps({"problem_id": "2", "final_proof": "Proof two."}),
                json.dumps({"problem_id": "1", "final_proof": "Proof one."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        input_path=input_path,
        proofs_path=proofs_path,
        problem_id=["2"],
    )

    cases = replay.load_cases(args)

    assert cases == [
        {
            "problem_id": "2",
            "question": "Second problem.",
            "proof": "Proof two.",
            "old_internal_score": None,
            "old_internal_status": None,
        }
    ]


def test_report_includes_role_scores_and_fatal_cap():
    payload = {
        "results": [
            {
                "problem_id": "6",
                "old_internal_score": 1.0,
                "aggregation": {
                    "final_score": 0.5,
                    "final_status": "validated_low_score",
                    "fatal_score_cap_applied": True,
                    "verifier_score_summaries": [
                        {
                            "verifier_role": "counterexample",
                            "verifier_score": 0.0,
                        }
                    ],
                },
            }
        ]
    }

    report = replay.render_report(payload)

    assert "| 6 | 1.0 | 0.5 | validated_low_score | True |" in report
    assert "counterexample=0.0" in report
