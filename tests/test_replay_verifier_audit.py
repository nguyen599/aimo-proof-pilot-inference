from __future__ import annotations

import asyncio
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
                    "validated_low_score_cap_applied": True,
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

    assert (
        "| 6 | 1.0 | 0.5 | validated_low_score | True | True |" in report
    )
    assert "counterexample=0.0" in report


def test_replay_passes_progress_for_raw_call_logging(tmp_path, monkeypatch):
    input_path = tmp_path / "problems.parquet"
    proofs_path = tmp_path / "records.jsonl"
    output_dir = tmp_path / "audit"
    pd.DataFrame(
        {"problem_idx": ["1"], "problem": ["Test problem."]}
    ).to_parquet(input_path)
    proofs_path.write_text(
        json.dumps({"problem_id": "1", "final_proof": "Test proof."}) + "\n",
        encoding="utf-8",
    )
    captured = {}

    class DummyScheduler:
        def __init__(self, **kwargs):
            captured["scheduler_kwargs"] = kwargs

        def close(self):
            captured["closed"] = True

    async def fake_verification_round(*args, **kwargs):
        captured["progress"] = kwargs.get("progress")
        return (
            [{"score": 1.0}],
            [],
            {},
            [],
            {
                "final_score": 1.0,
                "final_status": "strict_pass",
                "verifier_score_summaries": [],
            },
        )

    monkeypatch.setattr(replay, "ChatScheduler", DummyScheduler)
    monkeypatch.setattr(
        replay.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(replay, "run_verification_round", fake_verification_round)
    args = SimpleNamespace(
        input_path=input_path,
        proofs_path=proofs_path,
        model_path=tmp_path / "model",
        base_url=["http://127.0.0.1:8000/v1"],
        output_dir=output_dir,
        problem_id=["1"],
        served_model_name="proof-model",
        api_key="test",
        verify_n=1,
        meta_n=0,
        meta_policy="all-reviews",
        max_concurrent_problems=1,
        max_concurrent_requests=1,
        verifier_max_tokens=128,
        meta_max_tokens=128,
        temperature=1.0,
        top_p=0.95,
        request_timeout_seconds=60.0,
        max_attempts=3,
    )

    payload = asyncio.run(replay.replay(args))

    assert payload["results"][0]["problem_id"] == "1"
    assert captured["progress"].problem_id == "1"
    assert captured["scheduler_kwargs"]["llm_call_logdir"] == output_dir / "llm_calls"
    assert captured["closed"] is True


def test_replay_retries_incomplete_verifier_scores(tmp_path, monkeypatch):
    input_path = tmp_path / "problems.parquet"
    proofs_path = tmp_path / "records.jsonl"
    pd.DataFrame(
        {"problem_idx": ["1"], "problem": ["Test problem."]}
    ).to_parquet(input_path)
    proofs_path.write_text(
        json.dumps({"problem_id": "1", "final_proof": "Test proof."}) + "\n",
        encoding="utf-8",
    )
    calls = []

    class DummyScheduler:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    async def fake_verification_round(*args, **kwargs):
        calls.append(args[3])
        score = None if len(calls) == 1 else 0.0
        return (
            [{"score": score}],
            [],
            {},
            [],
            {"final_score": score, "verifier_score_summaries": []},
        )

    monkeypatch.setattr(replay, "ChatScheduler", DummyScheduler)
    monkeypatch.setattr(
        replay.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(replay, "run_verification_round", fake_verification_round)
    args = SimpleNamespace(
        input_path=input_path,
        proofs_path=proofs_path,
        model_path=tmp_path / "model",
        base_url=["http://127.0.0.1:8000/v1"],
        output_dir=tmp_path / "audit",
        problem_id=["1"],
        served_model_name="proof-model",
        api_key="test",
        verify_n=1,
        meta_n=0,
        meta_policy="all-reviews",
        max_concurrent_problems=1,
        max_concurrent_requests=1,
        verifier_max_tokens=128,
        meta_max_tokens=128,
        temperature=1.0,
        top_p=0.95,
        request_timeout_seconds=60.0,
        max_attempts=2,
    )

    payload = asyncio.run(replay.replay(args))

    result = payload["results"][0]
    assert calls == [100, 101]
    assert result["replay_attempt"] == 2
    assert result["failed_attempts"] == [
        {"attempt": 1, "verifier_scores": [None]}
    ]
    assert result["verifier_results"] == [{"score": 0.0}]
