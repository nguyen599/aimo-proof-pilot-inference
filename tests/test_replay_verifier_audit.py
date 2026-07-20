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


def test_load_cases_parses_completed_opd_proof_generation_calls(tmp_path):
    input_path = tmp_path / "problems.parquet"
    calls_dir = tmp_path / "llm_calls"
    call_path = calls_dir / "2" / "cand_7_proof_gen_r1.txt"
    pd.DataFrame(
        {
            "problem_idx": ["2"],
            "problem": ["Prove the test statement."],
        }
    ).to_parquet(input_path)
    call_path.parent.mkdir(parents=True)
    call_path.write_text(
        "\n".join(
            [
                "stage: proof_generation",
                "detail: candidate=7 round=1 mode=opd_xml prompt_family=opd",
                "prompt_tokens: 100",
                "max_tokens: 1000",
                "",
                "===== INPUT PROMPT =====",
                "Rendered prompt.",
                "",
                "===== OUTPUT =====",
                "success: True",
                "error: None",
                "finish_reason: stop",
                'usage: {"completion_tokens": 50}',
                "server_url: http://127.0.0.1:8000/v1",
                "latency_s: 1.5",
                "",
                "<think>discard this draft</think>",
                "<solution>A rigorous final proof.</solution>",
                "<self_evaluation>The proof is complete.</self_evaluation>",
                "<score>1</score>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        input_path=input_path,
        proofs_path=None,
        llm_calls_dir=calls_dir,
        problem_id=["2"],
        candidate_id=[],
        round_index=[],
        proof_prompt_family="opd",
        max_cases=0,
    )

    cases = replay.load_cases(args)

    assert cases == [
        {
            "problem_id": "2",
            "question": "Prove the test statement.",
            "proof": "A rigorous final proof.",
            "self_evaluation": "The proof is complete.",
            "prompt_family": "opd",
            "source_candidate_id": 7,
            "source_round_index": 1,
            "source_path": str(call_path),
            "source_finish_reason": "stop",
            "old_internal_score": 1.0,
            "old_internal_status": "raw_proof_generation",
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

    assert "| 6 | - | - | - | 1.0 | 0.5 | validated_low_score | True | True |" in report
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


def test_replay_preserves_successes_when_another_case_never_parses(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "problems.parquet"
    proofs_path = tmp_path / "records.jsonl"
    output_dir = tmp_path / "audit"
    pd.DataFrame(
        {
            "problem_idx": ["1", "2"],
            "problem": ["First problem.", "Second problem."],
        }
    ).to_parquet(input_path)
    proofs_path.write_text(
        "\n".join(
            [
                json.dumps({"problem_id": "1", "final_proof": "Proof one."}),
                json.dumps({"problem_id": "2", "final_proof": "Proof two."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class DummyScheduler:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    async def fake_verification_round(*args, **kwargs):
        score = 1.0 if args[3] < 200 else None
        return (
            [{"score": score}],
            [],
            {},
            [],
            {
                "final_score": score,
                "final_status": "strict_pass" if score == 1.0 else "invalid",
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
        problem_id=[],
        served_model_name="proof-model",
        api_key="test",
        verify_n=1,
        meta_n=0,
        meta_policy="all-reviews",
        max_concurrent_problems=2,
        max_concurrent_requests=2,
        verifier_max_tokens=128,
        meta_max_tokens=128,
        temperature=1.0,
        top_p=0.95,
        request_timeout_seconds=60.0,
        max_attempts=2,
    )

    payload = asyncio.run(replay.replay(args))

    assert payload["schema_version"] == 2
    assert [item["problem_id"] for item in payload["results"]] == ["1"]
    assert payload["errors"][0]["problem_id"] == "2"
    assert payload["errors"][0]["error_type"] == "incomplete_verifier_scores"
    assert len(payload["errors"][0]["failed_attempts"]) == 2
    assert json.loads(
        (output_dir / "cases" / "problem-1.json").read_text(encoding="utf-8")
    )["status"] == "ok"
    assert json.loads(
        (output_dir / "cases" / "problem-2.json").read_text(encoding="utf-8")
    )["status"] == "error"

    report = replay.render_report(payload)
    assert "## Incomplete cases" in report
    assert "| 2 | - | - | incomplete_verifier_scores | 2 |" in report
