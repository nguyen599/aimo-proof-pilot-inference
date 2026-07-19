import json
import importlib.util
from pathlib import Path


ANALYZER_PATH = Path(__file__).parents[1] / "evaluation" / "analyze_pipeline_run.py"
SPEC = importlib.util.spec_from_file_location("analyze_pipeline_run", ANALYZER_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)
analyze_run = ANALYZER.analyze_run
parse_call_file = ANALYZER.parse_call_file


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_analyze_run_tracks_selection_rounds_and_external_miscalibration(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": "synthetic",
            "metadata": {
                "meta_n": 1,
                "deepseek_math_v2_candidate_count": 1,
            },
        },
    )
    candidate = {
        "attempt_idx": 0,
        "prompt_family": "deepseek_math_v2",
        "generation_mode": "deepseek",
        "proof_solution": "x" * 40_000,
        "self_score": 1,
        "final_score": 1,
        "final_status": "strict_pass",
        "strict_pass": True,
        "all_verifiers_passed": True,
        "meta_valid_count": 1,
        "meta_checked_count": 1,
        "selected_verification_round": 1,
        "rollback_from_round": None,
        "budget_restart_count": 1,
        "proof_handoffs": ["handoff"],
        "proof_refine_output": [{}],
        "proof_verify_output": [
            {"round_idx": 0, "verifier_index": 0, "parsed": {"score": 0.5}},
            {"round_idx": 1, "verifier_index": 0, "parsed": {"score": 1}},
        ],
        "proof_meta_verify_output": [
            {
                "round_idx": 0,
                "verifier_index": 0,
                "parsed": {"score": 1},
            },
            {
                "round_idx": 1,
                "verifier_index": 0,
                "parsed": {"score": 1},
            },
        ],
        "success": True,
    }
    write_json(
        run_dir / "problems" / "0000_hash" / "rank_0000.json",
        {
            "problem_id": "1",
            "rank": 0,
            "assigned_attempts": [0, 1],
            "pipeline_result": {
                "candidates": [candidate],
                "failed_attempts": [{"attempt_idx": 1, "error": "invalid_generation"}],
            },
        },
    )
    results = run_dir / "logs" / "rank_0000" / "results.jsonl"
    results.parent.mkdir(parents=True)
    results.write_text(json.dumps({"id": "1", "selected_pipeline": 0}) + "\n")
    grader = run_dir / "grader.json"
    write_json(
        grader,
        {"problems": [{"problem_id": "1", "score_out_of_7": 1}]},
    )

    analysis = analyze_run(run_dir, grader_summary=grader)

    assert analysis["overview"]["valid_candidates"] == 1
    assert analysis["overview"]["failed_candidates"] == 1
    assert analysis["overview"]["refinement_improved_candidates"] == 1
    assert analysis["problems"][0]["selected_internal_rank"] == 1
    assert analysis["problems"][0]["selected_selector_truncated"] is True
    assert analysis["diagnosis"]["high_confidence_external_failures"] == ["1"]
    assert "verifier calibration" in analysis["diagnosis"]["summary"]


def test_parse_call_file_extracts_usage_detail_and_last_score(tmp_path):
    path = tmp_path / "llm_calls" / "3" / "cand_2_verify_r1_v0.txt"
    path.parent.mkdir(parents=True)
    path.write_text(
        "stage: proof_verify\n"
        "detail: candidate=2 round=1 verifier=0 prompt_family=opd\n"
        "prompt_tokens: 20\n"
        "max_tokens: 100\n\n"
        "===== INPUT PROMPT =====\ninput\n\n"
        "===== OUTPUT =====\n"
        "success: True\n"
        "error: None\n"
        "finish_reason: stop\n"
        'usage: {"completion_tokens": 8, "total_tokens": 28, '
        '"thinking_budget_applied": true}\n'
        "server_url: http://localhost\n"
        "latency_s: 1.25\n\n"
        "<score>0</score> scratch <score>0.5</score>\n",
        encoding="utf-8",
    )

    row = parse_call_file(path)

    assert row["problem_id"] == "3"
    assert row["candidate"] == 2
    assert row["round"] == 1
    assert row["verifier"] == 0
    assert row["prompt_tokens"] == 20
    assert row["completion_tokens"] == 8
    assert row["thinking_budget_applied"] is True
    assert row["parsed_output_score"] == 0.5
