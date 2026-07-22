from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.replay_pipeline_selector import (
    build_selector_config,
    load_candidate_export,
    replay_selectors,
    write_replay_outputs,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_export(root: Path) -> Path:
    candidate_dir = root / "candidates"
    records = []
    manifests = []
    rubrics = []
    for attempt in range(4):
        candidate_id = f"p4-c{attempt:02d}-final"
        proof = "TARGET PROOF" if attempt == 3 else f"ordinary proof {attempt}"
        records.append(
            {
                "problem_id": candidate_id,
                "source_problem_id": "4",
                "candidate_index": attempt,
                "proof_version": "final",
                "final_proof": proof,
            }
        )
        manifests.append(
            {
                "candidate_id": candidate_id,
                "problem_id": "4",
                "attempt_idx": attempt,
                "proof_version": "final",
                "final_score": 1.0 - attempt / 10,
                "pre_cap_score": 1.25 - attempt / 10,
                "final_status": "weighted_score_pass",
                "selected_by_pipeline": attempt == 0,
                "planning_strategy": "baseline",
                "verifier_score_summaries": [
                    {"verifier_index": 0, "verifier_score": 1.0 - attempt / 10},
                    {"verifier_index": 1, "verifier_score": 1.0},
                ],
            }
        )
        rubrics.append(
            {
                "Problem ID": candidate_id,
                "Problem": "Prove the target statement.",
                "Grading scheme": "1. [7 pts] Proof: Complete and correct.",
            }
        )
    write_jsonl(candidate_dir / "records.jsonl", records)
    write_jsonl(candidate_dir / "candidate_manifest.jsonl", manifests)
    write_jsonl(candidate_dir / "rubrics.jsonl", rubrics)
    return candidate_dir


class TargetScheduler:
    def __init__(self) -> None:
        self.calls = []

    async def call(self, stage, prompt, **kwargs):
        assert stage == "selector"
        self.calls.append(kwargs)
        content = prompt[-1]["content"]
        target_position = content.find("TARGET PROOF")
        candidate_position = content.rfind('<candidate id="R', 0, target_position)
        selected = int(content[candidate_position:].split('"', 2)[1][1:])
        return {
            "success": True,
            "error": None,
            "text": f"<selected_id>R{selected}</selected_id>",
            "finish_reason": "stop",
            "usage": {},
            "server_url": "mock",
            "latency_s": 0.0,
        }


def selector_config() -> SimpleNamespace:
    return SimpleNamespace(
        selector_mode="llm_stratified_tournament",
        selector_score_source="raw_verifier_mean",
        selector_max_candidate_chars=10_000,
        selector_thinking_budget_tokens=56_000,
        selector_thinking_budget_force_text=(
            "\n</think>\n\n<selected_id>"
        ),
        selection_temperature=0.3,
        selector_tournament_group_size=4,
        selector_tournament_rounds=8,
        selector_tournament_max_candidates=4,
        selector_tournament_threshold=0.5,
        selector_tournament_force_wide_pool=True,
        selector_score_window=0.2,
        selector_vote_count=4,
    )


def test_loads_final_candidates_and_preserves_uncapped_score(tmp_path: Path) -> None:
    problems = load_candidate_export(make_export(tmp_path))
    assert len(problems) == 1
    assert problems[0].problem_id == "4"
    assert [candidate["attempt_idx"] for candidate in problems[0].candidates] == [
        0,
        1,
        2,
        3,
    ]
    assert problems[0].candidates[3]["pre_cap_score"] == 0.95
    assert problems[0].candidates[3]["verifier_score_summaries"][0][
        "verifier_score"
    ] == 0.7


def test_replays_tournament_and_writes_grader_records(tmp_path: Path) -> None:
    candidate_dir = make_export(tmp_path)
    problems = load_candidate_export(candidate_dir)
    config = selector_config()
    scheduler = TargetScheduler()
    results = asyncio.run(replay_selectors(problems, scheduler, config))
    assert results[0]["selected_candidate_id"] == "p4-c03-final"
    assert scheduler.calls
    assert all(
        call["thinking_budget_tokens"] == 56_000 for call in scheduler.calls
    )
    assert all(
        call["thinking_budget_force_text"].endswith("<selected_id>")
        for call in scheduler.calls
    )

    output_dir = tmp_path / "replay"
    summary = write_replay_outputs(
        output_dir,
        results,
        candidate_dir=candidate_dir,
        selector_config=config,
    )
    assert summary["changed_from_original"] == 1
    assert summary["selector_score_source"] == "raw_verifier_mean"
    record = json.loads((output_dir / "records.jsonl").read_text().splitlines()[0])
    assert record["problem_id"] == "4"
    assert record["source_candidate_id"] == "p4-c03-final"
    assert record["final_proof"] == "TARGET PROOF"


def test_build_selector_config_clamps_count_fields() -> None:
    args = argparse.Namespace(
        selector_mode="llm_stratified_tournament",
        selector_score_source="raw_verifier_mean",
        selector_max_candidate_chars=1,
        selector_thinking_budget_tokens=56_000,
        selection_temperature=0.3,
        selector_tournament_group_size=1,
        selector_tournament_rounds=0,
        selector_tournament_max_candidates=1,
        selector_tournament_threshold=0.5,
        selector_tournament_force_wide_pool=True,
        selector_score_window=0.2,
        selector_vote_count=0,
    )
    config = build_selector_config(args)
    assert config.selector_max_candidate_chars == 1_000
    assert config.selector_score_source == "raw_verifier_mean"
    assert config.selector_thinking_budget_tokens == 56_000
    assert config.selector_tournament_group_size == 2
    assert config.selector_tournament_rounds == 1
    assert config.selector_tournament_max_candidates == 2
    assert config.selector_vote_count == 1
