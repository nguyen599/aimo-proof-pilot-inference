"""Export parseable refinement calls as deterministic grader inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.harness_vllm.run import parse_generation_response
from evaluation.harness_vllm.thinking_handoff import (
    parse_saved_proof_generation_call,
)
from evaluation.report_round0_proof_quality import load_rubrics


REFINEMENT_CALL_NAME = re.compile(
    r"^cand_(?P<candidate>\d+)_proof_refine"
    r"(?P<finalize>_finalize)?_r(?P<round>\d+)\.txt$"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_source(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def discover_calls(
    run_dir: Path,
    problem_ids: list[str],
    min_round: int,
    max_round: int | None,
) -> dict[tuple[str, int, int], dict[str, Path]]:
    calls: dict[tuple[str, int, int], dict[str, Path]] = {}
    pattern = "logs/rank*/llm_calls/*/cand_*_proof_refine*.txt"
    for path in sorted(run_dir.glob(pattern)):
        match = REFINEMENT_CALL_NAME.fullmatch(path.name)
        if match is None:
            continue
        problem_id = path.parent.name
        round_index = int(match.group("round"))
        if problem_id not in problem_ids or round_index < min_round:
            continue
        if max_round is not None and round_index > max_round:
            continue
        candidate_index = int(match.group("candidate"))
        kind = "finalize" if match.group("finalize") else "direct"
        key = (problem_id, candidate_index, round_index)
        by_kind = calls.setdefault(key, {})
        if kind in by_kind:
            raise RuntimeError(
                "duplicate refinement call for "
                f"problem={problem_id} candidate={candidate_index} "
                f"round={round_index} kind={kind}: "
                f"{by_kind[kind]} and {path}"
            )
        by_kind[kind] = path
    return calls


def parse_call(path: Path, expected_stage: str) -> dict[str, Any]:
    try:
        saved = parse_saved_proof_generation_call(path, allow_unintervened=True)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "path": path,
            "saved": None,
            "parsed": None,
            "thinking_budget_reached": False,
            "valid": False,
            "parse_error": f"{type(error).__name__}: {error}",
        }
    if saved.stage != expected_stage:
        raise RuntimeError(
            f"expected stage {expected_stage!r} in {path}, found {saved.stage!r}"
        )
    budget_value = saved.usage.get("thinking_budget_applied", False)
    if type(budget_value) is not bool:
        raise RuntimeError(
            f"thinking_budget_applied must be boolean when present in {path}"
        )
    parsed = parse_generation_response(
        saved.output_text,
        require_self_evaluation=True,
    )
    return {
        "path": path,
        "saved": saved,
        "parsed": parsed,
        "thinking_budget_reached": budget_value,
        "valid": bool(parsed["is_valid_candidate_response"]),
        "parse_error": None,
    }


def choose_call(paths: dict[str, Path]) -> tuple[str | None, dict[str, Any] | None, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for kind, path in paths.items():
        expected_stage = "proof_refine_finalize" if kind == "finalize" else "proof_refine"
        parsed[kind] = parse_call(path, expected_stage)

    direct = parsed.get("direct")
    finalize = parsed.get("finalize")
    if direct and direct["valid"] and not direct["thinking_budget_reached"]:
        selected_kind = "direct"
        selected = direct
    elif finalize and finalize["valid"]:
        selected_kind = "finalize"
        selected = finalize
    else:
        selected_kind = None
        selected = None
    return selected_kind, selected, parsed


def export_refinements(
    run_dir: Path,
    rubrics_file: Path,
    output_dir: Path,
    problem_ids: list[str],
    min_round: int,
    max_round: int | None = None,
) -> dict[str, Any]:
    calls = discover_calls(
        run_dir,
        problem_ids,
        min_round,
        max_round,
    )
    if not calls:
        raise RuntimeError("no refinement calls matched the requested filters")
    rubrics = load_rubrics(rubrics_file, problem_ids)
    records: list[dict[str, Any]] = []
    grader_rubrics: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for problem_id, candidate_index, round_index in sorted(
        calls,
        key=lambda key: (int(key[0]) if key[0].isdigit() else key[0], key[1], key[2]),
    ):
        paths = calls[(problem_id, candidate_index, round_index)]
        selected_kind, selected, parsed = choose_call(paths)
        candidate_id = f"p{problem_id}-c{candidate_index:02d}-r{round_index}"
        source_status = {
            kind: {
                "source_log": relative_source(item["path"], run_dir),
                "source_sha256": sha256(item["path"]),
                "thinking_budget_reached": item["thinking_budget_reached"],
                "structurally_complete": item["valid"],
                "finish_reason": (
                    item["saved"].finish_reason if item["saved"] is not None else None
                ),
                "proof_characters": (
                    len(str(item["parsed"]["proof"]))
                    if item["parsed"] is not None
                    else 0
                ),
                "self_score": (
                    item["parsed"]["self_score"]
                    if item["parsed"] is not None
                    else None
                ),
                "parse_error": item["parse_error"],
            }
            for kind, item in parsed.items()
        }
        manifest_row = {
            "candidate_id": candidate_id,
            "problem_id": problem_id,
            "candidate_index": candidate_index,
            "round": round_index,
            "selected": selected is not None,
            "selected_source": selected_kind,
            "sources": source_status,
        }
        manifest.append(manifest_row)
        if selected is None or selected_kind is None:
            continue

        selected_path = selected["saved"].path
        proof = str(selected["parsed"]["proof"]).strip()
        records.append(
            {
                "problem_id": candidate_id,
                "final_proof": proof,
                "source_problem_id": problem_id,
                "candidate_index": candidate_index,
                "round": round_index,
                "source_stage": selected["saved"].stage,
                "source_log": relative_source(selected_path, run_dir),
                "source_sha256": sha256(selected_path),
            }
        )
        source_rubric = rubrics[problem_id]
        grader_rubrics.append(
            {
                "Problem ID": candidate_id,
                "Problem": source_rubric["Problem"],
                "Grading scheme": source_rubric["Grading scheme"],
            }
        )

    if not records:
        raise RuntimeError("no requested refinement call produced a gradeable proof")

    selected_counts = Counter(
        (row["source_problem_id"], row["round"]) for row in records
    )
    rejected_counts = Counter(
        (row["problem_id"], row["round"])
        for row in manifest
        if not row["selected"]
    )
    summary = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "rubrics_file": str(rubrics_file),
        "problem_ids": problem_ids,
        "min_round": min_round,
        "max_round": max_round,
        "discovered_candidate_rounds": len(manifest),
        "grader_candidate_count": len(records),
        "selected_source_counts": dict(
            sorted(Counter(row["source_stage"] for row in records).items())
        ),
        "problems": [
            {
                "problem_id": problem_id,
                "rounds": [
                    {
                        "round": round_index,
                        "selected": selected_counts[(problem_id, round_index)],
                        "rejected": rejected_counts[(problem_id, round_index)],
                    }
                    for round_index in sorted(
                        {
                            row["round"]
                            for row in manifest
                            if row["problem_id"] == problem_id
                        }
                    )
                ],
            }
            for problem_id in problem_ids
        ],
    }
    write_jsonl(output_dir / "records.jsonl", records)
    write_jsonl(output_dir / "rubrics.jsonl", grader_rubrics)
    write_jsonl(output_dir / "manifest.jsonl", manifest)
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--rubrics-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--problem-ids", nargs="+", default=["4", "5"])
    parser.add_argument("--min-round", type=int, default=1)
    parser.add_argument("--max-round", type=int)
    args = parser.parse_args()
    if args.min_round < 1:
        raise ValueError("min round must be at least 1")
    if args.max_round is not None and args.max_round < args.min_round:
        raise ValueError("max round must be at least min round")
    if len(set(args.problem_ids)) != len(args.problem_ids):
        raise ValueError("problem IDs must be unique")
    summary = export_refinements(
        args.run_dir,
        args.rubrics_file,
        args.output_dir,
        [str(problem_id) for problem_id in args.problem_ids],
        args.min_round,
        args.max_round,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
