"""Export every final candidate from a completed distributed proof-search run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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


def render_grading_scheme(items: Any) -> str:
    rendered: list[str] = []
    total_points = 0
    for index, item in enumerate(list(items), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"grading scheme item {index} is not a mapping")
        missing = {"title", "points", "desc"} - set(item)
        if missing:
            raise ValueError(
                f"grading scheme item {index} missing fields: {sorted(missing)}"
            )
        points = int(item["points"])
        total_points += points
        rendered.append(f"{index}. [{points} pts] {item['title']}: {item['desc']}")
    if total_points != 7:
        raise ValueError(f"grading scheme totals {total_points}, expected 7")
    return "\n".join(rendered)


def load_rubrics(path: Path) -> dict[str, dict[str, str]]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        source_rows = pd.read_parquet(path).to_dict(orient="records")
        rows = [
            {
                "Problem ID": str(row["problem_idx"]),
                "Problem": str(row["problem"]),
                "Grading scheme": render_grading_scheme(row["grading_scheme"]),
            }
            for row in source_rows
        ]
    elif path.suffix.lower() in {".json", ".jsonl"}:
        rows = read_jsonl(path)
    else:
        raise ValueError(f"unsupported rubric file: {path}")

    normalized: dict[str, dict[str, str]] = {}
    required = {"Problem ID", "Problem", "Grading scheme"}
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"rubric row {index} missing fields: {sorted(missing)}")
        problem_id = str(row["Problem ID"])
        if problem_id in normalized:
            raise ValueError(f"duplicate rubric problem ID {problem_id!r}")
        normalized[problem_id] = {
            "Problem ID": problem_id,
            "Problem": str(row["Problem"]),
            "Grading scheme": str(row["Grading scheme"]),
        }
    return normalized


def load_primary_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    by_problem: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("logs/rank_*/results.jsonl")):
        for row in read_jsonl(path):
            if row.get("distributed_worker"):
                continue
            problem_id = str(row["id"])
            if problem_id in by_problem:
                raise RuntimeError(
                    f"duplicate primary result for problem {problem_id!r}"
                )
            by_problem[problem_id] = row
    return by_problem


def candidate_id(problem_id: str, attempt_idx: int, proof_version: str) -> str:
    safe_problem_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", problem_id).strip("_")
    if not safe_problem_id:
        raise ValueError(f"problem ID {problem_id!r} has no safe characters")
    if proof_version not in {"initial", "final"}:
        raise ValueError(f"unsupported proof version {proof_version!r}")
    return f"p{safe_problem_id}-c{attempt_idx:02d}-{proof_version}"


def count_outputs(candidate: dict[str, Any], field: str) -> int:
    value = candidate.get(field) or []
    if not isinstance(value, list):
        raise ValueError(f"candidate field {field!r} must be a list")
    return len(value)


def candidate_proofs(
    candidate: dict[str, Any],
    proof_versions: list[str],
) -> list[tuple[str, str]]:
    proofs: list[tuple[str, str]] = []
    for proof_version in proof_versions:
        if proof_version == "initial":
            generation = candidate.get("proof_generation_output") or {}
            parsed = generation.get("parsed") or {}
            proof = str(parsed.get("proof") or "").strip()
        elif proof_version == "final":
            proof = str(candidate.get("proof_solution") or "").strip()
        else:
            raise ValueError(f"unsupported proof version {proof_version!r}")
        if not proof:
            raise RuntimeError(
                f"empty {proof_version} proof for attempt {candidate.get('attempt_idx')}"
            )
        proofs.append((proof_version, proof))
    return proofs


def validate_problem_payloads(
    paths: list[Path],
    *,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    world_size = int(manifest["world_size"])
    pipelines_per_problem = int(manifest["metadata"]["pipelines_per_problem"])
    if len(paths) != world_size:
        raise RuntimeError(
            f"expected {world_size} rank payloads in {paths[0].parent}, "
            f"found {len(paths)}"
        )

    loaded = [(path, read_json(path)) for path in paths]
    loaded.sort(key=lambda item: int(item[1]["rank"]))
    ranks = [int(payload["rank"]) for _, payload in loaded]
    if ranks != list(range(world_size)):
        raise RuntimeError(f"incomplete rank set in {paths[0].parent}: {ranks}")

    first = loaded[0][1]
    identity_fields = (
        "run_id",
        "world_size",
        "problem_ordinal",
        "problem_id",
        "question_hash",
    )
    for path, payload in loaded:
        if payload.get("run_id") != manifest.get("run_id"):
            raise RuntimeError(f"run ID mismatch in {path}")
        for field in identity_fields:
            if payload.get(field) != first.get(field):
                raise RuntimeError(f"problem field {field!r} differs in {path}")

    assigned: set[int] = set()
    observed_outcomes = 0
    known_outcome_attempts: set[int] = set()
    for path, payload in loaded:
        rank_attempts = [int(value) for value in payload.get("assigned_attempts", [])]
        if len(rank_attempts) != len(set(rank_attempts)):
            raise RuntimeError(f"duplicate assigned attempts in {path}")
        overlap = assigned.intersection(rank_attempts)
        if overlap:
            raise RuntimeError(
                f"candidate attempts assigned to multiple ranks: {sorted(overlap)}"
            )
        assigned.update(rank_attempts)

        result = payload.get("pipeline_result")
        if not isinstance(result, dict):
            raise RuntimeError(f"missing pipeline_result in {path}")
        outcome_rows = (
            list(result.get("candidates") or [])
            + list(result.get("failed_attempts") or [])
            + list(result.get("skipped_generations") or [])
        )
        observed_outcomes += len(outcome_rows) + int(result.get("cancelled_count") or 0)
        rank_attempt_set = set(rank_attempts)
        for outcome in outcome_rows:
            attempt = outcome.get("attempt_idx")
            if attempt is None:
                continue
            attempt_idx = int(attempt)
            if attempt_idx not in rank_attempt_set:
                raise RuntimeError(
                    f"rank {payload['rank']} returned unassigned attempt {attempt_idx}"
                )
            if attempt_idx in known_outcome_attempts:
                raise RuntimeError(f"duplicate outcome for attempt {attempt_idx}")
            known_outcome_attempts.add(attempt_idx)

    expected_attempts = set(range(pipelines_per_problem))
    if assigned != expected_attempts:
        raise RuntimeError(
            "distributed assignment mismatch: "
            f"missing={sorted(expected_attempts - assigned)} "
            f"extra={sorted(assigned - expected_attempts)}"
        )
    if observed_outcomes != pipelines_per_problem:
        raise RuntimeError(
            f"problem {first['problem_id']!r} accounts for {observed_outcomes} "
            f"of {pipelines_per_problem} candidate attempts"
        )
    return first, loaded


def candidate_manifest_row(
    *,
    candidate: dict[str, Any],
    candidate_position: int,
    payload: dict[str, Any],
    payload_path: Path,
    run_dir: Path,
    selected_attempt: int | None,
    proof_version: str,
    proof: str,
) -> dict[str, Any]:
    attempt_idx = int(candidate["attempt_idx"])
    problem_id = str(payload["problem_id"])
    return {
        "candidate_id": candidate_id(problem_id, attempt_idx, proof_version),
        "run_id": str(payload["run_id"]),
        "problem_id": problem_id,
        "problem_ordinal": int(payload["problem_ordinal"]),
        "question_hash": str(payload["question_hash"]),
        "rank": int(payload["rank"]),
        "attempt_idx": attempt_idx,
        "candidate_position_in_payload": candidate_position,
        "proof_version": proof_version,
        "is_final_version": proof_version == "final",
        "selected_by_pipeline": attempt_idx == selected_attempt,
        "prompt_family": str(candidate.get("prompt_family") or "unknown"),
        "planning_strategy": str(candidate.get("planning_strategy") or "baseline"),
        "generation_mode": candidate.get("generation_mode"),
        "generation_only": bool(candidate.get("generation_only", False)),
        "final_score": candidate.get("final_score"),
        "pre_cap_score": candidate.get(
            "pre_cap_score",
            candidate.get("final_score"),
        ),
        "final_status": candidate.get("final_status"),
        "self_score": candidate.get("self_score"),
        "strict_pass": bool(candidate.get("strict_pass", False)),
        "all_verifiers_passed": bool(candidate.get("all_verifiers_passed", False)),
        "selected_verification_round": candidate.get("selected_verification_round"),
        "rollback_from_round": candidate.get("rollback_from_round"),
        "budget_restart_count": int(candidate.get("budget_restart_count") or 0),
        "refine_budget_restart_count": int(
            candidate.get("refine_budget_restart_count") or 0
        ),
        "proof_generation_output_count": count_outputs(
            candidate, "proof_generation_outputs"
        ),
        "proof_handoff_count": count_outputs(candidate, "proof_handoffs"),
        "verifier_call_count": count_outputs(candidate, "proof_verify_output"),
        "meta_call_count": count_outputs(candidate, "proof_meta_verify_output"),
        "refinement_count": count_outputs(candidate, "proof_refine_output"),
        "refine_attempt_count": count_outputs(candidate, "proof_refine_attempt_output"),
        "refine_handoff_count": count_outputs(candidate, "proof_refine_handoffs"),
        "validated_critique_count": count_outputs(candidate, "validated_critiques"),
        "proof_characters": len(proof),
        "proof_sha256": hashlib.sha256(proof.encode("utf-8")).hexdigest(),
        "source_payload": str(payload_path.relative_to(run_dir)),
        "source_payload_sha256": sha256(payload_path),
    }


def export_candidates(
    run_dir: Path,
    rubrics_file: Path,
    output_dir: Path,
    problem_ids: list[str] | None = None,
    proof_versions: list[str] | None = None,
) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    rubrics = load_rubrics(rubrics_file)
    selected_results = load_primary_results(run_dir)
    requested = set(problem_ids or [])
    resolved_proof_versions = proof_versions or ["final"]
    if (
        not resolved_proof_versions
        or len(resolved_proof_versions) != len(set(resolved_proof_versions))
        or any(value not in {"initial", "final"} for value in resolved_proof_versions)
    ):
        raise ValueError("proof versions must be a unique subset of initial and final")

    problem_dirs = sorted(
        path for path in (run_dir / "problems").iterdir() if path.is_dir()
    )
    if not problem_dirs:
        raise RuntimeError(f"no distributed problem payloads found under {run_dir}")

    grader_rows: list[dict[str, Any]] = []
    grader_rubrics: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    problem_summaries: list[dict[str, Any]] = []
    seen_problem_ids: set[str] = set()

    for problem_dir in problem_dirs:
        paths = sorted(problem_dir.glob("rank_*.json"))
        first, loaded = validate_problem_payloads(paths, manifest=manifest)
        problem_id = str(first["problem_id"])
        if requested and problem_id not in requested:
            continue
        if problem_id in seen_problem_ids:
            raise RuntimeError(
                f"duplicate payload directory for problem {problem_id!r}"
            )
        seen_problem_ids.add(problem_id)
        if problem_id not in rubrics:
            raise RuntimeError(f"missing rubric for problem {problem_id!r}")

        primary = selected_results.get(problem_id) or {}
        selected_value = primary.get("selected_pipeline")
        selected_attempt = int(selected_value) if selected_value is not None else None
        candidates: list[tuple[Path, dict[str, Any], int, dict[str, Any]]] = []
        failed_count = 0
        skipped_count = 0
        cancelled_count = 0
        for payload_path, payload in loaded:
            result = payload["pipeline_result"]
            for position, candidate in enumerate(result.get("candidates") or []):
                candidates.append((payload_path, payload, position, candidate))
            failed_count += len(result.get("failed_attempts") or [])
            skipped_count += len(result.get("skipped_generations") or [])
            cancelled_count += int(result.get("cancelled_count") or 0)
        candidates.sort(key=lambda item: int(item[3]["attempt_idx"]))

        problem_candidate_rows: list[dict[str, Any]] = []
        for payload_path, payload, position, candidate in candidates:
            candidate_versions = candidate_proofs(candidate, resolved_proof_versions)
            for proof_version, proof in candidate_versions:
                metadata = candidate_manifest_row(
                    candidate=candidate,
                    candidate_position=position,
                    payload=payload,
                    payload_path=payload_path,
                    run_dir=run_dir,
                    selected_attempt=selected_attempt,
                    proof_version=proof_version,
                    proof=proof,
                )
                candidate_identifier = metadata["candidate_id"]
                grader_rows.append(
                    {
                        "problem_id": candidate_identifier,
                        "final_proof": proof,
                        "source_problem_id": problem_id,
                        "candidate_index": metadata["attempt_idx"],
                        "proof_version": proof_version,
                        "source_payload": metadata["source_payload"],
                        "source_payload_sha256": metadata["source_payload_sha256"],
                    }
                )
                source_rubric = rubrics[problem_id]
                grader_rubrics.append(
                    {
                        "Problem ID": candidate_identifier,
                        "Problem": source_rubric["Problem"],
                        "Grading scheme": source_rubric["Grading scheme"],
                    }
                )
                candidate_rows.append(metadata)
                problem_candidate_rows.append(metadata)

        final_candidate_rows = [
            row for row in problem_candidate_rows if row["is_final_version"]
        ]
        candidate_summary_rows = final_candidate_rows or [
            row for row in problem_candidate_rows if row["proof_version"] == "initial"
        ]
        final_status_counts = Counter(
            str(row["final_status"]) for row in candidate_summary_rows
        )
        strategy_counts = Counter(
            row["planning_strategy"] for row in candidate_summary_rows
        )
        problem_summaries.append(
            {
                "problem_id": problem_id,
                "problem_ordinal": int(first["problem_ordinal"]),
                "assigned_candidates": int(
                    manifest["metadata"]["pipelines_per_problem"]
                ),
                "completed_candidates": len(candidates),
                "exported_proof_versions": len(problem_candidate_rows),
                "failed_candidates": failed_count,
                "skipped_candidates": skipped_count,
                "cancelled_candidates": cancelled_count,
                "selected_attempt": selected_attempt,
                "selected_candidate_exported": any(
                    row["selected_by_pipeline"] for row in candidate_summary_rows
                ),
                "final_status_counts": dict(sorted(final_status_counts.items())),
                "planning_strategy_counts": dict(sorted(strategy_counts.items())),
            }
        )

    if requested != seen_problem_ids and requested:
        raise RuntimeError(
            f"requested problems missing from run: {sorted(requested - seen_problem_ids)}"
        )
    if not candidate_rows:
        raise RuntimeError("no final candidates were exported")

    summary = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "source_manifest": str(manifest_path.relative_to(run_dir)),
        "source_manifest_sha256": sha256(manifest_path),
        "world_size": int(manifest["world_size"]),
        "pipelines_per_problem": int(manifest["metadata"]["pipelines_per_problem"]),
        "problem_ids": [row["problem_id"] for row in problem_summaries],
        "proof_versions": resolved_proof_versions,
        "problems": problem_summaries,
        "exported_candidates": sum(
            row["completed_candidates"] for row in problem_summaries
        ),
        "exported_proof_versions": len(candidate_rows),
    }
    write_jsonl(output_dir / "records.jsonl", grader_rows)
    write_jsonl(output_dir / "rubrics.jsonl", grader_rubrics)
    write_jsonl(output_dir / "candidate_manifest.jsonl", candidate_rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_problem_ids(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    normalized = [str(value).strip() for value in values]
    if any(not value for value in normalized) or len(set(normalized)) != len(
        normalized
    ):
        raise ValueError("problem IDs must be nonempty and unique")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--rubrics-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--problem-ids", nargs="+")
    parser.add_argument(
        "--proof-versions",
        choices=("final", "initial-final"),
        default="final",
        help="Export only final proofs or paired initial and final proofs.",
    )
    args = parser.parse_args()
    proof_versions = (
        ["final"] if args.proof_versions == "final" else ["initial", "final"]
    )
    summary = export_candidates(
        args.run_dir,
        args.rubrics_file,
        args.output_dir,
        parse_problem_ids(args.problem_ids),
        proof_versions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
