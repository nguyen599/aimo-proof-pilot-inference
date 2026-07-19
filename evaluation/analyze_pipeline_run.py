#!/usr/bin/env python3
"""Reduce a proof-pipeline run into candidate, stage, and bottleneck metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


INPUT_MARKER = "===== INPUT PROMPT ====="
OUTPUT_MARKER = "===== OUTPUT ====="
HEADER_PATTERN = re.compile(r"^([a-z_]+):\s*(.*)$")
DETAIL_PATTERNS = {
    "candidate": re.compile(r"\bcandidate=(\d+)"),
    "round": re.compile(r"\bround=(\d+)"),
    "verifier": re.compile(r"\bverifier=(\d+)"),
    "meta": re.compile(r"\bmeta=(\d+)"),
}
XML_SCORE_PATTERN = re.compile(r"<score>\s*(0(?:\.5)?|1(?:\.0)?)\s*</score>", re.I)
BOXED_SCORE_PATTERN = re.compile(
    r"\\boxed\s*\{\s*(0(?:\.5)?|1(?:\.0)?)\s*\}", re.I
)
SELECTED_ID_PATTERN = re.compile(r"<selected_id>\s*R?(\d+)\s*</selected_id>", re.I)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def format_number(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    if number.is_integer():
        return str(int(number))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def format_percent(value: Any, digits: int = 1) -> str:
    number = as_float(value)
    return "-" if number is None else f"{100.0 * number:.{digits}f}%"


def discover_payload_paths(run_dir: Path) -> list[Path]:
    roots = [run_dir / "problems", run_dir / "artifacts" / "problems"]
    paths: list[Path] = []
    for root in roots:
        if root.is_dir():
            paths.extend(root.glob("*/rank_*.json"))
    return sorted({path.resolve() for path in paths})


def discover_result_paths(run_dir: Path) -> list[Path]:
    candidates = sorted(run_dir.glob("logs/rank*/results.jsonl"))
    candidates.extend(sorted(run_dir.glob("results.jsonl")))
    return sorted({path.resolve() for path in candidates})


def discover_call_paths(run_dir: Path) -> list[Path]:
    paths: set[Path] = set()
    for directory in run_dir.rglob("llm_calls"):
        if directory.is_dir():
            paths.update(path.resolve() for path in directory.rglob("*.txt"))
    return sorted(paths)


def load_selected_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for path in discover_result_paths(run_dir):
        for row in read_jsonl(path):
            problem_id = str(row.get("id", row.get("problem_id", "")))
            if not problem_id:
                continue
            current = selected.get(problem_id)
            if current is None or len(row.get("candidates") or []) > len(
                current.get("candidates") or []
            ):
                selected[problem_id] = row
    return selected


def parsed_score(output: dict[str, Any]) -> float | None:
    parsed = output.get("parsed") or {}
    score = as_float(parsed.get("score"))
    if score is None:
        score = as_float(output.get("score"))
    return score if score in {0.0, 0.5, 1.0} else None


def candidate_rounds(candidate: dict[str, Any], meta_n: int) -> list[dict[str, Any]]:
    verifier_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    meta_by_round: dict[tuple[int, int], list[float]] = defaultdict(list)

    for output in candidate.get("proof_verify_output") or []:
        round_idx = as_int(output.get("round_idx"))
        if round_idx is None:
            round_idx = 0
        verifier_by_round[round_idx].append(output)

    for output in candidate.get("proof_meta_verify_output") or []:
        round_idx = as_int(output.get("round_idx"))
        verifier_idx = as_int(output.get("verifier_index"))
        if round_idx is None:
            round_idx = 0
        if verifier_idx is None:
            continue
        score = parsed_score(output)
        if score is not None:
            meta_by_round[(round_idx, verifier_idx)].append(score)

    rounds: list[dict[str, Any]] = []
    for round_idx in sorted(verifier_by_round):
        summaries: list[dict[str, Any]] = []
        for ordinal, output in enumerate(verifier_by_round[round_idx]):
            verifier_idx = as_int(output.get("verifier_index"))
            if verifier_idx is None:
                verifier_idx = ordinal
            verifier_score = parsed_score(output)
            if verifier_score is None:
                continue
            meta_scores = meta_by_round.get((round_idx, verifier_idx), [])
            if meta_n <= 0:
                meta_factor = 1.0
                meta_source = "disabled"
            elif meta_scores:
                meta_factor = sum(meta_scores) / len(meta_scores)
                meta_source = "parsed"
            else:
                meta_factor = 0.6
                meta_source = "missing_default"
            summaries.append(
                {
                    "verifier_index": verifier_idx,
                    "verifier_score": verifier_score,
                    "meta_scores": meta_scores,
                    "meta_factor": meta_factor,
                    "meta_source": meta_source,
                    "weighted_score": verifier_score * meta_factor,
                }
            )
        rounds.append(
            {
                "round": round_idx,
                "score": mean(item["weighted_score"] for item in summaries),
                "verifier_scores": [item["verifier_score"] for item in summaries],
                "meta_scores": [score for item in summaries for score in item["meta_scores"]],
                "parsed_verifiers": len(summaries),
            }
        )
    return rounds


def candidate_row(
    problem_id: str,
    rank: int,
    candidate: dict[str, Any],
    *,
    meta_n: int,
    selector_max_chars: int,
) -> dict[str, Any]:
    rounds = candidate_rounds(candidate, meta_n)
    round_scores = [as_float(item["score"]) for item in rounds]
    proof = str(candidate.get("proof_solution") or "")
    selected_round = as_int(candidate.get("selected_verification_round"))
    selected_round_score = next(
        (
            item["score"]
            for item in rounds
            if as_int(item.get("round")) == selected_round
        ),
        None,
    )
    initial_score = round_scores[0] if round_scores else None
    last_score = round_scores[-1] if round_scores else None
    max_score = max((score for score in round_scores if score is not None), default=None)
    return {
        "problem_id": problem_id,
        "rank": rank,
        "attempt_idx": as_int(candidate.get("attempt_idx")),
        "prompt_family": candidate.get("prompt_family"),
        "generation_mode": candidate.get("generation_mode"),
        "proof_chars": len(proof),
        "selector_truncated": len(proof) > selector_max_chars,
        "self_score": as_float(candidate.get("self_score")),
        "final_score": as_float(candidate.get("final_score")),
        "final_status": candidate.get("final_status"),
        "strict_pass": bool(candidate.get("strict_pass")),
        "all_verifiers_passed": bool(candidate.get("all_verifiers_passed")),
        "meta_valid_count": as_int(candidate.get("meta_valid_count")) or 0,
        "meta_checked_count": as_int(candidate.get("meta_checked_count")) or 0,
        "selected_round": selected_round,
        "selected_round_score": as_float(selected_round_score),
        "round_count": len(rounds),
        "round_scores": [item["score"] for item in rounds],
        "initial_round_score": initial_score,
        "last_round_score": last_score,
        "max_round_score": max_score,
        "refinement_gain": (
            max_score - initial_score
            if max_score is not None and initial_score is not None
            else None
        ),
        "refinement_last_delta": (
            last_score - initial_score
            if last_score is not None and initial_score is not None
            else None
        ),
        "refine_outputs": len(candidate.get("proof_refine_output") or []),
        "refine_attempt_outputs": len(
            candidate.get("proof_refine_attempt_output") or []
        ),
        "rollback_from_round": as_int(candidate.get("rollback_from_round")),
        "budget_restarts": as_int(candidate.get("budget_restart_count")) or 0,
        "handoffs": len(candidate.get("proof_handoffs") or []),
        "refine_budget_restarts": as_int(
            candidate.get("refine_budget_restart_count")
        )
        or 0,
        "success": bool(candidate.get("success")),
        "selected": False,
        "internal_rank": None,
        "internal_tie_count": None,
        "external_grade": None,
    }


def parse_headers(text: str) -> dict[str, str]:
    prefix = text.split(INPUT_MARKER, 1)[0]
    headers: dict[str, str] = {}
    for line in prefix.splitlines():
        match = HEADER_PATTERN.match(line.strip())
        if match:
            headers[match.group(1)] = match.group(2).strip()
    return headers


def split_output(text: str) -> tuple[dict[str, Any], str]:
    if OUTPUT_MARKER not in text:
        return {}, ""
    body = text.split(OUTPUT_MARKER, 1)[1].lstrip("\r\n")
    lines = body.splitlines()
    metadata: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            break
        if ":" not in line:
            break
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "usage":
            try:
                metadata[key] = json.loads(value)
            except json.JSONDecodeError:
                metadata[key] = {}
        elif key == "success":
            metadata[key] = value.lower() == "true"
        elif key == "latency_s":
            metadata[key] = as_float(value)
        elif key in {"error", "finish_reason", "server_url"}:
            metadata[key] = None if value == "None" else value
        else:
            metadata[key] = value
        index += 1
    return metadata, "\n".join(lines[index:]).strip()


def last_score(text: str, stage: str) -> float | int | None:
    if stage == "proof_meta_verify":
        matches = BOXED_SCORE_PATTERN.findall(text)
    elif stage in {"proof_verify", "proof_generation", "proof_refine"}:
        matches = XML_SCORE_PATTERN.findall(text)
    elif stage == "selector":
        selected = SELECTED_ID_PATTERN.findall(text)
        return int(selected[-1]) if selected else None
    else:
        return None
    return float(matches[-1]) if matches else None


def parse_call_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    headers = parse_headers(text)
    metadata, output = split_output(text)
    usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
    detail = headers.get("detail", "")
    row: dict[str, Any] = {
        "path": str(path),
        "problem_id": path.parent.name,
        "stage": headers.get("stage", "unknown"),
        "detail": detail,
        "success": bool(metadata.get("success")),
        "error": metadata.get("error"),
        "finish_reason": metadata.get("finish_reason"),
        "prompt_tokens": as_int(usage.get("prompt_tokens"))
        or as_int(usage.get("estimated_prompt_tokens"))
        or as_int(headers.get("prompt_tokens"))
        or 0,
        "completion_tokens": as_int(usage.get("completion_tokens")) or 0,
        "total_tokens": as_int(usage.get("total_tokens")) or 0,
        "latency_s": as_float(metadata.get("latency_s")) or 0.0,
        "output_chars": len(output),
        "thinking_budget_applied": bool(usage.get("thinking_budget_applied")),
        "thinking_budget_action": usage.get("thinking_budget_action"),
        "thinking_budget_fallback_finalized": bool(
            usage.get("thinking_budget_fallback_finalized")
        ),
        "parsed_output_score": last_score(output, headers.get("stage", "unknown")),
    }
    for key, pattern in DETAIL_PATTERNS.items():
        match = pattern.search(detail)
        row[key] = int(match.group(1)) if match else None
    return row


def aggregate_calls(call_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in call_rows:
        by_stage[str(row["stage"])].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for stage, rows in sorted(by_stage.items()):
        finish_reasons = Counter(str(row.get("finish_reason")) for row in rows)
        summary[stage] = {
            "calls": len(rows),
            "successes": sum(bool(row.get("success")) for row in rows),
            "failures": sum(not bool(row.get("success")) for row in rows),
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
            "completion_tokens": sum(
                int(row.get("completion_tokens") or 0) for row in rows
            ),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
            "latency_s": sum(float(row.get("latency_s") or 0.0) for row in rows),
            "thinking_budget_applied": sum(
                bool(row.get("thinking_budget_applied")) for row in rows
            ),
            "finish_reasons": dict(sorted(finish_reasons.items())),
        }
    return summary


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator else None


def grader_scores(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = read_json(path)
    return {
        str(problem["problem_id"]): float(problem["score_out_of_7"])
        for problem in payload.get("problems") or []
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def build_report(analysis: dict[str, Any]) -> str:
    overview = analysis["overview"]
    lines = [
        "# Proof pipeline bottleneck report",
        "",
        f"Run: `{analysis['run_id']}`",
        "",
        "## Diagnosis",
        "",
        analysis["diagnosis"]["summary"],
        "",
    ]
    for finding in analysis["diagnosis"]["findings"]:
        lines.append(f"- {finding}")
    lines.extend(
        [
            "",
            "## Problem outcomes",
            "",
            "| Problem | Valid / assigned | Selected | Internal | Internal rank | Best internal | External / 7 | Proof chars | Selector clipped |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for problem in analysis["problems"]:
        lines.append(
            "| {problem_id} | {valid}/{assigned} | {selected_attempt} | {selected_score} | "
            "{selected_rank} | {best_score} | {external_grade} | {proof_chars} | {clipped} |".format(
                problem_id=problem["problem_id"],
                valid=problem["valid_candidates"],
                assigned=problem["assigned_candidates"],
                selected_attempt=problem.get("selected_attempt", "-"),
                selected_score=format_number(problem.get("selected_score")),
                selected_rank=problem.get("selected_internal_rank") or "-",
                best_score=format_number(problem.get("best_internal_score")),
                external_grade=format_number(problem.get("external_grade")),
                proof_chars=problem.get("selected_proof_chars") or "-",
                clipped="yes" if problem.get("selected_selector_truncated") else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Candidate health",
            "",
            f"- Pipeline completion: {overview['valid_candidates']}/{overview['assigned_candidates']} "
            f"({format_percent(overview['candidate_completion_rate'])}); failures: "
            f"{overview['failed_candidates']}.",
            f"- Candidates above selector threshold: {overview['selector_eligible_candidates']}/"
            f"{overview['valid_candidates']} ({format_percent(overview['selector_eligible_rate'])}).",
            f"- Refinement improved the best round score for {overview['refinement_improved_candidates']} "
            f"candidates; {overview['rollback_candidates']} candidates rolled back from a later round.",
            f"- Thinking-budget restarts occurred in {overview['restart_candidates']} candidates; "
            f"their mean internal score was {format_number(overview['restart_mean_score'])} versus "
            f"{format_number(overview['no_restart_mean_score'])} without restarts.",
            f"- {overview['selector_truncated_candidates']} candidate proofs exceeded the selector's "
            f"character window; {overview['selected_selector_truncated']} selected proofs were clipped.",
            "",
            "## Prompt families",
            "",
            "| Family | Assigned | Valid | Completion | Mean score | Eligible |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for family, stats in analysis["prompt_families"].items():
        lines.append(
            f"| {family} | {stats['assigned']} | {stats['valid']} | "
            f"{format_percent(stats['completion_rate'])} | "
            f"{format_number(stats['mean_final_score'])} | {stats['eligible']} |"
        )
    lines.extend(
        [
            "",
            "## LLM stages",
            "",
            "| Stage | Calls | Failed | Prompt tokens | Completion tokens | Budget interventions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage, stats in analysis["call_stages"].items():
        lines.append(
            f"| {stage} | {stats['calls']} | {stats['failures']} | "
            f"{stats['prompt_tokens']} | {stats['completion_tokens']} | "
            f"{stats['thinking_budget_applied']} |"
        )
    lines.extend(
        [
            "",
            "## Failure reasons",
            "",
        ]
    )
    for reason, count in analysis["failure_reasons"].items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The external grader evaluates only the selected proof, not every candidate. Candidate-level "
            "claims therefore use internal verifier scores, while causal claims about final quality use "
            "the selected proof's external grade. Raw prompts and completions remain in the source run.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_run(
    run_dir: Path,
    *,
    grader_summary: Path | None = None,
    selector_max_chars: int = 32_000,
    selector_threshold: float = 0.5,
) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    metadata = manifest.get("metadata") or {}
    run_id = str(manifest.get("run_id") or run_dir.name)
    meta_n = int(metadata.get("meta_n", 1))
    external = grader_scores(grader_summary)
    selected_results = load_selected_results(run_dir)

    candidate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    assigned_by_problem: Counter[str] = Counter()
    assigned_family: Counter[str] = Counter()
    payloads_by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in discover_payload_paths(run_dir):
        payload = read_json(path)
        problem_id = str(payload.get("problem_id"))
        rank = int(payload.get("rank", 0))
        assigned_attempts = payload.get("assigned_attempts") or []
        assigned_by_problem[problem_id] += len(assigned_attempts)
        payloads_by_problem[problem_id].append(payload)
        pipeline = payload.get("pipeline_result") or {}
        candidates = pipeline.get("candidates") or []
        candidate_by_attempt = {
            as_int(candidate.get("attempt_idx")): candidate for candidate in candidates
        }
        deepseek_count = int(metadata.get("deepseek_math_v2_candidate_count", 0))
        for attempt in assigned_attempts:
            family = "deepseek_math_v2" if int(attempt) < deepseek_count else "opd"
            assigned_family[family] += 1
            if int(attempt) not in candidate_by_attempt:
                continue
        for candidate in candidates:
            candidate_rows.append(
                candidate_row(
                    problem_id,
                    rank,
                    candidate,
                    meta_n=meta_n,
                    selector_max_chars=selector_max_chars,
                )
            )
        for failure in pipeline.get("failed_attempts") or []:
            failure_rows.append(
                {
                    "problem_id": problem_id,
                    "rank": rank,
                    "attempt_idx": as_int(failure.get("attempt_idx")),
                    "error": str(failure.get("error") or "unknown"),
                }
            )

    problem_rows: list[dict[str, Any]] = []
    for problem_id in sorted(assigned_by_problem, key=lambda value: int(value)):
        rows = [row for row in candidate_rows if row["problem_id"] == problem_id]
        result = selected_results.get(problem_id, {})
        selected_attempt = as_int(result.get("selected_pipeline"))
        if selected_attempt is None:
            selected_attempt = as_int(result.get("selected_pipeline_idx"))
        selected = next(
            (row for row in rows if row["attempt_idx"] == selected_attempt), None
        )
        scores = [row["final_score"] for row in rows if row["final_score"] is not None]
        best_score = max(scores, default=None)
        if selected is not None:
            selected_score = selected["final_score"]
            better = sum(
                row["final_score"] is not None
                and selected_score is not None
                and row["final_score"] > selected_score
                for row in rows
            )
            ties = sum(row["final_score"] == selected_score for row in rows)
            selected["selected"] = True
            selected["internal_rank"] = better + 1
            selected["internal_tie_count"] = ties
            selected["external_grade"] = external.get(problem_id)
        problem_rows.append(
            {
                "problem_id": problem_id,
                "assigned_candidates": assigned_by_problem[problem_id],
                "valid_candidates": len(rows),
                "failed_candidates": assigned_by_problem[problem_id] - len(rows),
                "selected_attempt": selected_attempt,
                "selected_score": selected.get("final_score") if selected else None,
                "selected_status": selected.get("final_status") if selected else None,
                "selected_internal_rank": selected.get("internal_rank") if selected else None,
                "selected_internal_ties": selected.get("internal_tie_count") if selected else None,
                "best_internal_score": best_score,
                "best_internal_attempts": [
                    row["attempt_idx"] for row in rows if row["final_score"] == best_score
                ],
                "external_grade": external.get(problem_id),
                "selected_proof_chars": selected.get("proof_chars") if selected else None,
                "selected_selector_truncated": (
                    selected.get("selector_truncated") if selected else None
                ),
                "selected_round": selected.get("selected_round") if selected else None,
                "selected_round_scores": selected.get("round_scores") if selected else [],
                "selected_budget_restarts": (
                    selected.get("budget_restarts") if selected else None
                ),
                "score_distribution": dict(
                    sorted(Counter(str(row["final_score"]) for row in rows).items())
                ),
                "status_distribution": dict(
                    sorted(Counter(str(row["final_status"]) for row in rows).items())
                ),
            }
        )

    call_rows = [parse_call_file(path) for path in discover_call_paths(run_dir)]
    call_stages = aggregate_calls(call_rows)
    valid_count = len(candidate_rows)
    assigned_count = sum(assigned_by_problem.values())
    eligible = [
        row for row in candidate_rows if (row["final_score"] or 0.0) > selector_threshold
    ]
    restart_rows = [row for row in candidate_rows if row["budget_restarts"] > 0]
    no_restart_rows = [row for row in candidate_rows if row["budget_restarts"] == 0]

    family_stats: dict[str, dict[str, Any]] = {}
    all_families = sorted(set(assigned_family) | {str(row["prompt_family"]) for row in candidate_rows})
    for family in all_families:
        rows = [row for row in candidate_rows if row["prompt_family"] == family]
        assigned = assigned_family[family]
        family_stats[family] = {
            "assigned": assigned,
            "valid": len(rows),
            "completion_rate": len(rows) / assigned if assigned else None,
            "mean_final_score": mean(row["final_score"] for row in rows),
            "eligible": sum((row["final_score"] or 0.0) > selector_threshold for row in rows),
        }

    paired = [
        (float(problem["selected_score"]), float(problem["external_grade"]) / 7.0)
        for problem in problem_rows
        if problem["selected_score"] is not None and problem["external_grade"] is not None
    ]
    high_conf_external_failures = [
        problem
        for problem in problem_rows
        if (problem["selected_score"] or 0.0) > selector_threshold
        and problem["external_grade"] is not None
        and problem["external_grade"] <= 2.0
    ]
    selector_misses = [
        problem
        for problem in problem_rows
        if problem["selected_internal_rank"] is not None
        and problem["selected_internal_rank"] > 1
    ]
    low_ceiling = [
        problem
        for problem in problem_rows
        if (problem["best_internal_score"] or 0.0) <= selector_threshold
        and problem["external_grade"] is not None
        and problem["external_grade"] <= 2.0
    ]

    findings = [
        f"{assigned_count - valid_count}/{assigned_count} candidate pipelines failed before selection "
        f"({format_percent((assigned_count - valid_count) / assigned_count if assigned_count else None)}).",
        f"The selector chose a non-top internal score on {len(selector_misses)}/{len(problem_rows)} problems.",
        f"{len(high_conf_external_failures)} selected proofs passed the internal selector threshold but "
        "received at most 2/7 from the external grader.",
        f"{len(low_ceiling)} low-scoring problems had no candidate above the internal selector threshold.",
    ]
    if high_conf_external_failures:
        summary = (
            "Primary bottleneck: verifier/selector calibration. The pipeline assigned passing internal "
            "scores to selected proofs that an independent rubric grader found fatally incomplete. "
            "Generation quality is a secondary bottleneck because many candidates also failed or never "
            "crossed the selector threshold."
        )
    elif low_ceiling:
        summary = (
            "Primary bottleneck: proof generation. The low-scoring problems did not produce an internally "
            "credible candidate, so selector changes alone cannot recover them."
        )
    elif selector_misses:
        summary = (
            "Primary bottleneck: final selection. Higher internally scored candidates existed but were not "
            "selected."
        )
    else:
        summary = (
            "The available selected-proof grades do not isolate one dominant stage; inspect the problem and "
            "round tables before changing the pipeline."
        )

    analysis = {
        "schema_version": 1,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": metadata,
        "overview": {
            "problems": len(problem_rows),
            "assigned_candidates": assigned_count,
            "valid_candidates": valid_count,
            "failed_candidates": assigned_count - valid_count,
            "candidate_completion_rate": valid_count / assigned_count if assigned_count else None,
            "selector_eligible_candidates": len(eligible),
            "selector_eligible_rate": len(eligible) / valid_count if valid_count else None,
            "refinement_improved_candidates": sum(
                (row["refinement_gain"] or 0.0) > 1e-9 for row in candidate_rows
            ),
            "rollback_candidates": sum(
                row["rollback_from_round"] is not None for row in candidate_rows
            ),
            "restart_candidates": len(restart_rows),
            "restart_mean_score": mean(row["final_score"] for row in restart_rows),
            "no_restart_mean_score": mean(row["final_score"] for row in no_restart_rows),
            "selector_truncated_candidates": sum(
                bool(row["selector_truncated"]) for row in candidate_rows
            ),
            "selected_selector_truncated": sum(
                bool(row["selected"] and row["selector_truncated"])
                for row in candidate_rows
            ),
            "selected_internal_external_pearson": pearson(
                [item[0] for item in paired], [item[1] for item in paired]
            ),
        },
        "diagnosis": {
            "summary": summary,
            "findings": findings,
            "high_confidence_external_failures": [
                problem["problem_id"] for problem in high_conf_external_failures
            ],
            "selector_misses": [problem["problem_id"] for problem in selector_misses],
            "low_internal_ceiling": [problem["problem_id"] for problem in low_ceiling],
        },
        "problems": problem_rows,
        "prompt_families": family_stats,
        "failure_reasons": dict(
            sorted(Counter(row["error"] for row in failure_rows).items())
        ),
        "call_stages": call_stages,
        "candidate_count": len(candidate_rows),
        "call_count": len(call_rows),
        "candidates": candidate_rows,
        "failures": failure_rows,
        "calls": call_rows,
    }
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--grader-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector-max-chars", type=int, default=32_000)
    parser.add_argument("--selector-threshold", type=float, default=0.5)
    args = parser.parse_args()

    analysis = analyze_run(
        args.run_dir.resolve(),
        grader_summary=args.grader_summary.resolve() if args.grader_summary else None,
        selector_max_chars=args.selector_max_chars,
        selector_threshold=args.selector_threshold,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in analysis.items() if key not in {"candidates", "calls"}}
    (output_dir / "analysis.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(build_report(analysis), encoding="utf-8")
    write_csv(output_dir / "candidates.csv", analysis["candidates"])
    write_csv(output_dir / "calls.csv", analysis["calls"])
    write_csv(output_dir / "failures.csv", analysis["failures"])
    print(output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
