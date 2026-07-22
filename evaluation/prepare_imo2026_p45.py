#!/usr/bin/env python3
"""Prepare answer-free IMO 2026 P4/P5 inputs for pipeline comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


TARGET_PROBLEM_IDS = ("4", "5")


def load_problems(path: Path) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        problem_id = str(row.get("problem_idx", ""))
        if problem_id not in TARGET_PROBLEM_IDS:
            continue
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"line {line_number} has no nonempty problem text")
        selected.append({"problem_idx": problem_id, "problem": problem})

    actual = tuple(row["problem_idx"] for row in selected)
    if actual != TARGET_PROBLEM_IDS:
        raise ValueError(
            "IMO 2026 input must contain exactly problems 4 and 5 in order; "
            f"found {actual}"
        )
    return selected


def _atomic_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = _atomic_path(path)
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = _atomic_path(path)
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("id", "problem"))
        writer.writeheader()
        writer.writerows(
            {"id": row["problem_idx"], "problem": row["problem"]}
            for row in rows
        )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--jsonl-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()
    if args.jsonl_output is None and args.csv_output is None:
        parser.error("at least one output path is required")
    return args


def main() -> None:
    args = parse_args()
    rows = load_problems(args.input)
    if args.jsonl_output is not None:
        write_jsonl(args.jsonl_output, rows)
    if args.csv_output is not None:
        write_csv(args.csv_output, rows)


if __name__ == "__main__":
    main()
