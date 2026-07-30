"""Merge complete sharded submission CSVs in the original input order."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from run_submission import OUTPUT_COLUMNS, load_test_csv


def load_shard(path: Path, valid_ids: set[str]) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != OUTPUT_COLUMNS:
            raise ValueError(f"{path} must contain exactly id,proof")
        proofs: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{row_number} has extra fields")
            row_id = row["id"]
            proof = row["proof"]
            if row_id not in valid_ids:
                raise ValueError(f"{path}:{row_number} has unknown id {row_id!r}")
            if row_id in proofs:
                raise ValueError(f"{path} contains duplicate id {row_id!r}")
            if proof is None or not proof.strip():
                raise ValueError(f"{path}:{row_number} has an empty proof")
            proofs[row_id] = proof
    return proofs


def merge_submissions(
    input_path: Path,
    shard_paths: list[Path],
    output_path: Path,
) -> None:
    rows = load_test_csv(input_path)
    valid_ids = {row.id for row in rows}
    merged: dict[str, str] = {}
    for shard_path in shard_paths:
        shard = load_shard(shard_path, valid_ids)
        duplicates = sorted(merged.keys() & shard.keys())
        if duplicates:
            raise ValueError(
                f"submission shards overlap on id(s): {duplicates}"
            )
        merged.update(shard)

    missing = [row.id for row in rows if row.id not in merged]
    if missing:
        raise ValueError(f"submission shards are incomplete; missing id(s): {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row.id, "proof": merged[row.id]})
    os.replace(temporary, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    merge_submissions(args.input, args.shards, args.output)
    print(
        f"[merge] merged {len(args.shards)} shard(s) -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
