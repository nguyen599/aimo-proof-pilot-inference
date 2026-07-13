"""Run the YAML-configured proof-pool search for an explicit dataset manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from async_client import AsyncChatClient
from eval_config import active_model, load_config
from proof_search import ProblemSearch

REPO = Path(__file__).resolve().parents[2]
DATASETS = {
    "imo_2025": REPO / "evaluation/data/imo_2025.parquet",
    "aime_2026": REPO / "evaluation/data/aime_2026.parquet",
}
MANIFEST_KEYS = {"dataset", "problem_ids"}


def render_grading_scheme(items) -> str:
    return "\n".join(
        f"{index}. [{int(item['points'])} pts] {item['title']}: {item['desc']}"
        for index, item in enumerate(items, start=1)
    )


def load_problem_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ValueError(
            "problem manifest must contain exactly dataset and problem_ids"
        )
    dataset = manifest["dataset"]
    if dataset not in DATASETS:
        raise ValueError(f"unsupported problem dataset: {dataset!r}")
    problem_ids = manifest["problem_ids"]
    if not isinstance(problem_ids, list) or not problem_ids:
        raise ValueError("problem_ids must be a non-empty JSON array")
    if not all(isinstance(problem_id, str) for problem_id in problem_ids):
        raise ValueError("problem manifest IDs must be strings")
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("problem manifest contains duplicate IDs")
    return manifest


def dataset_path(manifest_path: Path) -> Path:
    return DATASETS[load_problem_manifest(manifest_path)["dataset"]]


def _imo_2025_rows(data: pd.DataFrame) -> list[dict]:
    expected = {"problem_idx", "problem", "points", "grading_scheme"}
    if set(data.columns) != expected:
        raise ValueError("MathArena/imo_2025 columns differ from the pinned schema")
    rows = []
    for source in data.to_dict(orient="records"):
        guidelines = render_grading_scheme(source["grading_scheme"])
        rows.append(
            {
                "Problem ID": str(source["problem_idx"]),
                "Problem": source["problem"],
                "Grading scheme": guidelines,
                "Competition": "IMO",
                "Year": 2025,
                "Category": "IMO",
                "Level": "2025",
                "Points": int(source["points"]),
                "Answer": None,
            }
        )
    return rows


def _aime_2026_rows(data: pd.DataFrame) -> list[dict]:
    expected = {"problem_idx", "answer", "problem"}
    if set(data.columns) != expected:
        raise ValueError("MathArena/aime_2026 columns differ from the pinned schema")
    rows = []
    for source in data.to_dict(orient="records"):
        answer = int(source["answer"])
        guidelines = (
            f"Award 7 points for a complete and logically valid derivation of the "
            f"official integer answer {answer}. For incomplete work, assign the "
            "integer 0-6 that matches the general rubric and the amount of correct, "
            "relevant progress. A claimed answer without a valid derivation is not "
            "a complete solution."
        )
        rows.append(
            {
                "Problem ID": str(source["problem_idx"]),
                "Problem": source["problem"],
                "Grading scheme": guidelines,
                "Competition": "AIME I",
                "Year": 2026,
                "Category": "AIME",
                "Level": "2026 I",
                "Points": 7,
                "Answer": answer,
            }
        )
    return rows


def load_requested_rows(manifest_path: Path) -> list[dict]:
    manifest = load_problem_manifest(manifest_path)
    data = pd.read_parquet(DATASETS[manifest["dataset"]])
    row_loaders = {
        "imo_2025": _imo_2025_rows,
        "aime_2026": _aime_2026_rows,
    }
    rows = row_loaders[manifest["dataset"]](data)
    by_id = {row["Problem ID"]: row for row in rows}
    missing = [
        problem_id
        for problem_id in manifest["problem_ids"]
        if problem_id not in by_id
    ]
    if missing:
        raise ValueError(
            f"unknown {manifest['dataset']} problem IDs: {missing}"
        )
    return [by_id[problem_id] for problem_id in manifest["problem_ids"]]


async def run_search(config_path: Path, ids_file: Path, output_dir: Path) -> list[dict]:
    config = load_config(config_path)
    model = active_model(config)
    server = config["server"]
    base_url = f"http://{server['host']}:{server['port']}/v1"
    rows = load_requested_rows(ids_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    completed: dict[str, dict] = {}
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = record["problem_id"]
            if problem_id in completed:
                raise RuntimeError(f"duplicate completed problem: {problem_id}")
            completed[problem_id] = record

    client = AsyncChatClient(
        base_url,
        str(model.target),
        api_key="EMPTY",
        max_connections=config["search"]["concurrency"] + 8,
        timeout=float(config["search"]["request_timeout_seconds"]),
    )
    semaphore = asyncio.Semaphore(config["search"]["concurrency"])
    results: list[dict] = []
    try:
        with records_path.open("a") as output:
            for row in rows:
                problem_id = row["Problem ID"]
                if problem_id in completed:
                    results.append(completed[problem_id])
                    continue
                search = ProblemSearch(
                    problem_id=problem_id,
                    problem=row["Problem"],
                    output_dir=output_dir / "problems" / problem_id,
                    client=client,
                    semaphore=semaphore,
                    config=config["search"],
                )
                final = await search.solve()
                record = {
                    "problem_id": problem_id,
                    "competition": row["Competition"],
                    "year": row["Year"],
                    "points": row["Points"],
                    "answer": row["Answer"],
                    **final,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                results.append(record)
                print(
                    f"[proof-search] {row['Competition'].lower().replace(' ', '-')}-"
                    f"{row['Year']}-{problem_id} "
                    f"rounds={record['rounds_completed']} "
                    f"pool={record['proofs_in_pool']} calls={record['calls_completed']} "
                    f"score={record['mean_verifier_score']:.5f} "
                    f"votes={record['valid_verification_count']}",
                    flush=True,
                )
    finally:
        await client.aclose()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(run_search(args.config, args.ids_file, args.output_dir))


if __name__ == "__main__":
    main()
