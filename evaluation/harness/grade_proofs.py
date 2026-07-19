"""Grade one selected proof per problem with arithmetic-mean aggregation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Annotated

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

from eval_config import load_config
from grader import parse_score

REPO = Path(__file__).resolve().parents[2]
GRADER_SYSTEM_PROMPT = REPO / "evaluation" / "prompts" / "grader_system.md"
GRADER_USER_PROMPT = REPO / "evaluation" / "prompts" / "grader_user.md"


class GraderOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: Annotated[
        list[Annotated[str, Field(min_length=1)]], Field(min_length=1)
    ]
    grade: Annotated[int, Field(ge=0, le=7)]
    reasoning: Annotated[str, Field(min_length=1)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_grading_scheme(items: object) -> str:
    if not isinstance(items, (list, tuple)):
        try:
            items = list(items)
        except TypeError as error:
            raise ValueError("grading_scheme must be a sequence") from error
    rendered = []
    total_points = 0
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"grading_scheme item {index} must be a mapping")
        missing = {"title", "points", "desc"} - set(item)
        if missing:
            raise ValueError(
                f"grading_scheme item {index} missing fields: {sorted(missing)}"
            )
        points = int(item["points"])
        if points <= 0:
            raise ValueError(f"grading_scheme item {index} has invalid points")
        total_points += points
        rendered.append(
            f"{index}. [{points} pts] {item['title']}: {item['desc']}"
        )
    if total_points != 7:
        raise ValueError(f"grading_scheme points sum to {total_points}, expected 7")
    return "\n".join(rendered)


def load_requested_rows(path: Path) -> list[dict]:
    """Load normalized JSONL rubrics or MathArena parquet rows."""
    if path.suffix == ".parquet":
        source_rows = pd.read_parquet(path).to_dict(orient="records")
        rows = [
            {
                "Problem ID": str(source["problem_idx"]),
                "Problem": source["problem"],
                "Grading scheme": render_grading_scheme(
                    source["grading_scheme"]
                ),
            }
            for source in source_rows
        ]
    elif path.suffix in {".jsonl", ".json"}:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    else:
        raise ValueError(
            f"unsupported rubric file extension {path.suffix!r}; "
            "expected .parquet or .jsonl"
        )

    required = {"Problem ID", "Problem", "Grading scheme"}
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"rubric row {index} missing fields: {sorted(missing)}")
        problem_id = str(row["Problem ID"])
        if problem_id in seen_ids:
            raise ValueError(f"duplicate rubric problem ID {problem_id!r}")
        seen_ids.add(problem_id)
        row["Problem ID"] = problem_id
        for field in ("Problem", "Grading scheme"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"rubric row {index} has empty {field}")
    if not rows:
        raise ValueError("rubric file contains no rows")
    return rows


def build_grader_request(
    row: dict,
    proof: str,
    model: str,
) -> tuple[list[dict[str, str]], str, str]:
    system_prompt = GRADER_SYSTEM_PROMPT.read_text().format(
        grading_scheme=row["Grading scheme"]
    )
    user_prompt = GRADER_USER_PROMPT.read_text().format(
        problem_statement=row["Problem"],
        student_answer=proof,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    messages_hash = hashlib.sha256(
        json.dumps(
            messages, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    prompt_cache_key = hashlib.sha256(
        f"{model}\0{messages_hash}".encode()
    ).hexdigest()
    return messages, messages_hash, prompt_cache_key


def load_selected_proofs(search_dir: Path, expected_ids: list[str]) -> dict[str, dict]:
    records_path = search_dir / "records.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text().splitlines()
        if line.strip()
    ]
    by_id: dict[str, dict] = {}
    for record in records:
        problem_id = record["problem_id"]
        if problem_id in by_id:
            raise RuntimeError(f"duplicate selected proof for {problem_id}")
        if not record["final_proof"].strip():
            raise RuntimeError(f"empty selected proof for {problem_id}")
        by_id[problem_id] = record
    if list(by_id) != expected_ids:
        raise RuntimeError(
            f"selected-proof IDs/order differ: expected={expected_ids}, actual={list(by_id)}"
        )
    return by_id


def arithmetic_mean_score(scores: list[int], attempts_per_proof: int) -> float:
    if len(scores) != attempts_per_proof:
        raise RuntimeError(
            f"expected {attempts_per_proof} grader scores, received {len(scores)}"
        )
    return mean(scores)


def is_retryable_error(error: Exception) -> bool:
    """Retry transient transport/server failures and structured-output failures."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        return True
    return status_code in {408, 409, 429} or status_code >= 500


def aggregate_grades(
    records: list[dict],
    problem_ids: list[str],
    attempts_per_proof: int,
) -> dict:
    grouped: dict[str, list[dict]] = {problem_id: [] for problem_id in problem_ids}
    for record in records:
        grouped[record["problem_id"]].append(record)
    problems = []
    for problem_id in problem_ids:
        attempts = sorted(grouped[problem_id], key=lambda item: item["attempt"])
        if [item["attempt"] for item in attempts] != list(range(attempts_per_proof)):
            raise RuntimeError(f"incomplete grader attempt sequence for {problem_id}")
        if any(item["error"] is not None for item in attempts):
            raise RuntimeError(f"failed grader attempt persisted for {problem_id}")
        scores = [item["score"] for item in attempts]
        score = arithmetic_mean_score(scores, attempts_per_proof)
        score_counts = Counter(scores)
        problems.append(
            {
                "problem_id": problem_id,
                "attempts": attempts_per_proof,
                "attempt_scores": scores,
                "score_distribution": {
                    str(value): score_counts.get(value, 0) for value in range(8)
                },
                "score_out_of_7": score,
                "score_percent": score / 7 * 100,
            }
        )
    overall = mean(item["score_out_of_7"] for item in problems)
    return {
        "schema_version": 1,
        "aggregation": "arithmetic_mean",
        "attempts_per_proof": attempts_per_proof,
        "problems": problems,
        "overall_score_out_of_7": overall,
        "overall_score_percent": overall / 7 * 100,
    }


async def grade_final_proofs(
    config_path: Path,
    ids_file: Path,
    search_dir: Path,
    output_dir: Path,
) -> dict:
    config, rows, selected = validate_grading_inputs(
        config_path,
        ids_file,
        search_dir,
    )
    grader = config["grader"]
    api_key = os.environ.get(grader["api_key_env"])
    if not api_key:
        raise RuntimeError(f"empty {grader['api_key_env']}")

    problem_ids = [row["Problem ID"] for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    failures_path = output_dir / "failures.jsonl"

    existing: dict[tuple[str, int], dict] = {}
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["problem_id"], record["attempt"])
            if key in existing:
                raise RuntimeError(f"duplicate persisted grader attempt: {key}")
            if record["error"] is not None:
                raise RuntimeError(f"successful grader record contains an error: {key}")
            existing[key] = record

    jobs: list[tuple[dict, int, list[dict[str, str]], str, str]] = []
    for row in rows:
        problem_id = row["Problem ID"]
        messages, messages_hash, prompt_cache_key = build_grader_request(
            row,
            selected[problem_id]["final_proof"],
            grader["model"],
        )
        for attempt in range(grader["attempts_per_proof"]):
            if (problem_id, attempt) not in existing:
                jobs.append(
                    (row, attempt, messages, messages_hash, prompt_cache_key)
                )

    client = AsyncOpenAI(
        base_url=grader["base_url"],
        api_key=api_key,
        max_retries=0,
        timeout=3600.0,
    )
    semaphore = asyncio.Semaphore(grader["concurrency"])
    write_lock = asyncio.Lock()
    progress = tqdm(
        total=len(rows) * grader["attempts_per_proof"],
        initial=len(existing),
        desc="Grading proofs",
        unit="call",
        dynamic_ncols=True,
    )

    async def append_success(record: dict) -> None:
        async with write_lock:
            with records_path.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
            existing[(record["problem_id"], record["attempt"])] = record
            progress.update(1)

    async def append_failure(record: dict) -> None:
        async with write_lock:
            with failures_path.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()

    async def work(
        row: dict,
        attempt: int,
        messages: list[dict[str, str]],
        messages_hash: str,
        prompt_cache_key: str,
    ) -> None:
        problem_id = row["Problem ID"]
        started = time.monotonic()
        response = None
        content = None
        parsed = None
        usage = None
        request_attempt = 0
        try:
            for request_attempt in range(grader["request_retries"] + 1):
                try:
                    request_kwargs = {
                        "model": grader["model"],
                        "input": messages,
                        "reasoning": {"effort": grader["reasoning"]},
                        "max_output_tokens": grader["max_completion_tokens"],
                        "prompt_cache_key": prompt_cache_key,
                        "text_format": GraderOutput,
                    }
                    if grader["prompt_cache_options_enabled"]:
                        request_kwargs["prompt_cache_options"] = {
                            "mode": grader["prompt_cache_mode"],
                            "ttl": grader["prompt_cache_ttl"],
                        }
                    async with semaphore:
                        response = await client.responses.parse(**request_kwargs)
                    if response.output_parsed is None:
                        raise RuntimeError(
                            "grader returned no parsed structured output"
                        )
                    content = response.output_text
                    parsed = parse_score(content)
                    usage = response.usage.model_dump() if response.usage else None
                    break
                except Exception as error:
                    if (
                        request_attempt >= grader["request_retries"]
                        or not is_retryable_error(error)
                    ):
                        raise
                    delay = min(
                        grader["retry_backoff_max_seconds"],
                        grader["retry_backoff_seconds"] * (2**request_attempt),
                    )
                    tqdm.write(
                        f"Retrying problem {problem_id} attempt {attempt} after "
                        f"{type(error).__name__} in {delay}s "
                        f"({request_attempt + 1}/{grader['request_retries']})"
                    )
                    await asyncio.sleep(delay)
            if response is None:
                raise RuntimeError("grader request completed without a response")
            if content is None or parsed is None:
                raise RuntimeError("grader response validation did not complete")
            record = {
                "problem_id": problem_id,
                "attempt": attempt,
                "messages_sha256": messages_hash,
                "prompt_cache_key": prompt_cache_key,
                "messages": messages,
                "grader_model": grader["model"],
                "grader_api": "responses",
                "grader_reasoning": grader["reasoning"],
                "grader_content": content,
                "response_status": response.status,
                "response": response.model_dump(mode="json"),
                "usage": usage,
                "latency_s": round(time.monotonic() - started, 3),
                "request_attempts": request_attempt + 1,
                "score": parsed["grade"],
                "findings": parsed["findings"],
                "reasoning": parsed["reasoning"],
                "error": None,
            }
        except Exception as error:
            record = {
                "problem_id": problem_id,
                "attempt": attempt,
                "messages_sha256": messages_hash,
                "prompt_cache_key": prompt_cache_key,
                "messages": messages,
                "grader_model": grader["model"],
                "grader_api": "responses",
                "grader_reasoning": grader["reasoning"],
                "latency_s": round(time.monotonic() - started, 3),
                "request_attempts": request_attempt + 1,
                "error": repr(error),
            }
            await append_failure(record)
            raise
        await append_success(record)

    try:
        warm_jobs = []
        remaining_jobs = []
        warmed_problem_ids = set()
        for job in jobs:
            problem_id = job[0]["Problem ID"]
            if problem_id not in warmed_problem_ids:
                warm_jobs.append(job)
                warmed_problem_ids.add(problem_id)
            else:
                remaining_jobs.append(job)
        for job in warm_jobs:
            await work(*job)
        results = await asyncio.gather(
            *[work(*job) for job in remaining_jobs],
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(
                f"{len(failures)} grader attempt(s) failed; rerun to retry only "
                "failed or missing attempts"
            ) from failures[0]
    finally:
        progress.close()
        await client.close()

    records = list(existing.values())
    cache_details = [
        ((record.get("usage") or {}).get("input_tokens_details") or {})
        for record in records
    ]
    cache_input_tokens = sum(
        (record.get("usage") or {}).get("input_tokens", 0) for record in records
    )

    summary = aggregate_grades(
        records,
        problem_ids,
        grader["attempts_per_proof"],
    )
    summary.update(
        {
            "grader_model": grader["model"],
            "grader_api": "responses",
            "grader_reasoning": grader["reasoning"],
            "grader_prompt_sha256": {
                "system": grader["system_prompt_sha256"],
                "user": grader["user_prompt_sha256"],
            },
            "prompt_cache": {
                "mode": grader["prompt_cache_mode"],
                "ttl": grader["prompt_cache_ttl"],
                "options_sent": grader["prompt_cache_options_enabled"],
                "warmup_attempts_per_problem": 1,
                "input_tokens": cache_input_tokens,
                "cache_write_tokens": sum(
                    detail.get("cache_write_tokens", 0) for detail in cache_details
                ),
                "cached_tokens": sum(
                    detail.get("cached_tokens", 0) for detail in cache_details
                ),
                "cache_hit_attempts": sum(
                    1 for detail in cache_details if detail.get("cached_tokens", 0) > 0
                ),
            },
            "request_retries": {
                "configured_per_attempt": grader["request_retries"],
                "calls_with_retries": sum(
                    1 for record in records if record.get("request_attempts", 1) > 1
                ),
                "total_retries": sum(
                    max(0, record.get("request_attempts", 1) - 1)
                    for record in records
                ),
            },
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def validate_grading_inputs(
    config_path: Path,
    rubrics_file: Path,
    search_dir: Path,
) -> tuple[dict, list[dict], dict[str, dict]]:
    """Validate config, prompt hashes, rubrics, and selected proof alignment."""
    config = load_config(config_path)
    if "grader" not in config:
        raise ValueError("evaluation config is missing the grader section")
    grader = config["grader"]
    if grader["zero_veto"]:
        raise ValueError("arithmetic-mean grading requires grader.zero_veto=false")
    if sha256(GRADER_SYSTEM_PROMPT) != grader["system_prompt_sha256"]:
        raise RuntimeError("grader system prompt hash differs from the YAML")
    if sha256(GRADER_USER_PROMPT) != grader["user_prompt_sha256"]:
        raise RuntimeError("grader user prompt hash differs from the YAML")

    rows = load_requested_rows(rubrics_file)
    problem_ids = [row["Problem ID"] for row in rows]
    selected = load_selected_proofs(search_dir, problem_ids)
    return config, rows, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--rubrics-file",
        "--ids-file",
        dest="rubrics_file",
        required=True,
        type=Path,
    )
    parser.add_argument("--search-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        config, rows, selected = validate_grading_inputs(
            args.config,
            args.rubrics_file,
            args.search_dir,
        )
        grader = config["grader"]
        print(
            json.dumps(
                {
                    "problem_ids": [row["Problem ID"] for row in rows],
                    "proof_characters": {
                        problem_id: len(record["final_proof"])
                        for problem_id, record in selected.items()
                    },
                    "model": grader["model"],
                    "reasoning": grader["reasoning"],
                    "attempts_per_proof": grader["attempts_per_proof"],
                    "concurrency": grader["concurrency"],
                    "request_retries": grader["request_retries"],
                    "prompt_cache_options_enabled": grader[
                        "prompt_cache_options_enabled"
                    ],
                    "aggregation": "arithmetic_mean",
                },
                indent=2,
            )
        )
        return
    asyncio.run(
        grade_final_proofs(
            args.config,
            args.rubrics_file,
            args.search_dir,
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
