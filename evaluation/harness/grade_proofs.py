"""Grade one selected proof per problem with the YAML-configured zero-veto policy."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Annotated

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from eval_config import load_config
from grader import parse_score
from run_proof_search import load_requested_rows

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
    prompt_cache_key = (
        f"final-grader:{model}:{row['Problem ID']}:{messages_hash}"
    )
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


def zero_veto_score(scores: list[int], attempts_per_proof: int) -> float:
    if len(scores) != attempts_per_proof:
        raise RuntimeError(
            f"expected {attempts_per_proof} grader scores, received {len(scores)}"
        )
    return 0.0 if 0 in scores else mean(scores)


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
        score = zero_veto_score(scores, attempts_per_proof)
        problems.append(
            {
                "problem_id": problem_id,
                "attempts": attempts_per_proof,
                "attempt_scores": scores,
                "zero_veto_triggered": 0 in scores,
                "score_out_of_7": score,
                "score_percent": score / 7 * 100,
            }
        )
    overall = mean(item["score_out_of_7"] for item in problems)
    return {
        "schema_version": 1,
        "aggregation": "zero_veto_else_arithmetic_mean",
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
    config = load_config(config_path)
    grader = config["grader"]
    if sha256(GRADER_SYSTEM_PROMPT) != grader["system_prompt_sha256"]:
        raise RuntimeError("grader system prompt hash differs from the YAML")
    if sha256(GRADER_USER_PROMPT) != grader["user_prompt_sha256"]:
        raise RuntimeError("grader user prompt hash differs from the YAML")
    api_key = os.environ.get(grader["api_key_env"])
    if not api_key:
        raise RuntimeError(f"empty {grader['api_key_env']}")

    rows = load_requested_rows(ids_file)
    problem_ids = [row["Problem ID"] for row in rows]
    selected = load_selected_proofs(search_dir, problem_ids)
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

    async def append_success(record: dict) -> None:
        async with write_lock:
            with records_path.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
            existing[(record["problem_id"], record["attempt"])] = record

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
        try:
            async with semaphore:
                response = await client.responses.parse(
                    model=grader["model"],
                    input=messages,
                    reasoning={"effort": grader["reasoning"]},
                    max_output_tokens=grader["max_completion_tokens"],
                    prompt_cache_key=prompt_cache_key,
                    prompt_cache_options={
                        "mode": grader["prompt_cache_mode"],
                        "ttl": grader["prompt_cache_ttl"],
                    },
                    text_format=GraderOutput,
                )
            if response.output_parsed is None:
                raise RuntimeError("grader returned no parsed structured output")
            content = response.output_text
            parsed = parse_score(content)
            usage = response.usage.model_dump() if response.usage else None
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
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--search-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(
        grade_final_proofs(
            args.config,
            args.ids_file,
            args.search_dir,
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
