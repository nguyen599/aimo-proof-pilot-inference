"""Verbatim ycchen Math-3R prompts, renderers, bundles, and XML parsers.

The templates are copied byte-for-byte from ycchen-tw/proof-pilot-codes commit
bc03a2c71a076990deaad3d712c6889682e12c69.  The same files occur in both
``distill_gen/math_3r/prompts`` and ``kaggle/proof_agent/prompts`` there.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path

PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "ycchen_math_3r"
PROMPT_SOURCE_COMMIT = "bc03a2c71a076990deaad3d712c6889682e12c69"
SYSTEM_DELIMITER = "===SYSTEM==="
USER_DELIMITER = "===USER==="

# Lenient, search-based extraction matching ycchen's gold parser (proof_agent/
# parser.py), not a strict whole-document fullmatch. See evaluation/PARSING_VS_GOLD.md
# for the full rationale and the per-case decisions.
_VALID_SCORES = (0.0, 0.5, 1.0)

_SOLUTION_OPEN = re.compile(r"<solution>", re.IGNORECASE)
# This model frequently omits </solution>; recover by stopping at the next section
# boundary (matches gold's _lenient_solution).
_SOLUTION_END = re.compile(
    r"</solution>|</?self_evaluation>|<score>", re.IGNORECASE
)
_SELF_EVALUATION = re.compile(
    r"<self_evaluation>(.*?)</self_evaluation>", re.IGNORECASE | re.DOTALL
)
_SCORE = re.compile(r"<score>(.*?)</score>", re.IGNORECASE | re.DOTALL)


def _recover_solution(text: str) -> str:
    """<solution> content, tolerating a missing </solution> and surrounding text."""
    opened = _SOLUTION_OPEN.search(text or "")
    if opened is None:
        return ""
    rest = text[opened.end():]
    end = _SOLUTION_END.search(rest)
    return (rest[: end.start()] if end else rest).strip()


def _parse_score(text: str) -> float | None:
    """<score> as a float snapped to {0, 0.5, 1}; None if absent or out of set.

    Accepts any float spelling (1, 1.0, 0.5, .5, ...) instead of a fixed literal,
    and compares with math.isclose rather than == so 1.0 == 1 and 0.5 are exact.
    """
    match = _SCORE.search(text or "")
    if match is None:
        return None
    try:
        value = float(match.group(1).strip())
    except ValueError:
        return None
    for valid in _VALID_SCORES:
        if math.isclose(value, valid, abs_tol=1e-9):
            return valid
    return None


@lru_cache(maxsize=None)
def template(name: str) -> str:
    return (PROMPT_ROOT / name).read_text()


def prompt_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((PROMPT_ROOT / name).read_bytes()).hexdigest()
        for name in ("prover.txt", "verifier.txt", "refiner.txt")
    }


def _messages(rendered: str) -> list[dict[str, str]]:
    system, user = rendered.split(USER_DELIMITER, 1)
    if not system.startswith(SYSTEM_DELIMITER):
        raise ValueError("ycchen prompt lacks the system delimiter")
    return [
        {"role": "system", "content": system.removeprefix(SYSTEM_DELIMITER).strip()},
        {"role": "user", "content": user.strip()},
    ]


def generation_messages(problem: str) -> list[dict[str, str]]:
    return _messages(template("prover.txt").replace("{problem}", problem))


def verification_messages(
    problem: str,
    proof: str,
    self_evaluation: str,
) -> list[dict[str, str]]:
    rendered = (
        template("verifier.txt")
        .replace("{problem}", problem)
        .replace("{candidate_solution}", proof)
        .replace("{candidate_self_eval}", self_evaluation)
    )
    return _messages(rendered)


def refinement_messages(
    problem: str,
    candidates: list[tuple[str, str, str, list[tuple[float, str]]]],
) -> list[dict[str, str]]:
    """Merge one or more parent candidates into a refiner prompt, gold-style.

    Each candidate is (candidate_id, proof, self_evaluation, reviews), where
    reviews is a list of (score, review_text). Mirrors gold's build_refine_bundle:
    per candidate, the proof, then each verifier review, then the self-evaluation
    only if non-empty (omitted, not sent empty, when dropped).
    """
    parts: list[str] = []
    for candidate_id, proof, self_evaluation, reviews in candidates:
        parts.append(f'<candidate id="{candidate_id}">')
        parts += ["<proof>", proof, "</proof>"]
        for review_score, review in reviews:
            parts += [
                f'<verifier_review score="{review_score:g}">',
                review,
                "</verifier_review>",
            ]
        if self_evaluation:
            parts += ["<self_evaluation>", self_evaluation, "</self_evaluation>"]
        parts.append("</candidate>")
    rendered = (
        template("refiner.txt")
        .replace("{problem}", problem)
        .replace("{candidate_bundle}", "\n".join(parts))
    )
    return _messages(rendered)


def parse_generation(text: str) -> tuple[str, str, float]:
    """(proof, self_evaluation, score). Lenient like gold: a non-empty <solution>
    (with missing-</solution> recovery) and a valid <score> are required; the
    self_evaluation may be empty, and surrounding/inter-section text is tolerated.
    """
    proof = _recover_solution(text)
    if not proof:
        raise ValueError("generation has no <solution> content")
    match = _SELF_EVALUATION.search(text or "")
    self_evaluation = match.group(1).strip() if match else ""
    score = _parse_score(text)
    if score is None:
        raise ValueError("generation has no valid <score> (0, 0.5, or 1)")
    return proof, self_evaluation, score


def parse_verification(text: str) -> tuple[str, float]:
    """(full verifier text, score). Only a valid <score> is required; the
    evaluation/suggestions body may be empty -- a perfect proof legitimately has
    no suggestions -- matching gold, which counts any parseable score.
    """
    score = _parse_score(text)
    if score is None:
        raise ValueError("verification has no valid <score> (0, 0.5, or 1)")
    return (text or "").strip(), score
