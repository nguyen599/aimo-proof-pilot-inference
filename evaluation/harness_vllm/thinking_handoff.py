from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


FINAL_PARTIAL_FORCE_TEXT = (
    "\n</think>\n\n<solution>\n"
    "We were unable to produce a complete proof. However, the strongest "
    "partial progress is as follows:\n"
)
FINAL_PARTIAL_FORCE_MARKER = (
    "We were unable to produce a complete proof. However, the strongest"
)
STRUCTURED_PARTIAL_FORCE_PREFIX = (
    "\n</think>\n\n<solution>\n"
    "We were unable to produce a complete proof within this reasoning budget. "
    "Stop solving now and write a compact research transfer note for a fresh "
    "independent attempt. Recover only mathematical state already present in "
    "your reasoning; do not derive new claims or guess the final answer.\n\n"
)
RESTART_FINALIZE_FORCE_TEXT = (
    "\nWe must stop exploratory reasoning now and write the final answer. "
    "Audit every construction, calculation, and impossibility claim against "
    "the exact problem before using it. Do not cite an omitted argument, an "
    "official solution, numerical evidence, or a supposedly standard fact "
    "without proving the needed statement. Remove any claim that is not fully "
    "justified. If a complete proof cannot be finished, give the strongest "
    "rigorous partial proof and assign score 0 or 0.5 honestly rather than "
    "claiming completeness. Do not continue searching.\n"
    "</think>\n\n<solution>\n"
)
HANDOFF_ASSISTANT_PREFIX = "\n</think>\n\n<handoff>\n"
# The chat template already opens the model's private reasoning turn. Let it
# close that turn itself before emitting the final checkpoint audit; opening
# the XML here makes private reasoning look like audit content.
CHECKPOINT_AUDIT_ASSISTANT_PREFIX = ""
HANDOFF_REQUIRED_SECTIONS = (
    "established",
    "promising",
    "failed",
    "uncertain",
    "bottleneck",
    "next_steps",
)
PROOF_CHECKPOINT_VARIANT = "proof_checkpoint"
HANDOFF_VARIANTS = (
    "evidence_first",
    "lemma_ledger",
    "continuation_frontier",
    PROOF_CHECKPOINT_VARIANT,
)
DEFAULT_HANDOFF_VARIANT = HANDOFF_VARIANTS[0]
HANDOFF_MODES = ("model", "lossless_partial")
DEFAULT_HANDOFF_MODE = HANDOFF_MODES[0]
RESTART_STRATEGIES = ("standard", "deadline_aware")
DEFAULT_RESTART_STRATEGY = RESTART_STRATEGIES[0]
HANDOFF_SECTION_MAX_TOKENS = {
    "established": 768,
    "promising": 640,
    "failed": 512,
    "uncertain": 384,
    "bottleneck": 256,
    "next_steps": 384,
}
MAP_REDUCE_SECTION_MAX_TOKENS = {
    "established": 320,
    "promising": 256,
    "failed": 192,
    "uncertain": 160,
    "bottleneck": 128,
    "next_steps": 192,
}
HANDOFF_SECTION_DESCRIPTIONS = {
    "established": (
        "Only facts, exact equations, reductions, or partial lemmas that were "
        "actually justified in the previous attempt. Use at most 6 bullets."
    ),
    "promising": (
        "Promising constructions, reductions, equations, or observations worth "
        "continuing. Use at most 5 bullets."
    ),
    "failed": (
        "Routes already tried, with the precise obstruction or missing step. "
        "Use at most 4 bullets."
    ),
    "uncertain": (
        "Potentially useful claims or patterns that were not proved. Label the "
        "uncertainty explicitly and use at most 4 bullets."
    ),
    "bottleneck": (
        "The narrowest unresolved point preventing a complete proof. Use one "
        "paragraph of at most 120 words."
    ),
    "next_steps": (
        "A prioritized continuation plan for a fresh independent solver. Use at "
        "most 5 concrete bullets."
    ),
}
RENDERED_ASSISTANT_MARKERS = (
    "<｜Assistant｜>",
    "<|assistant|>",
    "<|start_header_id|>assistant<|end_header_id|>",
    "<|im_start|>assistant",
)
RENDERED_USER_MARKERS = (
    "<｜User｜>",
    "<|user|>",
    "<|start_header_id|>user<|end_header_id|>",
    "<|im_start|>user",
)

_SECTION_HEADER = re.compile(r"^===== (?P<title>.+?) =====$", re.MULTILINE)
_HANDOFF_BLOCK = re.compile(r"(?is)<handoff>\s*(.*?)\s*</handoff>")
_HANDOFF_CONTROL_TAG = re.compile(
    r"(?is)<\s*/?\s*(?:handoff|"
    + "|".join(re.escape(section) for section in HANDOFF_REQUIRED_SECTIONS)
    + r")\s*>"
)
_PROVED_LEMMA_BLOCK = re.compile(
    r"(?is)\[PROVED_LEMMA\s+(?P<identifier>[A-Za-z0-9_.-]+)\]\s*"
    r"STATEMENT:\s*(?P<statement>.*?)\s*"
    r"DEPENDENCIES:\s*(?P<dependencies>.*?)\s*"
    r"FULL_PROOF:\s*(?P<proof>.*?)\s*"
    r"\[END_PROVED_LEMMA\]"
)
_PROOF_CHECKPOINT_NO_LEMMA = "NO_FULLY_PROVED_REUSABLE_LEMMA"
_PROOF_SHORTCUT = re.compile(
    r"(?i)\b(?:"
    r"straightforward|routine|similarly|analogously|"
    r"details? (?:are )?omitted|proof omitted|without proof|"
    r"left to (?:the )?reader|one can show|it can be shown|"
    r"after (?:some|lengthy|routine) (?:algebra|calculation)|"
    r"by a standard (?:argument|lemma|result)"
    r")\b"
)
_PROOF_UNCERTAINTY = re.compile(
    r"(?i)\b(?:maybe|likely|possibly|not sure|might|seems?|appears?|"
    r"conjectur(?:e|ed|al)|unproved|unverified|incomplete|"
    r"not (?:complete|proved|proven)|no complete proof|"
    r"remain(?:s|ed)? (?:incomplete|unproved|unproven)|"
    r"need(?:s)? to (?:be )?(?:prove|proved|proven))\b"
)
_CHECKPOINT_AUDIT_BLOCK = re.compile(
    r"(?is)\[LEMMA_AUDIT\s+(?P<identifier>[A-Za-z0-9_.-]+)\]\s*"
    r"VERDICT:\s*(?P<verdict>REUSABLE|MINOR_CHECK|REPROVE)\s*"
    r"REASON:\s*(?P<reason>.*?)\s*"
    r"\[END_LEMMA_AUDIT\]"
)


@dataclass(frozen=True)
class SavedProofGenerationCall:
    path: Path
    stage: str
    detail: str
    prompt_tokens: int
    max_tokens: int
    input_prompt: str
    continuation_prompt: str
    continuation_prompt_tokens: int
    continuation_max_tokens: int
    output_text: str
    finish_reason: str
    usage: dict[str, Any]


def _parse_header_lines(text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("====="):
            break
        key, separator, value = line.partition(":")
        if separator:
            headers[key.strip()] = value.strip()
    return headers


def _section_ranges(text: str) -> dict[str, tuple[int, int]]:
    matches = list(_SECTION_HEADER.finditer(text))
    ranges: dict[str, tuple[int, int]] = {}
    for index, match in enumerate(matches):
        start = match.end()
        if start < len(text) and text[start] == "\n":
            start += 1
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        ranges[match.group("title")] = (start, end)
    return ranges


def _parse_segment(section: str) -> tuple[int, int, str]:
    normalized = section.removeprefix("\n")
    metadata, separator, encoded_body = normalized.partition("\n\n")
    metadata_lines = metadata.splitlines()
    if not separator or len(metadata_lines) != 2:
        raise ValueError("continuation section is missing metadata")
    prompt_tokens = int(metadata_lines[0].removeprefix("prompt_tokens:").strip())
    max_tokens = int(metadata_lines[1].removeprefix("max_tokens:").strip())
    # The logger appends one newline after the decoded prompt, and the next
    # section begins with another newline. Remove exactly those delimiters so
    # prompt-significant trailing whitespace remains intact.
    if not encoded_body.endswith("\n\n"):
        raise ValueError("continuation section is missing its trailing delimiters")
    body = encoded_body[:-2]
    return prompt_tokens, max_tokens, body


def _parse_output(section: str) -> tuple[str, dict[str, Any], str]:
    lines = section.lstrip("\n").splitlines()
    finish_reason = ""
    usage: dict[str, Any] = {}
    body_start = 0
    for index, line in enumerate(lines):
        if not line.strip():
            body_start = index + 1
            break
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.strip() == "finish_reason":
            finish_reason = value.strip()
        elif key.strip() == "usage":
            usage = json.loads(value.strip())
    return finish_reason, usage, "\n".join(lines[body_start:]).rstrip()


def parse_saved_proof_generation_call(path: Path) -> SavedProofGenerationCall:
    text = path.read_text(encoding="utf-8")
    headers = _parse_header_lines(text)
    ranges = _section_ranges(text)
    input_range = ranges.get("INPUT PROMPT")
    output_range = ranges.get("OUTPUT")
    continuation_titles = sorted(
        (title for title in ranges if title.startswith("CONTINUATION INPUT PROMPT ")),
        key=lambda title: int(title.rsplit(" ", 1)[-1]),
    )
    if input_range is None or output_range is None or not continuation_titles:
        raise ValueError(f"{path} is not a budget-intervened proof-generation log")

    continuation_range = ranges[continuation_titles[-1]]
    continuation_prompt_tokens, continuation_max_tokens, continuation_prompt = (
        _parse_segment(text[slice(*continuation_range)])
    )
    finish_reason, usage, output_text = _parse_output(text[slice(*output_range)])
    return SavedProofGenerationCall(
        path=path,
        stage=headers.get("stage", ""),
        detail=headers.get("detail", ""),
        prompt_tokens=int(headers.get("prompt_tokens", "0")),
        max_tokens=int(headers.get("max_tokens", "0")),
        input_prompt=text[slice(*input_range)].strip(),
        continuation_prompt=continuation_prompt,
        continuation_prompt_tokens=continuation_prompt_tokens,
        continuation_max_tokens=continuation_max_tokens,
        output_text=output_text,
        finish_reason=finish_reason,
        usage=usage,
    )


def remove_final_partial_force_text(text: str) -> str:
    force_index = text.rfind(FINAL_PARTIAL_FORCE_TEXT)
    if force_index >= 0:
        return text[:force_index]

    marker_index = text.rfind(FINAL_PARTIAL_FORCE_MARKER)
    if marker_index < 0:
        raise ValueError(
            "the saved continuation does not contain the final force marker"
        )
    close_index = text.rfind("</think>", 0, marker_index)
    if close_index < 0:
        raise ValueError("the saved continuation lacks the forced </think> marker")
    return text[:close_index].rstrip()


def extract_forced_partial_progress(text: str) -> str:
    raw = str(text or "")
    marker_index = raw.rfind(FINAL_PARTIAL_FORCE_MARKER)
    if marker_index < 0:
        raise ValueError("proof output does not contain the final partial marker")
    end = len(raw)
    for marker in ("</solution>", "<self_evaluation>", "<score>"):
        position = raw.find(marker, marker_index)
        if position >= 0:
            end = min(end, position)
    progress = raw[marker_index:end].strip()
    if not progress:
        raise ValueError("forced partial progress is empty")
    return progress


def escape_handoff_control_tags(text: str) -> str:
    """Prevent quoted research notes from closing the deterministic wrapper."""

    return _HANDOFF_CONTROL_TAG.sub(
        lambda match: "&lt;" + match.group(0)[1:],
        str(text or ""),
    )


def build_lossless_partial_handoff(partial_progress: str) -> str:
    """Wrap an untrusted cutoff report without asking another model to edit it."""

    report = escape_handoff_control_tags(partial_progress).strip()
    if not report:
        raise ValueError("partial progress report is empty")
    return assemble_handoff(
        {
            "established": (
                "No claim from the previous attempt is accepted as established. "
                "Recheck every carried statement against the original problem."
            ),
            "promising": (
                "The complete untrusted partial-progress report is preserved below. "
                "It may contain useful constructions, calculations, and gaps, but it "
                "may also contain contradictions or errors.\n\n"
                "<untrusted_partial_progress>\n"
                f"{report}\n"
                "</untrusted_partial_progress>"
            ),
            "failed": (
                "The previous attempt exhausted its reasoning budget before a "
                "complete proof. Do not resume a repetitive route merely because it "
                "appears in the report."
            ),
            "uncertain": (
                "Every mathematical claim in the quoted report remains unverified "
                "until independently proved or checked."
            ),
            "bottleneck": (
                "No complete general proof was produced. The fresh solver must "
                "identify the actual missing argument after auditing the report."
            ),
            "next_steps": (
                "First verify the concrete constructions and impossibility claims. "
                "Discard contradictions, recover only sound lemmas, then restart the "
                "general proof independently and complete every missing case."
            ),
        }
    )


def build_proof_checkpoint_instruction() -> str:
    """Request a proof-carrying checkpoint from an exhausted proof context."""

    return f"""The previous proof attempt exhausted its reasoning budget before producing a final answer.

Stop solving. Archive the reusable mathematical state for a fresh solver. This is not a short summary: every lemma placed in `established` must include its complete proof, with all equations, cases, and dependencies needed to check and reuse it without deriving it again.

The response has already been prefixed with `<handoff>`. Begin immediately with `<established>`; do not emit another `<handoff>` opening tag. Close `</established>` only after the final complete lemma block, then emit the other five sections in the exact order below. Emit `</handoff>` exactly once, at the very end.

For every completely proved reusable lemma, use exactly this plain-text block inside `<established>`:

[PROVED_LEMMA L1]
STATEMENT:
The exact statement, including every hypothesis and quantifier.
DEPENDENCIES:
`ORIGINAL_PROBLEM_ONLY` or a comma-separated list of earlier lemma IDs.
FULL_PROOF:
The complete proof already present in the previous reasoning. Preserve every necessary calculation and case. Copy it verbatim when possible; light normalization is allowed only when it does not change the mathematics.
[END_PROVED_LEMMA]

Number later blocks L2, L3, and so on. If no reusable lemma received a complete proof, write exactly `{_PROOF_CHECKPOINT_NO_LEMMA}` inside `<established>`.

Proof-preservation rules:
- Never promote a plausible, numerical, incomplete, or contradicted claim into `established`.
- Never replace proof steps with "straightforward", "routine", "similarly", "one can show", an omitted calculation, or a reference to an unstated standard result.
- If preserving a complete proof would exceed the response budget, omit that lemma from `established` and put its proved prefix plus exact gap in `uncertain`; do not resume solving to complete it.
- Do not use the original problem statement or final target as a lemma unless the previous reasoning already contains its complete final proof. Prefer independently reusable intermediate results.
- A lemma whose proof has any gap belongs in `uncertain`, together with the exact proved prefix and exact missing step.
- Do not add new mathematics, repair a proof, guess a result, or continue searching.
- Retain exact notation, constructions, formulas, equality conditions, edge cases, and counterexamples needed by a fresh solver.
- Prefer one complete reusable lemma over several compressed or incomplete lemmas.
- There is no word or bullet limit other than the response token limit.

Use the remaining sections as follows:
- `promising`: detailed reusable derivations or constructions that are not complete proofs.
- `failed`: attempted routes and their precise failure points.
- `uncertain`: unproved claims, incomplete lemma proofs, and contradictions. Never put these in `established`.
- `bottleneck`: the narrowest remaining obstacle.
- `next_steps`: a concrete continuation plan that cites lemma IDs when applicable.

Output exactly one block and no text outside it:

<handoff>
<established>proof blocks or {_PROOF_CHECKPOINT_NO_LEMMA}</established>
<promising>detailed unproved but reusable work, or NONE</promising>
<failed>failed routes with exact obstructions, or NONE</failed>
<uncertain>all remaining gaps and contradictions, or NONE</uncertain>
<bottleneck>the precise unresolved point</bottleneck>
<next_steps>the prioritized continuation plan</next_steps>
</handoff>

Close every tag. Do not emit a solution, self-evaluation, score, or new reasoning."""


def analyze_proof_checkpoint(text: str) -> dict[str, Any]:
    """Check checkpoint structure without claiming mathematical correctness."""

    parsed = parse_handoff_response(text)
    established = parsed["sections"].get("established", "")
    matches = list(_PROVED_LEMMA_BLOCK.finditer(established))
    identifiers = [match.group("identifier") for match in matches]
    issues: list[str] = []
    lemmas: list[dict[str, Any]] = []

    if len(set(identifiers)) != len(identifiers):
        issues.append("duplicate_lemma_ids")

    residual = _PROVED_LEMMA_BLOCK.sub("", established).strip()
    declared_no_lemma = residual == _PROOF_CHECKPOINT_NO_LEMMA
    if matches and residual:
        issues.append("text_outside_lemma_blocks")
    if not matches and not declared_no_lemma:
        issues.append("missing_lemma_blocks_or_no_lemma_marker")

    for match in matches:
        statement = match.group("statement").strip()
        dependencies = match.group("dependencies").strip()
        proof = match.group("proof").strip()
        lemma_issues: list[str] = []
        if not statement:
            lemma_issues.append("missing_statement")
        if not dependencies:
            lemma_issues.append("missing_dependencies")
        if len(proof) < 80:
            lemma_issues.append("proof_too_short")
        if _PROOF_SHORTCUT.search(proof):
            lemma_issues.append("proof_contains_omission_shortcut")
        if _PROOF_UNCERTAINTY.search(proof):
            lemma_issues.append("proof_contains_uncertainty")
        lemmas.append(
            {
                "id": match.group("identifier"),
                "statement": statement,
                "dependencies": dependencies,
                "proof": proof,
                "proof_chars": len(proof),
                "issues": lemma_issues,
                "structurally_reusable": not lemma_issues,
            }
        )
        issues.extend(f"{match.group('identifier')}:{issue}" for issue in lemma_issues)

    return {
        "is_valid_handoff": bool(parsed["is_valid"]),
        "is_valid_checkpoint": bool(parsed["is_valid"] and not issues),
        "declared_no_complete_lemma": declared_no_lemma,
        "lemma_count": len(lemmas),
        "structurally_reusable_lemma_count": sum(
            bool(lemma["structurally_reusable"]) for lemma in lemmas
        ),
        "total_proof_chars": sum(int(lemma["proof_chars"]) for lemma in lemmas),
        "issues": issues,
        "lemmas": lemmas,
    }


def build_structured_partial_force_text(
    variant: str = DEFAULT_HANDOFF_VARIANT,
) -> str:
    """Force a bounded research ledger directly from the exhausted context."""

    guidance = handoff_variant_guidance(variant)
    return (
        STRUCTURED_PARTIAL_FORCE_PREFIX
        + f"""Global emphasis: {guidance}

Use exactly these plain-text headings:

VERIFIED:
- Only facts, equations, constructions, or partial lemmas fully justified above.

UNVERIFIED:
- Concrete potentially useful claims or patterns that still need proof.

FAILED:
- Routes already tried, each followed by its exact obstruction.

BOTTLENECK:
- The narrowest unresolved mathematical point.

NEXT:
- At most five concrete steps for a fresh solver.

Rules:
- Treat an item as UNVERIFIED unless its proof was completed above.
- Preserve exact formulas, examples, and counterexamples when useful.
- Explicitly retain contradictions instead of silently choosing one side.
- Do not restate the full problem or its output format.
- Do not continue solving, add new arguments, or claim completeness.
- Keep the entire transfer note below 1,200 words.
"""
    )


def _render_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=False,
        )
    )


def build_user_turn_transition_text(
    tokenizer: Any,
    user_instruction: str,
    *,
    close_open_thinking: bool,
    assistant_prefix: str = HANDOFF_ASSISTANT_PREFIX,
) -> str:
    marker = "__AIMO_ASSISTANT_CONTENT_MARKER_7BCA2C7E__"
    assistant_only = _render_chat_template(
        tokenizer,
        [{"role": "assistant", "content": marker}],
        add_generation_prompt=False,
    )
    assistant_then_user = _render_chat_template(
        tokenizer,
        [
            {"role": "assistant", "content": marker},
            {"role": "user", "content": user_instruction},
        ],
        add_generation_prompt=True,
    )
    marker_index = assistant_only.find(marker)
    if marker_index < 0:
        raise ValueError("chat template did not preserve the assistant marker")
    if not assistant_then_user.startswith(assistant_only):
        raise ValueError(
            "chat template does not preserve an assistant-only rendering when "
            "a user turn is appended"
        )

    assistant_tail = assistant_only[marker_index + len(marker) :]
    user_and_generation = assistant_then_user[len(assistant_only) :]
    close_reasoning = "\n</think>" if close_open_thinking else ""
    transition = (
        close_reasoning + assistant_tail + user_and_generation + assistant_prefix
    )
    if user_instruction not in transition:
        raise ValueError("rendered transition lost the user handoff instruction")
    return transition


def build_user_turn_prompt_ids(
    tokenizer: Any,
    active_context_ids: Iterable[int],
    user_instruction: str,
    *,
    close_open_thinking: bool,
    assistant_prefix: str = HANDOFF_ASSISTANT_PREFIX,
) -> list[int]:
    transition = build_user_turn_transition_text(
        tokenizer,
        user_instruction,
        close_open_thinking=close_open_thinking,
        assistant_prefix=assistant_prefix,
    )
    transition_ids = tokenizer.encode(transition, add_special_tokens=False)
    if hasattr(transition_ids, "tolist"):
        transition_ids = transition_ids.tolist()
    return [int(value) for value in active_context_ids] + [
        int(value) for value in transition_ids
    ]


def build_handoff_instruction(variant: str = DEFAULT_HANDOFF_VARIANT) -> str:
    if variant not in HANDOFF_VARIANTS:
        raise ValueError(f"unsupported handoff prompt variant: {variant!r}")
    if variant == PROOF_CHECKPOINT_VARIANT:
        return build_proof_checkpoint_instruction()

    variant_guidance = handoff_variant_guidance(variant)
    return f"""The previous proof attempt exhausted its reasoning budget before producing a final proof.

Do not continue solving the problem and do not pretend that the proof is complete. Compress the previous attempt into a faithful handoff for a fresh solver. Preserve useful formulas, definitions, reductions, and partial lemmas. Clearly label every unproved claim. Explain why abandoned approaches failed. {variant_guidance}

Strict compression rules:
- Keep the entire handoff below 1,200 words.
- Do not restate the full problem or copy its output-format instructions.
- Never repeat the same fact, route, or caveat in multiple sections.
- Avoid narration such as "the previous attempt attempted"; state the mathematical fact directly.
- Use at most 6 bullets in established, 5 in promising, 4 in failed, 4 in uncertain, and 5 in next_steps.
- Keep every bullet to at most 2 sentences.
- Keep bottleneck to one paragraph of at most 120 words.
- If a section has no useful content, state that in one short sentence instead of inventing content.

Output exactly these XML sections and no text outside them:

<handoff>
<established>Only facts or lemmas actually justified in the previous attempt.</established>
<promising>Promising constructions, reductions, equations, or observations worth continuing.</promising>
<failed>Approaches already tried, including the precise obstruction or missing step.</failed>
<uncertain>Claims, patterns, or conjectures that may be useful but were not proved.</uncertain>
<bottleneck>The exact unresolved point preventing a complete proof.</bottleneck>
<next_steps>A short prioritized plan for a fresh independent attempt.</next_steps>
</handoff>

Close every XML tag. Be concise, factual, and useful. Do not include a final solution, self-evaluation, or score."""


def handoff_variant_guidance(variant: str) -> str:
    if variant not in HANDOFF_VARIANTS:
        raise ValueError(f"unsupported handoff prompt variant: {variant!r}")
    return {
        "evidence_first": (
            "Prioritize rigorously established facts and exact equations. Remove "
            "repetition and speculation unless it identifies a concrete dead end."
        ),
        "lemma_ledger": (
            "Organize the mathematical state as a lemma ledger: distinguish proved "
            "claims, plausible but unproved claims, and disproved or abandoned routes."
        ),
        "continuation_frontier": (
            "Optimize for the next solver: identify the narrowest unresolved frontier "
            "and give a ranked, concrete continuation plan with reusable notation."
        ),
        PROOF_CHECKPOINT_VARIANT: (
            "Preserve complete, dependency-labelled lemma proofs as reusable proof "
            "checkpoints; never compress an established proof into a claim."
        ),
    }[variant]


def build_handoff_section_instruction(
    section: str,
    variant: str = DEFAULT_HANDOFF_VARIANT,
) -> str:
    if section not in HANDOFF_REQUIRED_SECTIONS:
        raise ValueError(f"unsupported handoff section: {section!r}")
    guidance = handoff_variant_guidance(variant)
    description = HANDOFF_SECTION_DESCRIPTIONS[section]
    return f"""The previous proof attempt exhausted its reasoning budget before producing a final proof.

Extract only the `{section}` portion of a faithful handoff for a fresh solver. Do not continue solving the problem. Do not restate the full problem, repeat ideas, invent claims, or discuss output formatting.

Section requirement: {description}
Global emphasis: {guidance}

Output only the section contents. Do not emit XML tags, a heading, a final solution, a self-evaluation, or a score. Stop as soon as this section is complete."""


def handoff_section_assistant_prefix(section: str) -> str:
    if section not in HANDOFF_REQUIRED_SECTIONS:
        raise ValueError(f"unsupported handoff section: {section!r}")
    return f"\n</think>\n\n<{section}>\n"


def normalize_handoff_section(text: str, section: str) -> str:
    if section not in HANDOFF_REQUIRED_SECTIONS:
        raise ValueError(f"unsupported handoff section: {section!r}")
    content = str(text or "")
    closing_tag = f"</{section}>"
    if closing_tag in content:
        content = content.split(closing_tag, 1)[0]
    content = re.sub(r"(?is)</?handoff>", "", content)
    content = re.sub(
        rf"(?is)</?{re.escape(section)}>",
        "",
        content,
    ).strip()
    if content:
        return content
    return "No useful information for this section was preserved."


def assemble_handoff(sections: dict[str, str]) -> str:
    return (
        "<handoff>\n"
        + "\n".join(
            f"<{section}>{normalize_handoff_section(sections.get(section, ''), section)}</{section}>"
            for section in HANDOFF_REQUIRED_SECTIONS
        )
        + "\n</handoff>"
    )


def extract_rendered_problem_text(rendered_prompt: str) -> str:
    text = str(rendered_prompt or "")
    user_match: tuple[int, str] | None = None
    for marker in RENDERED_USER_MARKERS:
        position = text.rfind(marker)
        if position >= 0 and (user_match is None or position > user_match[0]):
            user_match = (position, marker)
    if user_match is None:
        raise ValueError("rendered prompt does not contain a supported user marker")
    start = user_match[0] + len(user_match[1])
    end = len(text)
    for marker in RENDERED_ASSISTANT_MARKERS:
        position = text.find(marker, start)
        if position >= 0:
            end = min(end, position)
    problem = text[start:end].strip()
    for separator in (
        "Respond in EXACTLY this format:",
        "Output ONLY the final answer",
    ):
        if separator in problem:
            problem = problem.split(separator, 1)[0].strip()
    if not problem:
        raise ValueError("rendered prompt contains an empty user problem")
    return problem


def truncate_consecutive_token_repetition(
    token_ids: list[int],
    *,
    search_tail_tokens: int = 16_384,
    block_sizes: tuple[int, ...] = (256, 128, 64, 32, 16),
    minimum_repeats: int = 4,
) -> tuple[list[int], dict[str, int] | None]:
    values = [int(value) for value in token_ids]
    search_start = max(0, len(values) - max(1, search_tail_tokens))
    for block_size in block_sizes:
        required = block_size * minimum_repeats
        for start in range(search_start, len(values) - required + 1):
            block = values[start : start + block_size]
            if all(
                values[start + repeat * block_size : start + (repeat + 1) * block_size]
                == block
                for repeat in range(1, minimum_repeats)
            ):
                return values[:start], {
                    "start": start,
                    "block_tokens": block_size,
                    "minimum_repeats": minimum_repeats,
                }
    return values, None


def truncate_low_novelty_token_tail(
    token_ids: list[int],
    *,
    window_tokens: int = 2_048,
    stride_tokens: int = 512,
    ngram_tokens: int = 8,
    ngram_stride: int = 2,
    novelty_threshold: float = 0.12,
    minimum_consecutive_windows: int = 2,
    minimum_prefix_tokens: int = 2_048,
) -> tuple[list[int], dict[str, Any] | None]:
    values = [int(value) for value in token_ids]
    if len(values) < minimum_prefix_tokens + window_tokens:
        return values, None

    low_novelty_start: int | None = None
    low_novelty_windows: list[dict[str, Any]] = []
    final_start = len(values) - window_tokens
    for start in range(minimum_prefix_tokens, final_start + 1, stride_tokens):
        window = values[start : start + window_tokens]
        ngrams = [
            tuple(window[index : index + ngram_tokens])
            for index in range(
                0,
                len(window) - ngram_tokens + 1,
                ngram_stride,
            )
        ]
        novelty = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
        if novelty <= novelty_threshold:
            if low_novelty_start is None:
                low_novelty_start = start
                low_novelty_windows = []
            low_novelty_windows.append(
                {
                    "start": start,
                    "end": start + window_tokens,
                    "novelty": novelty,
                }
            )
            if len(low_novelty_windows) >= minimum_consecutive_windows:
                return values[:low_novelty_start], {
                    "kind": "low_novelty",
                    "start": low_novelty_start,
                    "window_tokens": window_tokens,
                    "ngram_tokens": ngram_tokens,
                    "threshold": novelty_threshold,
                    "windows": low_novelty_windows,
                }
        else:
            low_novelty_start = None
            low_novelty_windows = []
    return values, None


def select_reasoning_token_windows(
    token_ids: list[int],
    *,
    total_tokens: int = 32_768,
    window_tokens: int = 4_096,
) -> list[tuple[int, int, list[int]]]:
    values = [int(value) for value in token_ids]
    if len(values) <= total_tokens:
        return [(0, len(values), values)]
    window_tokens = max(1, min(window_tokens, total_tokens))
    window_count = max(1, total_tokens // window_tokens)
    if window_count == 1:
        start = max(0, len(values) - window_tokens)
        return [(start, len(values), values[start:])]
    final_start = len(values) - window_tokens
    starts = [
        round(index * final_start / (window_count - 1)) for index in range(window_count)
    ]
    windows: list[tuple[int, int, list[int]]] = []
    for start in dict.fromkeys(starts):
        end = min(len(values), start + window_tokens)
        windows.append((start, end, values[start:end]))
    return windows


def prepare_handoff_research_windows(
    tokenizer: Any,
    *,
    original_input_prompt: str,
    pre_force_text: str,
    reasoning_total_tokens: int = 32_768,
    reasoning_window_tokens: int = 4_096,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if not pre_force_text.startswith(original_input_prompt):
        raise ValueError("pre-force context does not start with the original prompt")
    reasoning_text = pre_force_text[len(original_input_prompt) :]
    reasoning_ids = tokenizer.encode(reasoning_text, add_special_tokens=False)
    if hasattr(reasoning_ids, "tolist"):
        reasoning_ids = reasoning_ids.tolist()
    original_reasoning_ids = [int(value) for value in reasoning_ids]
    cleaned_reasoning_ids, repetition = truncate_consecutive_token_repetition(
        original_reasoning_ids
    )
    cleaned_reasoning_ids, low_novelty = truncate_low_novelty_token_tail(
        cleaned_reasoning_ids
    )
    selected = select_reasoning_token_windows(
        cleaned_reasoning_ids,
        total_tokens=reasoning_total_tokens,
        window_tokens=reasoning_window_tokens,
    )
    windows = [
        {
            "start": start,
            "end": end,
            "text": tokenizer.decode(ids, skip_special_tokens=False),
        }
        for start, end, ids in selected
    ]
    return (
        extract_rendered_problem_text(original_input_prompt),
        windows,
        {
            "reasoning_original_tokens": len(original_reasoning_ids),
            "reasoning_cleaned_tokens": len(cleaned_reasoning_ids),
            "repetition": repetition,
            "low_novelty": low_novelty,
            "window_ranges": [(window["start"], window["end"]) for window in windows],
        },
    )


def render_handoff_extraction_prompt_ids(
    tokenizer: Any,
    *,
    user_content: str,
    assistant_prefix: str,
    system_content: str | None = None,
) -> list[int]:
    messages = [
        {
            "role": "system",
            "content": system_content
            or (
                "You are a mathematical research-state compressor. Follow the "
                "latest extraction instruction exactly. Do not solve the problem."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
    )
    rendered += assistant_prefix
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    return [int(value) for value in prompt_ids]


def build_proof_checkpoint_audit_prompt_ids(
    tokenizer: Any,
    *,
    original_input_prompt: str,
    checkpoint_text: str,
) -> list[int]:
    """Build a fresh-context audit of checkpoint lemma proofs."""

    problem = extract_rendered_problem_text(original_input_prompt)
    user_content = f"""Audit a proof checkpoint against the original problem. Do not solve the original problem from scratch and do not improve or complete any proof. Check only whether every claimed `PROVED_LEMMA` is a self-contained, logically valid consequence of its hypotheses and listed dependencies.

Original problem:
{problem}

Checkpoint:
{checkpoint_text}

For every `PROVED_LEMMA`, output one block in the same order:

[LEMMA_AUDIT L1]
VERDICT: REUSABLE
REASON: A concrete justification in at most four sentences.
[END_LEMMA_AUDIT]

The verdict must be exactly one of:
- REUSABLE: the written proof is complete and can be cited without rederivation after checking dependencies.
- MINOR_CHECK: the proof is essentially complete but one precise local detail needs a brief verification.
- REPROVE: a missing argument, invalid step, circular dependency, omitted calculation, or false statement requires substantial new proof work.

After all lemma blocks, output exactly `OVERALL: PASS` only if every lemma is REUSABLE or MINOR_CHECK; otherwise output `OVERALL: FAIL`. If the checkpoint declares that no lemma was completed, output only `OVERALL: NO_LEMMAS`.

Do not accept phrases such as "straightforward", "routine", "similarly", or "one can show" in place of an argument. You may reason privately in the model's `<think>` section. After closing `</think>`, output exactly one `<checkpoint_audit>` block, no other visible text, and close the tag."""
    return render_handoff_extraction_prompt_ids(
        tokenizer,
        user_content=user_content,
        assistant_prefix=CHECKPOINT_AUDIT_ASSISTANT_PREFIX,
        system_content=(
            "You are a strict mathematical proof-checkpoint auditor. Judge only "
            "the supplied lemma proofs and never repair them."
        ),
    )


def parse_proof_checkpoint_audit(text: str) -> dict[str, Any]:
    raw = str(text or "")
    final_region = raw
    think_close = raw.lower().rfind("</think>")
    if think_close >= 0:
        final_region = raw[think_close + len("</think>") :]
    matches = list(
        re.finditer(
            r"(?is)<checkpoint_audit>\s*(.*?)\s*</checkpoint_audit>",
            final_region,
        )
    )
    closed = bool(matches)
    if matches:
        body = matches[-1].group(1).strip()
    else:
        open_matches = list(
            re.finditer(r"(?is)<checkpoint_audit>\s*", final_region)
        )
        body = (
            final_region[open_matches[-1].end() :].strip()
            if open_matches
            else ""
        )
    audits = [
        {
            "id": match.group("identifier"),
            "verdict": match.group("verdict").upper(),
            "reason": match.group("reason").strip(),
        }
        for match in _CHECKPOINT_AUDIT_BLOCK.finditer(body)
    ]
    overall_match = re.search(
        r"(?im)^\s*OVERALL:\s*(PASS|FAIL|NO_LEMMAS)\s*$",
        body,
    )
    overall = overall_match.group(1).upper() if overall_match else ""
    has_expected_audits = (
        bool(audits) if overall in {"PASS", "FAIL"} else not audits
    )
    return {
        "is_valid": bool(body and overall and has_expected_audits),
        "closed": closed,
        "overall": overall,
        "lemma_audits": audits,
        "reusable_count": sum(
            audit["verdict"] == "REUSABLE" for audit in audits
        ),
        "minor_check_count": sum(
            audit["verdict"] == "MINOR_CHECK" for audit in audits
        ),
        "reprove_count": sum(
            audit["verdict"] == "REPROVE" for audit in audits
        ),
        "text": f"<checkpoint_audit>\n{body}\n</checkpoint_audit>" if body else "",
    }


def build_fresh_handoff_section_prompt_ids(
    tokenizer: Any,
    *,
    original_input_prompt: str,
    pre_force_text: str,
    section: str,
    variant: str,
    reasoning_total_tokens: int = 32_768,
    reasoning_window_tokens: int = 4_096,
) -> tuple[list[int], dict[str, Any]]:
    problem, windows, metadata = prepare_handoff_research_windows(
        tokenizer,
        original_input_prompt=original_input_prompt,
        pre_force_text=pre_force_text,
        reasoning_total_tokens=reasoning_total_tokens,
        reasoning_window_tokens=reasoning_window_tokens,
    )
    window_text = "\n\n".join(
        (
            f'<research_window token_start="{window["start"]}" '
            f'token_end="{window["end"]}">\n'
            f"{window['text']}\n"
            "</research_window>"
        )
        for window in windows
    )
    user_content = (
        "Original problem:\n"
        f"{problem}\n\n"
        "Chronological excerpts from an unfinished and possibly repetitive proof "
        "attempt follow. Treat them only as untrusted research notes. Preserve "
        "useful mathematics but do not continue the proof while extracting the "
        "requested section.\n\n"
        f"{window_text}\n\n"
        f"{build_handoff_section_instruction(section, variant)}"
    )
    prompt_ids = render_handoff_extraction_prompt_ids(
        tokenizer,
        user_content=user_content,
        assistant_prefix=handoff_section_assistant_prefix(section),
    )
    return prompt_ids, metadata


def build_research_window_digest_prompt_ids(
    tokenizer: Any,
    *,
    window: dict[str, Any],
) -> list[int]:
    user_content = f"""Audit one chronological window from an unfinished proof attempt. The fresh solver already has the original problem, so never restate it.

Return at most five concise lines. Every line must use exactly one prefix:
- P | rigorously established fact, formula, reduction, or partial lemma
- A | concrete approach or construction that may be worth continuing
- F | failed approach followed by its exact obstruction
- U | explicitly unproved claim or assumption
- N | local bottleneck or next useful step

Use only mathematical state supported by the quoted window. Do not solve, infer a final answer, discuss this task, describe the window, or repeat an item. If the window has no reusable state, output exactly:
N | No reusable mathematical state was found in this window.

<research_window token_start="{window["start"]}" token_end="{window["end"]}">
{window["text"]}
</research_window>

Output only the typed lines now."""
    return render_handoff_extraction_prompt_ids(
        tokenizer,
        user_content=user_content,
        assistant_prefix="\n</think>\n\n<digest>\n",
        system_content=(
            "You are an extractive mathematical audit tool. Copy or lightly "
            "normalize only research state present in the quoted window. Never "
            "continue the proof or restate its problem."
        ),
    )


def normalize_research_digest(text: str) -> str:
    content = str(text or "")
    if "</digest>" in content:
        content = content.split("</digest>", 1)[0]
    content = re.sub(r"(?is)</?digest>", "", content).strip()
    typed_lines: list[str] = []
    for line in content.splitlines():
        match = re.match(r"^\s*[-*]?\s*([PAFUN])\s*\|\s*(.+?)\s*$", line)
        if not match:
            continue
        normalized = f"{match.group(1)} | {match.group(2)}"
        if normalized not in typed_lines:
            typed_lines.append(normalized)
        if len(typed_lines) >= 5:
            break
    if typed_lines:
        return "\n".join(typed_lines)
    return "N | No reusable mathematical state was found in this window."


def build_handoff_from_digests_prompt_ids(
    tokenizer: Any,
    *,
    problem: str,
    digests: list[dict[str, Any]],
    variant: str,
) -> list[int]:
    digest_text = "\n\n".join(
        (
            f'<research_digest token_start="{digest["start"]}" '
            f'token_end="{digest["end"]}">\n'
            f"{digest['text']}\n"
            "</research_digest>"
        )
        for digest in digests
    )
    user_content = (
        "Original problem:\n"
        f"{problem}\n\n"
        "Chronological extractive digests from an unfinished proof attempt:\n\n"
        f"{digest_text}\n\n"
        f"{build_handoff_instruction(variant)}"
    )
    return render_handoff_extraction_prompt_ids(
        tokenizer,
        user_content=user_content,
        assistant_prefix=HANDOFF_ASSISTANT_PREFIX,
    )


def build_handoff_section_from_digests_prompt_ids(
    tokenizer: Any,
    *,
    digests: list[dict[str, Any]],
    section: str,
    variant: str,
) -> list[int]:
    digest_text = "\n\n".join(
        (
            f'<research_digest token_start="{digest["start"]}" '
            f'token_end="{digest["end"]}">\n'
            f"{digest['text']}\n"
            "</research_digest>"
        )
        for digest in digests
    )
    user_content = (
        "The fresh solver already has the original problem. Organize only the "
        "extractive research digests below; do not restate or solve the problem.\n\n"
        f"{digest_text}\n\n"
        f"{build_handoff_section_instruction(section, variant)}"
    )
    return render_handoff_extraction_prompt_ids(
        tokenizer,
        user_content=user_content,
        assistant_prefix=handoff_section_assistant_prefix(section),
        system_content=(
            "You are a mathematical research-state editor. Use only the supplied "
            "typed digests. Never add proof steps, conjectures, or problem text."
        ),
    )


def build_handoff_section_from_partial_progress_prompt_ids(
    tokenizer: Any,
    *,
    partial_progress: str,
    section: str,
    variant: str,
) -> list[int]:
    user_content = (
        "The fresh solver already has the original problem. The quoted report was "
        "generated after an earlier attempt exhausted its reasoning budget. Treat "
        "every claim as untrusted and organize only what the report actually says. "
        "Do not restate or solve the problem.\n\n"
        "<partial_progress_report>\n"
        f"{partial_progress}\n"
        "</partial_progress_report>\n\n"
        f"{build_handoff_section_instruction(section, variant)}"
    )
    return render_handoff_extraction_prompt_ids(
        tokenizer,
        user_content=user_content,
        assistant_prefix=handoff_section_assistant_prefix(section),
        system_content=(
            "You are a mathematical research-state editor. Extract only from the "
            "quoted partial-progress report. Never add proof steps or claims."
        ),
    )


def build_handoff_repair_instruction(
    variant: str = DEFAULT_HANDOFF_VARIANT,
) -> str:
    checkpoint_rules = ""
    if variant == PROOF_CHECKPOINT_VARIANT:
        checkpoint_rules = f"""

Preserve the proof-checkpoint contract. Inside `established`, use only complete `[PROVED_LEMMA Lx]` blocks with nonempty `STATEMENT`, `DEPENDENCIES`, and `FULL_PROOF` fields, or exactly `{_PROOF_CHECKPOINT_NO_LEMMA}`. Do not shorten a proof, replace steps with omission language, promote an unproved claim, or add new mathematics. Move every incomplete block to `uncertain`."""
    return f"""Your previous handoff did not satisfy the required structure.

Re-emit the same mathematical handoff, without adding new reasoning, using exactly one nonempty instance of every required section:
<handoff><established>...</established><promising>...</promising><failed>...</failed><uncertain>...</uncertain><bottleneck>...</bottleneck><next_steps>...</next_steps></handoff>
Output no text outside the handoff block.{checkpoint_rules}"""


def parse_handoff_response(text: str) -> dict[str, Any]:
    raw = str(text or "")
    matches = list(_HANDOFF_BLOCK.finditer(raw))
    body = matches[-1].group(1) if matches else ""
    sections: dict[str, str] = {}
    for name in HANDOFF_REQUIRED_SECTIONS:
        section_matches = list(
            re.finditer(
                rf"(?is)<{name}>\s*(.*?)\s*</{name}>",
                body,
            )
        )
        sections[name] = section_matches[-1].group(1).strip() if section_matches else ""
    missing = [name for name, value in sections.items() if not value]
    return {
        "is_valid": bool(matches and not missing),
        "has_handoff_block": bool(matches),
        "missing_sections": missing,
        "sections": sections,
        "text": (
            "<handoff>\n"
            + "\n".join(
                f"<{name}>{sections[name]}</{name}>"
                for name in HANDOFF_REQUIRED_SECTIONS
            )
            + "\n</handoff>"
            if matches
            else ""
        ),
    }


def build_restart_instruction(
    handoff_text: str,
    restart_round: int,
    strategy: str = DEFAULT_RESTART_STRATEGY,
) -> str:
    if strategy not in RESTART_STRATEGIES:
        raise ValueError(f"unsupported restart strategy: {strategy!r}")
    deadline_guidance = ""
    if strategy == "deadline_aware":
        deadline_guidance = """

Treat this as the final proof-writing attempt, not another open-ended research pass. Audit the carried notes briefly, choose one coherent route, and stop exploratory case enumeration well before the external token cutoff. Reserve enough budget to close your reasoning and emit the required final answer. If a complete proof remains out of reach, voluntarily stop and present the strongest rigorous partial proof with its gap stated explicitly instead of running until the cutoff."""
    checkpoint_guidance = ""
    if "[PROVED_LEMMA " in handoff_text:
        checkpoint_guidance = """

The `PROVED_LEMMA` blocks are proof-carrying checkpoints, not mere claims. Perform a brief local audit that each statement matches its hypotheses, its listed dependencies are available, and its written proof has no gap. If that audit passes, cite and reuse the lemma directly; do not spend the new attempt rederiving it. Reprove only a block for which you identify a concrete defect. Everything outside those blocks remains untrusted."""
    return f"""A previous attempt exhausted its reasoning budget. Start a fresh independent attempt from the original problem, using the handoff below as preserved research state.

Verify unproved carried claims before relying on them. Do not merely repeat the previous route. Continue promising work where justified, replace failed approaches, and solve the problem completely if possible. This is restart round {restart_round}.{checkpoint_guidance}{deadline_guidance}

<previous_attempt_handoff>
{handoff_text}
</previous_attempt_handoff>"""


def insert_restart_instruction_into_rendered_prompt(
    rendered_prompt: str,
    handoff_text: str,
    restart_round: int,
    strategy: str = DEFAULT_RESTART_STRATEGY,
) -> str:
    """Insert a restart note into the final user turn of a rendered prompt."""
    marker_positions = [
        (rendered_prompt.rfind(marker), marker) for marker in RENDERED_ASSISTANT_MARKERS
    ]
    marker_index, marker = max(marker_positions, key=lambda item: item[0])
    if marker_index < 0:
        raise ValueError(
            "rendered proof prompt does not contain a recognized assistant marker"
        )
    instruction = build_restart_instruction(
        handoff_text,
        restart_round,
        strategy,
    )
    restarted = (
        rendered_prompt[:marker_index]
        + "\n\n"
        + instruction
        + rendered_prompt[marker_index:]
    )
    if restarted.count(instruction) != 1 or marker not in restarted:
        raise ValueError("failed to insert restart instruction into rendered prompt")
    return restarted


def append_restart_instruction(
    prompt: str | list[dict[str, str]],
    handoff_text: str,
    restart_round: int,
    strategy: str = DEFAULT_RESTART_STRATEGY,
) -> str | list[dict[str, str]]:
    instruction = build_restart_instruction(
        handoff_text,
        restart_round,
        strategy,
    )
    if isinstance(prompt, str):
        return prompt + "\n\n---\n\n" + instruction
    messages = [dict(message) for message in prompt]
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("generation prompt must end in a user message")
    messages[-1]["content"] = messages[-1]["content"].rstrip() + "\n\n" + instruction
    return messages


def append_final_output_discipline(
    prompt: str | list[dict[str, str]],
    target_tokens: int,
) -> str | list[dict[str, str]]:
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    instruction = f"""Final-output discipline for this repair:

- Use the hidden reasoning phase to choose and audit the argument.
- Once `<solution>` begins, write only the finalized proof. Do not narrate search, try alternative cases, or debate possible approaches inside the answer.
- Target at most {target_tokens:,} tokens for the complete visible response.
- Always close `<solution>`, `<self_evaluation>`, and `<score>` before stopping.
- If a necessary lemma remains unproved, stop early and give the strongest rigorous partial proof, name the exact gap in `<self_evaluation>`, and use score 0 or 0.5. Never spend the remaining budget searching inside `<solution>`."""
    if isinstance(prompt, str):
        return prompt + "\n\n" + instruction
    messages = [dict(message) for message in prompt]
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("generation prompt must end in a user message")
    messages[-1]["content"] = (
        messages[-1]["content"].rstrip() + "\n\n" + instruction
    )
    return messages
