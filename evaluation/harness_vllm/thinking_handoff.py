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
HANDOFF_ASSISTANT_PREFIX = "\n</think>\n\n<handoff>\n"
HANDOFF_REQUIRED_SECTIONS = (
    "established",
    "promising",
    "failed",
    "uncertain",
    "bottleneck",
    "next_steps",
)
HANDOFF_VARIANTS = ("evidence_first", "lemma_ledger", "continuation_frontier")
DEFAULT_HANDOFF_VARIANT = HANDOFF_VARIANTS[0]
HANDOFF_SECTION_MAX_TOKENS = {
    "established": 768,
    "promising": 640,
    "failed": 512,
    "uncertain": 384,
    "bottleneck": 256,
    "next_steps": 384,
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

_SECTION_HEADER = re.compile(r"^===== (?P<title>.+?) =====$", re.MULTILINE)
_HANDOFF_BLOCK = re.compile(r"(?is)<handoff>\s*(.*?)\s*</handoff>")


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


def build_handoff_repair_instruction() -> str:
    return """Your previous handoff did not satisfy the required XML contract.

Re-emit the same mathematical handoff, without adding new reasoning, using exactly one nonempty instance of every required section:
<handoff><established>...</established><promising>...</promising><failed>...</failed><uncertain>...</uncertain><bottleneck>...</bottleneck><next_steps>...</next_steps></handoff>
Output no text outside the handoff block."""


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


def build_restart_instruction(handoff_text: str, restart_round: int) -> str:
    return f"""A previous attempt exhausted its reasoning budget. Start a fresh independent attempt from the original problem, using the handoff below only as untrusted research notes.

Verify every carried claim before relying on it. Do not merely repeat the previous route. Continue promising work where justified, replace failed approaches, and solve the problem completely if possible. This is restart round {restart_round}.

<previous_attempt_handoff>
{handoff_text}
</previous_attempt_handoff>"""


def insert_restart_instruction_into_rendered_prompt(
    rendered_prompt: str,
    handoff_text: str,
    restart_round: int,
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
    instruction = build_restart_instruction(handoff_text, restart_round)
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
) -> str | list[dict[str, str]]:
    instruction = build_restart_instruction(handoff_text, restart_round)
    if isinstance(prompt, str):
        return prompt + "\n\n---\n\n" + instruction
    messages = [dict(message) for message in prompt]
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("generation prompt must end in a user message")
    messages[-1]["content"] = messages[-1]["content"].rstrip() + "\n\n" + instruction
    return messages
