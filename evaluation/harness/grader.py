"""Parse strict findings-grade-reasoning JSON on the full IMO scale."""

import json

FIELDS = ["findings", "grade", "reasoning"]
VALID_GRADES = set(range(8))


def parse_score(text: str) -> dict:
    """Extract one strictly ordered structured IMO grading result."""
    try:
        pairs = json.loads(text, object_pairs_hook=lambda items: items)
    except json.JSONDecodeError as error:
        raise ValueError("grader output is not valid JSON") from error
    if (
        type(pairs) is not list
        or len(pairs) != len(FIELDS)
        or any(type(item) is not tuple or len(item) != 2 for item in pairs)
    ):
        raise ValueError("grader output must be one JSON object")
    keys = [item[0] for item in pairs]
    if keys != FIELDS:
        raise ValueError(f"grader fields/order differ: {keys}")
    values = dict(pairs)
    findings = values["findings"]
    grade = values["grade"]
    reasoning = values["reasoning"]
    if (
        type(findings) is not list
        or not findings
        or any(type(item) is not str or not item.strip() for item in findings)
    ):
        raise ValueError("findings must be a non-empty array of non-empty strings")
    if type(grade) is not int or grade not in VALID_GRADES:
        raise ValueError(f"off-scale grader grade: {grade!r}")
    if type(reasoning) is not str or not reasoning.strip():
        raise ValueError("reasoning must be a non-empty string")
    return {
        "findings": [item.strip() for item in findings],
        "grade": grade,
        "reasoning": reasoning.strip(),
    }
