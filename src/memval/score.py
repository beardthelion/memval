"""Mechanical scoring (R11). No LLM judge anywhere in this file.

exact: normalized final answer equals expected.value.
contains_all: every keyword appears in the normalized final answer on token
boundaries, so '12pm' cannot satisfy a '2pm' keyword.
contains_none: no forbidden token appears in ANY assistant message of the
closing leg: a mid-session leak must not pass just because the last reply
was clean. Forbidden terms match as substrings on purpose: a leak is a
leak even inside a larger token.
"""
from __future__ import annotations

import re

from .tasks import Task


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _contains_term(haystack: str, term: str) -> bool:
    """Token-boundary keyword match on normalized text."""
    needle = normalize(term)
    if not needle:
        return False
    return (
        re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack)
        is not None
    )


def score(task: Task, final_answer: str, closing_messages: list[str]) -> str:
    """Return 'pass' or 'fail' for a completed cell."""
    exp = task.expected
    if exp.mode == "exact":
        return "pass" if normalize(final_answer) == normalize(exp.value or "") else "fail"
    if exp.mode == "contains_all":
        hay = normalize(final_answer)
        return "pass" if all(_contains_term(hay, k) for k in (exp.keywords or [])) else "fail"
    if exp.mode == "contains_none":
        hay = normalize(" ".join(closing_messages))
        return "fail" if any(normalize(k) in hay for k in (exp.forbidden or [])) else "pass"
    raise ValueError(f"unknown scoring mode: {exp.mode}")


def control_outcome(
    task_type: str,
    live_outcome: str,
    control_result: str,
    sentinel_landed: bool,
    read_calls_blinded: int,
    witness_ok: bool | None,
) -> str:
    """Classify a control cell (R5, R23).

    collapsed: retrieval provably blinded, writes provably landed, and the
    score dropped versus live. Both halves must be proven: a control that
    never attempted a blinded read, or whose sentinel write never landed,
    produced no causal evidence and is inconclusive, not collapsed.
    no_drop: the control scored the same as a passing live run; the task is
    flagged invalid (it was not measuring memory).
    inconclusive: the harness could not prove both halves.

    Isolation tasks are exempt from the drop rule: a correct isolation run
    passes under control too, since blinded reads can never leak. Their
    control validity rides on the witness instead, so a failed or absent
    witness is inconclusive.
    """
    if task_type == "isolation":
        if not sentinel_landed or witness_ok is not True:
            return "inconclusive"
        # Passing under blinded reads is correct behavior here, not a defect.
        return "collapsed" if control_result == "pass" else "no_drop"
    if not sentinel_landed or read_calls_blinded < 1:
        return "inconclusive"
    if live_outcome == "pass" and control_result == "pass":
        return "no_drop"
    if control_result in ("fail", "error") and live_outcome == "pass":
        return "collapsed"
    return "inconclusive"
