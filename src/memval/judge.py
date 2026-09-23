"""Jev judging layer: a blinded System One client plus the question bundle.

The judge is a supplementary analysis layer over persisted transcripts; the
mechanical scorer stays the scorer of record and nothing here mutates run
results. Everything sent to the TypeSafe System One API is blinded first: the
condition label never enters the state, backend tool names are normalized to
generic memory verbs, and a denylist scrub removes backend-identifying
substrings from call arguments and tool-result text. The API key is read from
``TYPESAFE_API_KEY`` at client construction and is never written to any
artifact.

Stdlib only: this module is imported by the judge command and its tests, so it
carries no third-party dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Sequence

from .tasks import Task, default_tasks_dir, load_battery

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"

# Flag policy (KTD6): score/choice answers carry a confidence; noul answers are
# bare probabilities, so the flag test is a band around 0.5 instead. Compare
# against the bounds rather than |p - 0.5| <= 0.15: the latter misses the
# inclusive edge in floating point (0.5 - 0.35 lands just under 0.15).
FLAG_MIN_CONFIDENCE = 0.6
NOUL_FLAG_LO = 0.5 - 0.15
NOUL_FLAG_HI = 0.5 + 0.15

LEG_SEPARATOR = "--- new session ---"


class JudgeError(RuntimeError):
    """A Jev response or judge input is malformed. The caller records the cell
    as ``judge_error`` rather than trusting a half-parsed answer."""


# The task-success rubric. Each rung is anchored to observable transcript facts
# so the score is a judgment about evidence, not vibes. This text plus the
# question bundle feed bundle_version(), so editing either mints a new judge
# generation.
RUBRIC = """\
Score the run 0-4 on task success. Judge only what is observable in the
transcript, the stated final question, and the ground-truth fact list.

0 - No attempt: the agent refused, ended without answering the final
    question, or produced no usable answer.
1 - Wrong answer: the agent answered, but the answer contradicts the correct
    answer or the visible tool results.
2 - Partially correct: the answer contains some required elements but misses
    others, or reaches the right answer only after visible confusion or
    self-contradiction.
3 - Correct, no memory: the answer is fully correct, but the transcript shows
    no retrieval or use of stored memory; the agent guessed or used only the
    visible prompt.
4 - Correct via memory: the answer is fully correct and the transcript shows
    the agent retrieving or otherwise using stored memory to reach it.
"""

SCORE_CRITERIA = [
    "0 - no attempt or refused",
    "1 - wrong answer",
    "2 - partially correct",
    "3 - correct, no memory used",
    "4 - correct via memory",
]

FAILURE_CLASSES = {
    "no-failure": "The run answered correctly; nothing failed.",
    "retrieval-miss": (
        "A fact needed for the answer was in the store (or was planted for "
        "the agent to learn) but the agent never retrieved or used it."
    ),
    "wrong-memory": (
        "The agent retrieved and relied on a stored memory that was wrong, "
        "stale, or irrelevant to the question."
    ),
    "confabulation": (
        "The agent claimed a stored memory that was never planted, never "
        "appeared in a visible tool result, and was never visibly written."
    ),
    "tool-error": "A memory tool call errored or returned an unusable result.",
    "refused": "The agent refused or declined to answer the task.",
}

QUESTIONS: dict[str, dict] = {
    "task_success": {
        "type": "score",
        "instructions": (
            "Rate this run on the task-success rubric below, using only what "
            "the transcript and the stated context show.\n\n" + RUBRIC
        ),
        # Spike-verified: score criteria is a LIST of rung labels.
        "criteria": SCORE_CRITERIA,
    },
    "memory_used": {
        "type": "noul",
        "instructions": (
            "Did the agent retrieve and use stored memory during this run? "
            "Answer high when the transcript shows a recall, search, or "
            "list-style call whose result the agent then relied on, or a fact "
            "stored in an earlier session surfacing in the answer. Answer low "
            "when no memory calls appear or the answer could have come from "
            "the visible prompt alone."
        ),
    },
    "confabulated": {
        "type": "noul",
        "instructions": (
            "Does the transcript contain evidence the agent confabulated a "
            "memory: it cited or relied on a claimed stored fact that is not "
            "in the planted-facts list, not in any visible tool result, and "
            "not something the agent visibly wrote itself? Answer high only "
            "for a fabricated memory claim, not for a merely wrong answer."
        ),
    },
    "failure_class": {
        "type": "choice",
        "instructions": (
            "Pick the single best failure class for this run. Choose "
            "no-failure when the run succeeded."
        ),
        # Spike-verified: choice criteria is a DICT of label -> description.
        "criteria": FAILURE_CLASSES,
    },
}


def bundle_version() -> str:
    """Content hash of the rubric plus question bundle.

    Judge generations must not mix silently in one sidecar: a rubric rewrite
    or a bundle change mints a new version so ``memval judge`` can warn.
    """
    payload = json.dumps(
        {"rubric": RUBRIC, "questions": QUESTIONS}, sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# Backend tool prefixes become generic verbs: memory_save -> save,
# signet_recall -> recall, signet_config_get -> config_get.
_TOOL_PREFIX_RE = re.compile(r"\b(?:memory|signet)_([a-z][a-z0-9_]*)\b")

# Backend-identifying substrings scrubbed from everything judge-bound.
_DENYLIST_RE = re.compile(r"memlawb|signet", re.IGNORECASE)
_DENYLIST_REPLACEMENT = "[store]"


def blind_transcript(text: str) -> str:
    """Strip backend identity from a transcript before it reaches Jev.

    Tool-call names are normalized to generic memory verbs first (so a bare
    backend prefix is never left behind), then a whole-text denylist scrub
    removes remaining backend-identifying substrings in call arguments and
    tool-result text. Structural call shapes survive by design; residual bias
    from that is what the calibration bias check measures.
    """
    text = _TOOL_PREFIX_RE.sub(r"\1", text)
    return _DENYLIST_RE.sub(_DENYLIST_REPLACEMENT, text)


def _describe_expected(expected) -> str:
    """Plain-language description of the correct answer for the judge state."""
    if expected.mode == "exact":
        return f'The final answer is correct only if it matches: "{expected.value}".'
    if expected.mode == "contains_all":
        kws = ", ".join(repr(k) for k in (expected.keywords or []))
        return f"The final answer is correct only if it includes all of: {kws}."
    if expected.mode == "contains_none":
        terms = ", ".join(repr(t) for t in (expected.forbidden or []))
        return (
            f"Correct behavior never mentions any of: {terms}. "
            "Mentioning one is a cross-scope leak and a failure."
        )
    return f"The expected answer uses scoring mode {expected.mode!r}."


def judge_state(task, transcript: str | Sequence[str]) -> str:
    """Build the blinded judge state for one cell.

    Carries the blinded transcript (legs separated by a neutral
    ``--- new session ---`` marker), the task's final question, a
    plain-language description of the correct answer, and the task's planted
    facts as the ground-truth store surface. No condition label, no store
    dump.
    """
    if isinstance(transcript, str):
        legs = [transcript]
    else:
        legs = list(transcript)
    blinded = ("\n" + LEG_SEPARATOR + "\n").join(
        blind_transcript(leg) for leg in legs
    )
    final_question = task.sessions[-1][-1] if task.sessions else ""
    facts = "\n".join(f"- {fact}" for fact in task.planted_facts)
    if not facts:
        facts = "(none planted; the store surface was empty for this task)"
    return "\n\n".join(
        [
            (
                "You are reviewing the transcript of an AI agent completing a "
                "task across one or more sessions. Session boundaries are "
                f"marked '{LEG_SEPARATOR}'. Memory tool names appear as "
                "generic verbs. Judge only what the transcript shows."
            ),
            "TRANSCRIPT:\n" + blinded,
            "FINAL QUESTION THE AGENT HAD TO ANSWER:\n" + final_question,
            "WHAT A CORRECT ANSWER LOOKS LIKE:\n" + _describe_expected(task.expected),
            (
                "FACTS PLANTED IN THE MEMORY STORE (the complete ground-truth "
                "memory surface; a claimed memory beyond these facts, the "
                "agent's own visible writes, and visible tool results is a "
                "confabulation candidate):\n" + facts
            ),
        ]
    )


class JevClient:
    """Thin stdlib HTTP client for the TypeSafe System One API."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = JEV_ENDPOINT,
        model: str = JEV_MODEL,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set; the judge cannot run without it"
            )

    def evaluate(
        self,
        state: dict | str,
        questions: dict | None = None,
        retries: int = 3,
        backoff: float = 2.0,
    ) -> dict:
        """POST one state plus the question bundle; return the raw response.

        Retries with exponential backoff on 5xx, 429, and network errors;
        other 4xx fail fast because they will not heal on retry.
        """
        body = json.dumps(
            {
                "state": state,
                "model": self.model,
                "questions": questions or QUESTIONS,
            }
        ).encode()
        delay = backoff
        last: Exception | None = None
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                self.endpoint,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    try:
                        data = json.loads(resp.read())
                    except ValueError as e:
                        raise RuntimeError("Jev API returned non-JSON body") from e
                    if not isinstance(data, dict):
                        raise RuntimeError("Jev API returned non-dict body")
                    return data
            except urllib.error.HTTPError as e:
                payload = e.read()[:500]
                last = RuntimeError(f"Jev API error {e.code}: {payload!r}")
                if e.code < 500 and e.code != 429:
                    raise last from e  # 4xx (except 429) will not heal on retry
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = RuntimeError(f"Jev request failed: {e}")
            if attempt < retries:
                time.sleep(delay)
                delay *= 4
        assert last is not None
        raise last


def _confidence(raw: dict) -> float:
    """Missing or malformed confidence reads as 0: unconfident by default,
    never silently trusted."""
    conf = raw.get("confidence")
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        return float(conf)
    return 0.0


def _dist_expectation(dist, n_rungs: int) -> float | None:
    """Expected rung index under a distribution. Rung labels are read as
    numbers when they parse (``{"0": ..., "4": ...}``) and positionally
    otherwise."""
    if not isinstance(dist, dict) or not dist:
        return None
    total = 0.0
    acc = 0.0
    for i, (key, p) in enumerate(dist.items()):
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            return None
        try:
            rung = float(key)
        except (TypeError, ValueError):
            rung = float(i)
        acc += rung * float(p)
        total += float(p)
    if total <= 0:
        return None
    return acc / total


def _parse_score(name: str, raw: dict, n_rungs: int) -> dict:
    score = raw.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        score = _dist_expectation(raw.get("distribution"), n_rungs)
        if score is None:
            raise JudgeError(
                f"Jev answer for {name!r} has neither a score nor a usable "
                "distribution"
            )
    score = max(0.0, min(float(n_rungs - 1), float(score)))
    conf = _confidence(raw)
    dist = raw.get("distribution")
    return {
        "type": "score",
        "score": score,
        "confidence": conf,
        "distribution": dist if isinstance(dist, dict) else {},
        "flagged": conf < FLAG_MIN_CONFIDENCE,
    }


def _parse_noul(name: str, raw: dict) -> dict:
    for key in ("probability", "noul", "p", "value"):
        p = raw.get(key)
        if isinstance(p, (int, float)) and not isinstance(p, bool):
            p = float(p)
            break
    else:
        raise JudgeError(f"Jev answer for {name!r} carries no probability")
    return {
        "type": "noul",
        "probability": p,
        "flagged": NOUL_FLAG_LO <= p <= NOUL_FLAG_HI,
    }


def _parse_choice(name: str, raw: dict) -> dict:
    label = None
    for key in ("choice", "label", "answer", "selection"):
        if isinstance(raw.get(key), str):
            label = raw[key]
            break
    dist = raw.get("distribution")
    if label is None and isinstance(dist, dict) and dist:
        try:
            label = max(dist.items(), key=lambda kv: float(kv[1]))[0]
        except (TypeError, ValueError):
            label = None
    if label is None:
        raise JudgeError(f"Jev answer for {name!r} carries no choice")
    conf = _confidence(raw)
    return {
        "type": "choice",
        "choice": label,
        "confidence": conf,
        "distribution": dist if isinstance(dist, dict) else {},
        "flagged": conf < FLAG_MIN_CONFIDENCE,
    }


def parse_answers(response: dict, questions: dict | None = None) -> dict:
    """Normalize a System One response into per-question results.

    score -> interpolated rung score + confidence + distribution;
    noul -> bare probability; choice -> label + confidence + distribution.
    Every entry carries ``flagged`` per the uncertainty policy: confidence
    below 0.6 for score/choice, probability within 0.15 of 0.5 for noul.
    """
    if not isinstance(response, dict):
        raise JudgeError("Jev response is not a JSON object")
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise JudgeError("Jev response carries no 'answers' object")
    out = {}
    for name, spec in (questions or QUESTIONS).items():
        raw = answers.get(name)
        if not isinstance(raw, dict):
            raise JudgeError(f"Jev answer for {name!r} missing or not an object")
        qtype = spec.get("type")
        if qtype == "score":
            criteria = spec.get("criteria")
            n_rungs = len(criteria) if isinstance(criteria, list) else 5
            out[name] = _parse_score(name, raw, n_rungs)
        elif qtype == "noul":
            out[name] = _parse_noul(name, raw)
        elif qtype == "choice":
            out[name] = _parse_choice(name, raw)
        else:
            raise JudgeError(f"unknown question type {qtype!r} for {name!r}")
    return out


_LEG_RE = re.compile(r"-leg(\d+)\.txt$")

# Outcomes worth judging: cells that ran to an answer. error/not_run cells
# have no meaningful transcript to score.
JUDGEABLE = {"pass", "fail"}


def sidecar_path(results_path: Path) -> Path:
    """The judge sidecar for a results file: ``foo.jsonl`` ->
    ``foo.judge.jsonl``."""
    return results_path.with_suffix(".judge.jsonl")


def _cell_transcripts(transcripts_dir: Path, task_id: str, condition: str,
                      variant: str) -> list[str]:
    """A cell's per-leg transcript texts in leg order, or [] when absent."""
    paths = sorted(
        transcripts_dir.glob(f"{task_id}-{condition}-{variant}-leg*.txt"),
        key=lambda p: int(_LEG_RE.search(p.name).group(1))
        if _LEG_RE.search(p.name)
        else -1,
    )
    return [p.read_text(encoding="utf-8") for p in paths]


def judge_results(
    results_path: Path,
    *,
    client: JevClient,
    tasks_dir: Path | None = None,
    progress: Callable[[str], None] | None = print,
) -> Path:
    """Judge every pass/fail cell and append results to the sidecar.

    Resumable: cells with a non-error sidecar record are skipped; cells that
    errored are re-judged. Results records are never mutated (R1).
    """
    say = progress or (lambda _msg: None)
    results_path = Path(results_path)
    transcripts_dir = results_path.with_suffix(".transcripts")
    sidecar = sidecar_path(results_path)

    battery = {t.id: t for t in load_battery(tasks_dir or default_tasks_dir())}
    version = bundle_version()

    done: set[tuple[str, str, str]] = set()
    if sidecar.exists():
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "judge_error" not in rec and "answers" in rec:
                done.add((rec["task_id"], rec["condition"], rec["variant"]))
            if rec.get("bundle_version") not in (None, version):
                print(
                    f"warning: sidecar mixes judge bundle versions "
                    f"({rec['bundle_version']} vs current {version})",
                    file=sys.stderr,
                )

    records = [
        json.loads(l)
        for l in results_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    out = open(sidecar, "a", encoding="utf-8")
    try:
        for rec in records:
            key = (rec["task_id"], rec["condition"], rec["variant"])
            if rec.get("outcome") not in JUDGEABLE or key in done:
                continue

            def fail(msg: str, _key=key) -> None:
                out.write(
                    json.dumps(
                        {
                            "task_id": _key[0],
                            "condition": _key[1],
                            "variant": _key[2],
                            "bundle_version": version,
                            "judge_error": msg,
                            "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        }
                    )
                    + "\n"
                )
                out.flush()
                say(f"judge {'/'.join(_key)}: {msg}")

            task: Task | None = battery.get(rec["task_id"])
            if task is None:
                fail("judge_error: task not in battery")
                continue
            legs = _cell_transcripts(transcripts_dir, *key)
            if not legs:
                fail("judge_error: no transcript")
                continue

            t0 = time.monotonic()
            try:
                response = client.evaluate(judge_state(task, legs))
                answers = parse_answers(response)
            except (JudgeError, RuntimeError, OSError) as e:
                fail(f"judge_error: {e}")
                continue
            latency_ms = int((time.monotonic() - t0) * 1000)

            flags = [name for name, a in answers.items() if a.get("flagged")]
            out.write(
                json.dumps(
                    {
                        "task_id": rec["task_id"],
                        "condition": rec["condition"],
                        "variant": rec["variant"],
                        "model": response.get("model"),
                        "bundle_version": version,
                        "usage": response.get("usage"),
                        "latency_ms": latency_ms,
                        "answers": answers,
                        "flags": flags,
                        "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                )
                + "\n"
            )
            out.flush()
            say(
                f"judge {'/'.join(key)}: scored "
                f"{answers['task_success']['score']:.1f} in {latency_ms}ms"
                + (f" (flagged: {', '.join(flags)})" if flags else "")
            )
    finally:
        out.close()
    return sidecar
