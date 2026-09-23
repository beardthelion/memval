"""Task battery loader and validator.

Each file in ``tasks/`` is one task: scripted user turns split into legs
across a simulated session boundary, the facts planted before that boundary,
and a machine-checkable expected answer. The schema is the contract between
the battery and the runner/scorer, so it is validated eagerly: task ids land
in memlawb namespaces and on-disk paths (slug charset only), and a task that
is malformed or context-answerable would silently corrupt results rather than
fail loudly.

Stdlib only: this module is imported by the runner, the scorer tests, and
battery tooling, so it carries no third-party dependencies.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

TASK_TYPES = {"recall", "follow-through", "isolation"}
EXPECTED_MODES = {"exact", "contains_all", "contains_none"}

# Fields the schema recognizes. Anything else in a task file is an authoring
# bug, not forward compatibility: silent ignorance is how batteries rot.
TOP_LEVEL_FIELDS = {"id", "type", "sessions", "planted_facts", "expected", "control", "users"}
EXPECTED_FIELDS = {"mode", "value", "keywords", "forbidden"}

# Which payload field each scoring mode consumes (mirrors R11).
MODE_FIELD = {"exact": "value", "contains_all": "keywords", "contains_none": "forbidden"}

REQUIRED_FIELDS = {"id", "type", "sessions", "planted_facts", "expected", "control"}


class TaskValidationError(ValueError):
    """A task file violates the battery schema. The message names the file
    or task id and the rule it broke."""


@dataclass
class Expected:
    """The machine-checkable answer. The mode picks which payload field the
    scorer reads; the others stay None."""

    mode: str
    value: str | None = None
    keywords: list[str] | None = None
    forbidden: list[str] | None = None


@dataclass
class Task:
    id: str
    type: str
    sessions: list[list[str]]
    planted_facts: list[str]
    expected: Expected
    control: bool
    users: list[str] = field(default_factory=list)

    @property
    def memory_dependent(self) -> bool:
        """A task that plants facts can only be answered across the boundary
        via memory, so it must ship a control variant (R4)."""
        return bool(self.planted_facts)


def validate_task(raw: dict, source: str = "<dict>") -> Task:
    """Validate one parsed task object. ``source`` is the file name or other
    provenance label used in error messages."""

    def fail(rule: str) -> NoReturn:
        tid = raw.get("id") if isinstance(raw, dict) else None
        label = f"{source} (id={tid})" if isinstance(tid, str) and tid else str(source)
        raise TaskValidationError(f"{label}: {rule}")

    if not isinstance(raw, dict):
        fail("task must be a JSON object")

    for key in raw:
        if key not in TOP_LEVEL_FIELDS:
            fail(f"unknown field {key!r}")

    for key in REQUIRED_FIELDS:
        if key not in raw:
            fail(f"missing required field {key!r}")

    task_id = raw["id"]
    if not isinstance(task_id, str) or not TASK_ID_RE.match(task_id):
        fail("id must match ^[a-z0-9][a-z0-9-]*$ (ids land in namespaces and paths)")

    task_type = raw["type"]
    if task_type not in TASK_TYPES:
        fail(f"type must be one of {sorted(TASK_TYPES)}")

    sessions = raw["sessions"]
    if not isinstance(sessions, list) or not sessions:
        fail("sessions must be a non-empty list of legs")
    for i, leg in enumerate(sessions):
        if not isinstance(leg, list) or not leg:
            fail(f"sessions leg {i} must be a non-empty list of user turns")
        for turn in leg:
            if not isinstance(turn, str):
                fail(f"sessions leg {i} contains a non-string turn")

    planted_facts = raw["planted_facts"]
    if not isinstance(planted_facts, list) or not all(
        isinstance(f, str) for f in planted_facts
    ):
        fail("planted_facts must be a list of strings")

    expected = _validate_expected(raw["expected"], fail)

    control = raw["control"]
    if not isinstance(control, bool):
        fail("control must be a boolean")

    if planted_facts:
        # Memory-dependent tasks need the boundary (R2) and a control variant
        # to prove the measured gain is causal (R4).
        if not control:
            fail("memory-dependent task requires control: true")
        if len(sessions) < 2:
            fail("memory-dependent task requires at least 2 session legs")

    users = raw.get("users", [])
    if not isinstance(users, list) or not all(isinstance(u, str) for u in users):
        fail("users must be a list of strings")
    if task_type == "isolation" and len(users) < 2:
        fail("isolation task requires at least 2 users")

    return Task(
        id=task_id,
        type=task_type,
        sessions=sessions,
        planted_facts=planted_facts,
        expected=expected,
        control=control,
        users=users,
    )


def _validate_expected(raw: object, fail) -> Expected:
    if not isinstance(raw, dict):
        fail("expected must be an object")

    for key in raw:
        if key not in EXPECTED_FIELDS:
            fail(f"expected: unknown field {key!r}")

    mode = raw.get("mode")
    if mode not in EXPECTED_MODES:
        fail(f"expected.mode must be one of {sorted(EXPECTED_MODES)}")

    required = MODE_FIELD[mode]
    if required not in raw:
        fail(f"expected.{mode} requires field {required!r}")

    payload = raw[required]
    if mode == "exact":
        if not isinstance(payload, str):
            fail("expected.value must be a string")
    else:
        if not isinstance(payload, list) or not payload:
            fail(f"expected.{required} must be a non-empty list of strings")
        if not all(isinstance(k, str) for k in payload):
            fail(f"expected.{required} must be a non-empty list of strings")

    return Expected(
        mode=mode,
        value=raw.get("value"),
        keywords=raw.get("keywords"),
        forbidden=raw.get("forbidden"),
    )


def default_tasks_dir() -> Path:
    """Battery location: package-relative when running from the checkout,
    else cwd so an installed package can run against a battery in the
    working directory."""
    here = Path(__file__).resolve().parent.parent.parent / "tasks"
    return here if here.is_dir() else Path.cwd() / "tasks"


def load_battery(tasks_dir: str | Path) -> list[Task]:
    """Load and validate every ``*.json`` task file in ``tasks_dir``,
    in filename order."""
    tasks_dir = Path(tasks_dir)
    tasks = []
    seen_ids: set[str] = set()
    for path in sorted(tasks_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TaskValidationError(f"{path.name}: invalid JSON: {exc}") from exc
        task = validate_task(raw, source=path.name)
        if task.id in seen_ids:
            raise TaskValidationError(
                f"{path.name} (id={task.id}): duplicate task id; ids land in "
                "namespaces and paths, so collisions cross-contaminate scopes"
            )
        seen_ids.add(task.id)
        tasks.append(task)
    return tasks


def echo_guard(task: Task) -> list[str]:
    """Planted-fact strings that leak into post-boundary prompts (R20).

    If a fact's wording appears verbatim in a later leg's user turns, a
    no-memory run could answer by lexical echo alone, which makes the task a
    battery defect rather than a measurement. Returns the offending fact
    strings; a healthy battery returns ``[]`` for every task.
    """
    post_boundary = [
        turn.casefold() for leg in task.sessions[1:] for turn in leg
    ]
    return [
        fact
        for fact in task.planted_facts
        if any(fact.casefold() in turn for turn in post_boundary)
    ]
