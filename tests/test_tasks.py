"""Tests for the task battery schema and loader (U2).

The battery is the eval contract: every task crosses a session boundary, so
validation exists to catch authoring bugs that would silently make a task
context-answerable or leak its own answer through the prompt (R2, R20).
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from memval.tasks import TaskValidationError, echo_guard, load_battery, validate_task

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


def _base_task(**overrides):
    task = {
        "id": "recall-test",
        "type": "recall",
        "sessions": [
            ["Remember that I prefer pepperjack cheese."],
            ["What cheese should go on the sandwich?"],
        ],
        "planted_facts": ["I prefer pepperjack cheese", "pepperjack"],
        "expected": {"mode": "contains_all", "keywords": ["pepperjack"]},
        "control": True,
    }
    task.update(overrides)
    return task


def test_validate_task_returns_task():
    task = validate_task(_base_task())
    assert task.id == "recall-test"
    assert task.type == "recall"
    assert task.expected.mode == "contains_all"
    assert task.expected.keywords == ["pepperjack"]
    assert task.control is True


def test_missing_expected_rejected():
    task = _base_task()
    del task["expected"]
    with pytest.raises(TaskValidationError, match="expected"):
        validate_task(task)


def test_unknown_mode_rejected():
    task = _base_task(expected={"mode": "fuzzy", "value": "x"})
    with pytest.raises(TaskValidationError, match="mode"):
        validate_task(task)


def test_control_false_rejected():
    task = _base_task(control=False)
    with pytest.raises(TaskValidationError, match="control"):
        validate_task(task)


@pytest.mark.parametrize("bad_id", ["Recall-01", "recall 01", "-recall", "", "recall_01"])
def test_bad_id_rejected(bad_id):
    task = _base_task(id=bad_id)
    with pytest.raises(TaskValidationError, match="id"):
        validate_task(task)


def test_isolation_one_user_rejected():
    task = _base_task(
        id="leak-test",
        type="isolation",
        users=["alice"],
        expected={"mode": "contains_none", "forbidden": ["biscuit"]},
    )
    with pytest.raises(TaskValidationError, match="users"):
        validate_task(task)


def test_isolation_no_users_rejected():
    task = _base_task(
        id="leak-test",
        type="isolation",
        expected={"mode": "contains_none", "forbidden": ["biscuit"]},
    )
    with pytest.raises(TaskValidationError, match="users"):
        validate_task(task)


def test_unknown_top_level_field_rejected():
    task = _base_task(difficulty="easy")
    with pytest.raises(TaskValidationError, match="difficulty"):
        validate_task(task)


def test_empty_sessions_rejected():
    task = _base_task(sessions=[])
    with pytest.raises(TaskValidationError, match="sessions"):
        validate_task(task)


def test_single_leg_rejected_for_memory_task():
    task = _base_task(sessions=[["Remember that I prefer pepperjack cheese."]])
    with pytest.raises(TaskValidationError, match="leg"):
        validate_task(task)


def test_expected_missing_mode_field_rejected():
    task = _base_task(expected={"mode": "exact"})
    with pytest.raises(TaskValidationError, match="value"):
        validate_task(task)


def test_unknown_type_rejected():
    task = _base_task(type="trivia")
    with pytest.raises(TaskValidationError, match="type"):
        validate_task(task)


def test_load_battery_loads_all_shipped_tasks():
    tasks = load_battery(TASKS_DIR)
    assert len(tasks) == 25


def test_battery_mix():
    tasks = load_battery(TASKS_DIR)
    counts = Counter(t.type for t in tasks)
    assert counts == {"recall": 10, "follow-through": 9, "isolation": 6}


def test_battery_echo_guard_clean():
    for task in load_battery(TASKS_DIR):
        assert echo_guard(task) == [], f"{task.id} echoes planted facts post-boundary"


def test_battery_facts_actually_planted():
    # planted_facts must be reachable via scripted pre-boundary turns, or the
    # memory conditions have nothing to learn.
    for task in load_battery(TASKS_DIR):
        pre_boundary = [turn.casefold() for turn in task.sessions[0]]
        for fact in task.planted_facts:
            assert any(
                fact.casefold() in turn for turn in pre_boundary
            ), f"{task.id}: planted fact never scripted: {fact!r}"


def test_battery_isolation_users():
    for task in load_battery(TASKS_DIR):
        if task.type == "isolation":
            assert len(task.users) >= 2
            assert task.expected.mode == "contains_none"


def test_echo_guard_catches_echoing_task():
    task = validate_task(
        _base_task(
            sessions=[
                ["Remember that I prefer pepperjack cheese."],
                ["Should I use pepperjack on the sandwich?"],
            ]
        )
    )
    leaks = echo_guard(task)
    assert "pepperjack" in leaks


def test_load_battery_bad_json(tmp_path):
    bad = tmp_path / "task-bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(TaskValidationError, match="task-bad.json"):
        load_battery(tmp_path)


def test_load_battery_rejects_duplicate_ids(tmp_path):
    for name in ("task-a.json", "task-b.json"):
        (tmp_path / name).write_text(json.dumps(_base_task()), encoding="utf-8")
    with pytest.raises(TaskValidationError, match="duplicate"):
        load_battery(tmp_path)


def test_load_battery_validation_names_file(tmp_path):
    bad = tmp_path / "task-bad.json"
    bad.write_text(json.dumps(_base_task(id="BAD ID")), encoding="utf-8")
    with pytest.raises(TaskValidationError, match="task-bad.json"):
        load_battery(tmp_path)
