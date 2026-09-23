from memval.score import control_outcome, normalize, score
from memval.tasks import validate_task


def _task(mode, **exp):
    return validate_task(
        {
            "id": "t-1",
            "type": "recall",
            "sessions": [["x"], ["y"]],
            "planted_facts": ["f"],
            "expected": {"mode": mode, **exp},
            "control": True,
        }
    )


def _isolation_task():
    return validate_task(
        {
            "id": "t-2",
            "type": "isolation",
            "sessions": [["x"], ["y"]],
            "planted_facts": ["f"],
            "expected": {"mode": "contains_none", "forbidden": ["anchovies"]},
            "control": True,
            "users": ["a", "b"],
        }
    )


def test_exact():
    t = _task("exact", value="Pepperjack, please.")
    assert score(t, " pepperjack,   please. ", ["x"]) == "pass"
    assert score(t, "cheddar", ["x"]) == "fail"


def test_contains_all():
    t = _task("contains_all", keywords=["pepperjack", "sourdough"])
    assert score(t, "Use pepperjack on SOURDOUGH.", ["x"]) == "pass"
    assert score(t, "Use pepperjack on rye.", ["x"]) == "fail"


def test_contains_none_scans_all_closing_messages():
    t = _task("contains_none", forbidden=["anchovies"])
    # Mid-session leak, clean final answer: must still fail (R11).
    assert score(t, "I don't know.", ["You like anchovies!", "I don't know."]) == "fail"
    assert score(t, "I don't know.", ["I don't know."]) == "pass"


def test_control_collapsed():
    t = _task("contains_all", keywords=["k"])
    assert control_outcome(t, "pass", "fail", True, 1, None) == "collapsed"


def test_control_no_drop_flags_invalid():
    t = _task("contains_all", keywords=["k"])
    assert control_outcome(t, "pass", "pass", True, 1, None) == "no_drop"


def test_control_inconclusive_without_sentinel():
    t = _task("contains_all", keywords=["k"])
    assert control_outcome(t, "pass", "fail", False, 0, None) == "inconclusive"


def test_isolation_control_exempt_from_drop_rule():
    t = _isolation_task()
    # Pass under blinded reads is correct for isolation, not a defect.
    assert control_outcome(t, "pass", "pass", True, 1, True) == "collapsed"
    assert control_outcome(t, "pass", "pass", False, 1, True) == "inconclusive"


def test_normalize():
    assert normalize("  A  B\nC ") == "a b c"
