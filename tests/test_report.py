import json

from memval.report import render_report


def _rec(task, cond, variant, outcome, **extra):
    return {
        "task_id": task,
        "condition": cond,
        "variant": variant,
        "outcome": outcome,
        "tool_calls": 3,
        "read_calls_blinded": 1,
        "sentinel_landed": True,
        "witness_ok": None,
        "control_outcome": "collapsed",
        "detail": "",
        "model": "fake",
        "seed": 1,
        **extra,
    }


def test_three_way_table_and_totals():
    records = [
        _rec("t1", "none", "live", "fail"),
        _rec("t1", "memlawb", "live", "pass"),
        _rec("t1", "signet", "live", "pass"),
        _rec("t1", "memlawb", "control", "fail"),
        _rec("t1", "signet", "control", "fail"),
    ]
    report = render_report(records, {"model": "fake"})
    assert "| `t1` | fail | pass | pass |" in report
    assert "**memlawb**: 1/1 passed" in report
    assert "collapsed" in report


def test_not_run_never_reads_as_zero():
    records = [
        _rec("t1", "none", "live", "fail"),
        _rec("t1", "signet", "live", "not_run", detail="store refused"),
        _rec("t1", "signet", "control", "not_run"),
    ]
    report = render_report(records)
    assert "**signet**: not run" in report
    # A condition that never ran must not render as a zero pass rate.
    assert "**signet**: 0/" not in report


def test_invalid_flag_by_name():
    records = [
        _rec("t-bad", "memlawb", "live", "pass"),
        _rec("t-bad", "memlawb", "control", "pass", control_outcome="no_drop"),
    ]
    report = render_report(records)
    assert "t-bad" in report and "INVALID" in report


def test_inconclusive_control():
    records = [
        _rec("t1", "memlawb", "control", "fail", control_outcome="inconclusive", sentinel_landed=False),
    ]
    assert "inconclusive" in render_report(records)


def test_resume_last_record_wins():
    records = [
        _rec("t1", "none", "live", "error"),
        _rec("t1", "none", "live", "pass"),
    ]
    report = render_report(records)
    assert "**none**: 1/1 passed" in report


def test_fixture_marker():
    report = render_report([], {"fixture": True, "model": "fake"})
    assert "fixture upstream" in report
