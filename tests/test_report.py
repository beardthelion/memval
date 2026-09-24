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


# --- U4: judge section -----------------------------------------------------

from memval.report import load_judge_records


def _judge_rec(task_id, condition, variant, mem_p=0.9, score_v=3.5,
               confab_p=0.1, cls="no-failure", flags=None):
    return {
        "task_id": task_id,
        "condition": condition,
        "variant": variant,
        "model": "jev-latest",
        "bundle_version": "abc123",
        "usage": {"input_tokens": 1000, "output_tokens": 10},
        "answers": {
            "task_success": {"score": score_v, "confidence": 0.9},
            "memory_used": {"probability": mem_p},
            "confabulated": {"probability": confab_p},
            "failure_class": {"choice": cls, "confidence": 0.9},
        },
        "flags": flags or [],
    }


def test_judge_section_renders_separation(tmp_path):
    records = [
        {"task_id": "t1", "condition": "memlawb", "variant": "live", "outcome": "pass"},
        {"task_id": "t1", "condition": "memlawb", "variant": "control", "outcome": "fail"},
    ]
    judge = [
        _judge_rec("t1", "memlawb", "live", mem_p=0.9),
        _judge_rec("t1", "memlawb", "control", mem_p=0.1),
    ]
    out = render_report(records, judge_records=judge)
    assert "## Judge analysis (Jev)" in out
    assert "Memory-use probability" in out
    assert "| memlawb | 0.90 | 0.10 |" in out
    assert "no-failure: 2" in out
    assert "judge spend: 2000 input" in out


def test_report_without_sidecar_is_unchanged(tmp_path):
    records = [
        {"task_id": "t1", "condition": "none", "variant": "live", "outcome": "pass"},
    ]
    with_judge = render_report(records)
    without = render_report(records, judge_records=None)
    assert with_judge == without
    assert "Judge" not in without


def test_load_judge_records_absent_and_present(tmp_path):
    results = tmp_path / "run.jsonl"
    results.write_text("{}\n")
    assert load_judge_records(results) is None
    sidecar = tmp_path / "run.judge.jsonl"
    sidecar.write_text(json.dumps(_judge_rec("t1", "memlawb", "live")) + "\n")
    assert len(load_judge_records(results)) == 1


def test_flagged_cells_excluded_from_means(tmp_path):
    judge = [
        _judge_rec("t1", "memlawb", "live", mem_p=0.9),
        _judge_rec("t2", "memlawb", "live", mem_p=0.1, flags=["memory_used"]),
        _judge_rec("t1", "memlawb", "control", mem_p=0.2),
    ]
    records = [
        {"task_id": "t1", "condition": "memlawb", "variant": "live", "outcome": "pass"},
        {"task_id": "t2", "condition": "memlawb", "variant": "live", "outcome": "pass"},
        {"task_id": "t1", "condition": "memlawb", "variant": "control", "outcome": "fail"},
    ]
    out = render_report(records, judge_records=judge)
    # The flagged t2 cell is excluded: live mean is 0.90, not 0.50.
    assert "| memlawb | 0.90 | 0.20 |" in out
    assert "1 flagged" in out


def test_judge_section_invalid_over_flag_threshold(tmp_path):
    judge = [
        _judge_rec(f"t{i}", "memlawb", "live", flags=["task_success"])
        for i in range(3)
    ] + [_judge_rec(f"t{i}", "memlawb", "live") for i in range(3, 8)]
    out = render_report([], judge_records=judge)
    assert "INVALID" in out


def test_judge_section_skips_absent_conditions(tmp_path):
    judge = [_judge_rec("t1", "signet", "live")]
    out = render_report([], judge_records=judge)
    assert "| signet |" in out
    assert "| memlawb |" not in out


def test_not_run_control_renders_its_outcome():
    records = [
        _rec("t1", "memlawb", "live", "not_run"),
        _rec("t1", "memlawb", "control", "not_run", control_outcome=None),
    ]
    out = render_report(records)
    assert "| `t1` | memlawb | not_run |" in out


def test_control_verdict_recomputed_from_evidence():
    # The stored verdict was frozen when the live sibling had errored; the
    # resumed live pass must flip it to collapsed at report time.
    records = [
        _rec("t1", "memlawb", "live", "pass", task_type="recall"),
        _rec(
            "t1", "memlawb", "control", "fail",
            task_type="recall", control_outcome="inconclusive",
        ),
    ]
    out = render_report(records)
    assert "| `t1` | memlawb | collapsed |" in out


def test_judge_section_tolerates_stale_bundle_records():
    judge = [
        _judge_rec("t1", "memlawb", "live"),
        {
            "task_id": "t2",
            "condition": "memlawb",
            "variant": "live",
            "bundle_version": "old",
            "answers": {"legacy_question": {"score": 1.0}},
        },
    ]
    out = render_report([], judge_records=judge)
    assert "stale bundle" in out


def test_judge_errors_deduped_and_adjudications_not_errors():
    judge = [
        _judge_rec("t1", "memlawb", "live"),
        {"task_id": "t2", "condition": "memlawb", "variant": "live",
         "judge_error": "boom"},
        {"task_id": "t2", "condition": "memlawb", "variant": "live",
         "judge_error": "boom"},
        {"task_id": "t1", "condition": "memlawb", "variant": "live",
         "adjudicated": True, "human_score": 3.0},
    ]
    out = render_report([], judge_records=judge)
    assert "1 unresolved judge errors" in out
