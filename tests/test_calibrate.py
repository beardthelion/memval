"""Calibration worksheet and agreement scoring (R7).

Worksheets are blinded (opaque row ids, neutral transcript names) and the
join map stays harness-side; scoring joins filled labels against the
sidecar and reports the spec's measures. Labels on flagged cells write an
adjudication record that clears the flag (R6).
"""
import json
from pathlib import Path

import pytest

from memval.calibrate import emit_worksheet, score_labels


def _judge_rec(task_id, condition, variant, score_v=3.0, conf=0.9,
               confab_p=0.1, mem_p=0.9, flags=None):
    return {
        "task_id": task_id,
        "condition": condition,
        "variant": variant,
        "model": "jev-latest",
        "bundle_version": "abc123",
        "usage": {"input_tokens": 100},
        "answers": {
            "task_success": {"score": score_v, "confidence": conf},
            "memory_used": {"probability": mem_p},
            "confabulated": {"probability": confab_p},
            "failure_class": {"choice": "no-failure", "confidence": 0.9},
        },
        "flags": flags or [],
    }


def _seed(tmp_path: Path, records: list[dict]) -> Path:
    results = tmp_path / "run.jsonl"
    results.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": r["task_id"],
                    "condition": r["condition"],
                    "variant": r["variant"],
                    "outcome": "pass",
                }
            )
            + "\n"
            for r in records
        )
    )
    sidecar = tmp_path / "run.judge.jsonl"
    sidecar.write_text("".join(json.dumps(r) + "\n" for r in records))
    transcripts = tmp_path / "run.transcripts"
    transcripts.mkdir()
    for r in records:
        for leg in (0, 1):
            (
                transcripts
                / f"{r['task_id']}-{r['condition']}-{r['variant']}-leg{leg}.txt"
            ).write_text(f"leg {leg} text\n")
    return results


def _fill(worksheet_path: Path, score=3.0, mem=True, confab=False,
          rows=None) -> Path:
    labels = tmp_labels = worksheet_path.with_suffix(".labels.jsonl")
    out = []
    for line in worksheet_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if rows is None or row["row_id"] in rows:
            row["human_score"] = score
            row["human_memory_used"] = mem
            row["human_confabulated"] = confab
        out.append(row)
    labels.write_text("".join(json.dumps(r) + "\n" for r in out))
    return labels


def test_worksheet_is_blinded_and_stratified(tmp_path):
    records = [
        _judge_rec(f"task-{i:02d}", cond, var)
        for i in range(10)
        for cond in ("none", "memlawb", "signet")
        for var in (["live", "control"] if cond != "none" else ["live"])
    ]
    results = _seed(tmp_path, records)
    ws = emit_worksheet(results, sample=30)
    rows = [json.loads(l) for l in ws.read_text().splitlines() if l.strip()]

    assert len(rows) == 30
    for row in rows:
        assert row["row_id"].startswith("row-")
        assert row["human_score"] is None
        assert row["human_memory_used"] is None
        assert row["human_confabulated"] is None
        # Blinded: no condition or task id leaks into the labeler-facing row.
        blob = json.dumps(row)
        for marker in ("memlawb", "signet", "none", "task-", "control", "live"):
            assert marker not in blob, marker
    # Stratification: the join map covers multiple conditions.
    join_map = json.loads(
        (tmp_path / "run.calibration.map.json").read_text()
    )
    conds = {v["condition"] for v in join_map.values()}
    assert conds == {"none", "memlawb", "signet"}


def test_worksheet_includes_flagged_cells_on_top(tmp_path):
    records = [
        _judge_rec(f"task-{i:02d}", "memlawb", "live") for i in range(8)
    ]
    records[0]["flags"] = ["task_success"]
    records[1]["flags"] = ["memory_used"]
    results = _seed(tmp_path, records)
    ws = emit_worksheet(results, sample=4)
    rows = [json.loads(l) for l in ws.read_text().splitlines()]
    join_map = json.loads(
        (tmp_path / "run.calibration.map.json").read_text()
    )
    flagged_rows = [
        join_map[r["row_id"]]["task_id"] for r in rows if r["flagged"]
    ]
    assert sorted(flagged_rows) == ["task-00", "task-01"]
    assert len(rows) == 6  # 2 flagged + 4 sampled


def test_score_labels_agreement_bias_and_reliability(tmp_path):
    # Jev says 3.0; humans say 3.0 for the first half, 1.0 for the second.
    records = [
        _judge_rec(f"task-{i}", "memlawb", "live", score_v=3.0)
        for i in range(10)
    ]
    results = _seed(tmp_path, records)
    ws = emit_worksheet(results, sample=10)
    join_map = json.loads(
        (tmp_path / "run.calibration.map.json").read_text()
    )
    ordered = sorted(join_map)
    labels_rows = []
    for i, line in enumerate(ws.read_text().splitlines()):
        row = json.loads(line)
        row["human_score"] = 3.0 if i % 2 == 0 else 1.0
        row["human_memory_used"] = True
        row["human_confabulated"] = False
        labels_rows.append(row)
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(r) + "\n" for r in labels_rows))

    metrics = score_labels(results, labels)
    assert metrics["labeled"] == 10
    assert metrics["agreement"] == 0.5  # half within 1 rung, half off by 2
    assert metrics["bias"]["memlawb"] > 0  # jev above human on the 1.0 rows
    assert metrics["reliability"]["0.8-1.0"]["n"] == 10
    assert metrics["passed"] is False


def test_score_labels_adjudicates_flagged_cells(tmp_path):
    records = [_judge_rec("task-1", "memlawb", "live", flags=["task_success"])]
    results = _seed(tmp_path, records)
    ws = emit_worksheet(results, sample=1)
    labels = _fill_labels(tmp_path, ws, score=3.0)
    metrics = score_labels(results, labels)
    assert metrics["adjudicated"] == 1
    sidecar = [json.loads(l) for l in
               (tmp_path / "run.judge.jsonl").read_text().splitlines()]
    assert sidecar[-1]["adjudicated"] is True


def _fill_labels(tmp_path, ws, score=3.0):
    rows = []
    for line in ws.read_text().splitlines():
        row = json.loads(line)
        row["human_score"] = score
        row["human_memory_used"] = True
        row["human_confabulated"] = False
        rows.append(row)
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return labels


def test_score_labels_reports_invalid_and_missing(tmp_path):
    records = [_judge_rec(f"task-{i}", "memlawb", "live") for i in range(4)]
    results = _seed(tmp_path, records)
    ws = emit_worksheet(results, sample=4)
    rows = [json.loads(l) for l in ws.read_text().splitlines()]
    rows[0]["human_score"] = 9  # out of range
    rows[1]["human_score"] = 3.0
    rows[1]["human_memory_used"] = True
    rows[1]["human_confabulated"] = False
    # rows[2], rows[3] left blank -> missing
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(r) + "\n" for r in rows))
    metrics = score_labels(results, labels)
    assert metrics["labeled"] == 1
    assert metrics["invalid"] == 1
    assert metrics["missing"] == 2


def test_score_labels_confabulation_precision_recall(tmp_path):
    records = [
        _judge_rec("tp", "memlawb", "live", confab_p=0.9),
        _judge_rec("fp", "memlawb", "live", confab_p=0.9),
        _judge_rec("fn", "memlawb", "live", confab_p=0.1),
        _judge_rec("tn", "memlawb", "live", confab_p=0.1),
    ]
    results = _seed(tmp_path, records)
    ws = emit_worksheet(results, sample=4)
    join_map = json.loads(
        (tmp_path / "run.calibration.map.json").read_text()
    )
    confab_truth = {"tp": True, "fp": False, "fn": True, "tn": False}
    rows = []
    for line in ws.read_text().splitlines():
        row = json.loads(line)
        row["human_score"] = 3.0
        row["human_memory_used"] = True
        row["human_confabulated"] = confab_truth[
            join_map[row["row_id"]]["task_id"]
        ]
        rows.append(row)
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(r) + "\n" for r in rows))
    metrics = score_labels(results, labels)
    assert metrics["confabulation"]["precision"] == pytest.approx(0.5)
    assert metrics["confabulation"]["recall"] == pytest.approx(0.5)


def test_score_labels_passes_above_bar(tmp_path):
    records = [_judge_rec(f"task-{i}", "memlawb", "live") for i in range(10)]
    results = _seed(tmp_path, records)
    ws = emit_worksheet(results, sample=10)
    labels = _fill_labels(tmp_path, ws, score=3.0)  # identical to jev
    metrics = score_labels(results, labels)
    assert metrics["agreement"] == 1.0
    assert metrics["passed"] is True


def test_calibrate_fails_closed_without_sidecar(tmp_path):
    results = tmp_path / "run.jsonl"
    results.write_text("")
    with pytest.raises(FileNotFoundError, match="judge sidecar"):
        emit_worksheet(results)
