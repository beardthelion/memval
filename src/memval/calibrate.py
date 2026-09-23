"""Calibration worksheet emission and agreement scoring (R6, R7).

Two halves: ``emit_worksheet`` writes a blinded human-labeling worksheet
for a stratified sample of judged cells plus every flagged cell (so flags
get adjudicated rather than silently excluded), and ``score_labels`` joins
a filled worksheet back to the sidecar to measure agreement, per-condition
bias, confidence reliability, and confabulation precision/recall. Labeled
flagged cells get an adjudication record appended to the sidecar, which
clears the flag in the report.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .judge import bundle_version, sidecar_path
from .report import load_records

AGREEMENT_BAR = 0.8


def _judge_records(results_path: Path) -> list[dict]:
    sidecar = sidecar_path(Path(results_path))
    if not sidecar.exists():
        raise FileNotFoundError(
            f"no judge sidecar at {sidecar}; run `memval judge` first"
        )
    return load_records(sidecar)


def _stratified_sample(records: list[dict], n: int) -> list[dict]:
    """Deterministic stratified sample: proportional allocation across
    (condition, variant) buckets, evenly spaced picks across task ids."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for r in sorted(records, key=lambda r: (r["task_id"], r["condition"], r["variant"])):
        buckets.setdefault((r["condition"], r["variant"]), []).append(r)
    total = len(records)
    picked = []
    for key in sorted(buckets):
        bucket = buckets[key]
        quota = max(1, round(n * len(bucket) / total)) if total else 0
        if quota >= len(bucket):
            picked.extend(bucket)
        else:
            step = len(bucket) / quota
            picked.extend(bucket[int(i * step)] for i in range(quota))
    return picked


def emit_worksheet(results_path: Path, sample: int = 30) -> Path:
    """Write a blinded labeling worksheet and its harness-side join map.

    Labeler rows carry only an opaque row id, the rubric, and transcripts
    copied to neutral names, so the human cannot see the condition while
    the per-condition bias metric still joins correctly.
    """
    results_path = Path(results_path)
    all_records = _judge_records(results_path)
    adjudicated = {
        (r["task_id"], r["condition"], r["variant"])
        for r in all_records
        if r.get("adjudicated")
    }
    records = [r for r in all_records if "answers" in r]
    if not records:
        raise FileNotFoundError(f"no judged cells in sidecar for {results_path}")

    flagged = [
        r
        for r in records
        if r.get("flags")
        and (r["task_id"], r["condition"], r["variant"]) not in adjudicated
    ]
    unflagged = [r for r in records if r not in flagged]
    selected = flagged + _stratified_sample(unflagged, sample)

    transcripts_dir = results_path.with_suffix(".transcripts")
    neutral_dir = results_path.with_suffix(".calibration-transcripts")
    neutral_dir.mkdir(exist_ok=True)
    worksheet_path = results_path.with_suffix(".calibration.jsonl")
    map_path = results_path.with_suffix(".calibration.map.json")

    from .judge import RUBRIC

    join_map: dict[str, dict] = {}
    out = open(worksheet_path, "w", encoding="utf-8")
    try:
        for i, rec in enumerate(selected):
            row_id = f"row-{i:03d}"
            key = (rec["task_id"], rec["condition"], rec["variant"])
            legs = sorted(
                transcripts_dir.glob(
                    f"{rec['task_id']}-{rec['condition']}-{rec['variant']}-leg*.txt"
                )
            )
            neutral_paths = []
            for j, leg in enumerate(legs):
                dest = neutral_dir / f"{row_id}-leg{j}.txt"
                dest.write_text(leg.read_text(encoding="utf-8"), encoding="utf-8")
                neutral_paths.append(str(dest))
            join_map[row_id] = {
                "task_id": key[0],
                "condition": key[1],
                "variant": key[2],
            }
            out.write(
                json.dumps(
                    {
                        "row_id": row_id,
                        "transcripts": neutral_paths,
                        "rubric": RUBRIC,
                        "flagged": bool(rec.get("flags")),
                        "human_score": None,
                        "human_memory_used": None,
                        "human_confabulated": None,
                    }
                )
                + "\n"
            )
    finally:
        out.close()
    map_path.write_text(json.dumps(join_map, indent=2))
    return worksheet_path


def score_labels(results_path: Path, labels_path: Path) -> dict:
    """Join a filled worksheet against the sidecar and score agreement.

    Returns a metrics dict; flagged cells with labels also append an
    adjudication record to the sidecar so they re-enter judge metrics.
    """
    results_path = Path(results_path)
    labels_path = Path(labels_path)
    map_path = results_path.with_suffix(".calibration.map.json")
    join_map = json.loads(map_path.read_text())

    records = _judge_records(results_path)
    by_key = {}
    for r in records:
        if "answers" in r:
            by_key[(r["task_id"], r["condition"], r["variant"])] = r

    labeled, missing, invalid = [], [], []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score_v = row.get("human_score")
        mem = row.get("human_memory_used")
        confab = row.get("human_confabulated")
        if score_v is None and mem is None and confab is None:
            missing.append(row)
            continue
        if (
            not isinstance(score_v, (int, float))
            or not 0 <= score_v <= 4
            or not isinstance(mem, bool)
            or not isinstance(confab, bool)
        ):
            invalid.append(row)
            continue
        cell = join_map.get(row["row_id"])
        rec = cell and by_key.get(
            (cell["task_id"], cell["condition"], cell["variant"])
        )
        if rec is None:
            invalid.append(row)
            continue
        labeled.append((row, cell, rec))

    n = len(labeled)
    agree = sum(
        1
        for row, _c, r in labeled
        if abs(row["human_score"] - r["answers"]["task_success"]["score"]) <= 1
    )
    bias: dict[str, list[float]] = {}
    for row, cell, r in labeled:
        bias.setdefault(cell["condition"], []).append(
            r["answers"]["task_success"]["score"] - row["human_score"]
        )

    # Confidence reliability: per bucket, does accuracy track confidence?
    buckets: dict[str, list[tuple[float, bool]]] = {}
    for row, _c, r in labeled:
        conf = r["answers"]["task_success"]["confidence"]
        ok = abs(row["human_score"] - r["answers"]["task_success"]["score"]) <= 1
        name = (
            "0.0-0.6" if conf < 0.6 else "0.6-0.8" if conf < 0.8 else "0.8-1.0"
        )
        buckets.setdefault(name, []).append((conf, ok))
    reliability = {
        name: {
            "n": len(vals),
            "mean_confidence": sum(c for c, _ in vals) / len(vals),
            "accuracy": sum(1 for _, ok in vals if ok) / len(vals),
        }
        for name, vals in sorted(buckets.items())
    }

    # Confabulation precision/recall and memory-use accuracy.
    tp = fp = fn = tn = mem_ok = 0
    for row, _c, r in labeled:
        pred = r["answers"]["confabulated"]["probability"] > 0.65
        actual = row["human_confabulated"]
        if pred and actual:
            tp += 1
        elif pred:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
        pred_mem = r["answers"]["memory_used"]["probability"] > 0.5
        if pred_mem == row["human_memory_used"]:
            mem_ok += 1

    # Adjudication write-back: labeled flagged cells get their flag cleared.
    flagged_keys = {
        (r["task_id"], r["condition"], r["variant"])
        for r in records
        if r.get("flags")
    }
    adjudicated = 0
    sidecar = sidecar_path(results_path)
    with open(sidecar, "a", encoding="utf-8") as out:
        for row, cell, _r in labeled:
            key = (cell["task_id"], cell["condition"], cell["variant"])
            if key not in flagged_keys:
                continue
            out.write(
                json.dumps(
                    {
                        "task_id": key[0],
                        "condition": key[1],
                        "variant": key[2],
                        "adjudicated": True,
                        "human_score": row["human_score"],
                        "human_memory_used": row["human_memory_used"],
                        "human_confabulated": row["human_confabulated"],
                        "bundle_version": bundle_version(),
                        "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                )
                + "\n"
            )
            adjudicated += 1

    agreement = agree / n if n else 0.0
    return {
        "labeled": n,
        "missing": len(missing),
        "invalid": len(invalid),
        "adjudicated": adjudicated,
        "agreement": agreement,
        "bias": {c: sum(v) / len(v) for c, v in bias.items()},
        "reliability": reliability,
        "confabulation": {
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        },
        "memory_used_accuracy": mem_ok / n if n else None,
        "bar": AGREEMENT_BAR,
        "passed": agreement >= AGREEMENT_BAR,
    }
