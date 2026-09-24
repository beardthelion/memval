"""Calibration worksheet emission and agreement scoring (R6, R7).

Two halves: ``emit_worksheet`` writes a blinded human-labeling worksheet
for a stratified sample of judged cells plus every flagged cell (so flags
get adjudicated rather than silently excluded), and ``score_labels`` joins
a filled worksheet back to the sidecar to measure agreement, per-condition
bias, confidence reliability, and confabulation precision/recall. Labeled
flagged cells get an adjudication record appended to the sidecar, which
clears the flag in the report.

The worksheet and its join map carry a generation token: re-emitting the
worksheet rekeys the positional row ids, so a labels file written against
an older worksheet must be rejected rather than silently misjoined to the
wrong cells.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .judge import (
    CONFAB_SUSPECT_P,
    FLAG_MIN_CONFIDENCE,
    RUBRIC,
    blind_transcript,
    bundle_version,
)
from .records import (
    calibration_map_path,
    calibration_transcripts_dir,
    cell_key,
    cell_leg_paths,
    is_judged_record,
    load_judge_records,
    load_records,
    now_iso,
    partition_flagged,
    sidecar_path,
    transcripts_dir,
    worksheet_path,
)

AGREEMENT_BAR = 0.8
AGREE_WITHIN = 1  # a human/jev score pair agrees within one rubric rung


def _judge_records(results_path: Path) -> list[dict]:
    records = load_judge_records(Path(results_path))
    if records is None:
        raise FileNotFoundError(
            f"no judge sidecar at {sidecar_path(Path(results_path))}; "
            "run `memval judge` first"
        )
    return records


def _stratified_sample(records: list[dict], n: int) -> list[dict]:
    """Deterministic stratified sample: proportional allocation across
    (condition, variant) buckets, evenly spaced picks across task ids."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for r in sorted(records, key=cell_key):
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


def _generation(selected: list[dict]) -> str:
    """A token binding a worksheet to its join map: any re-emit over a
    different selection mints a new generation, and score_labels refuses
    labels written against a stale one."""
    keys = sorted(cell_key(r) for r in selected)
    return hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:12]


def emit_worksheet(results_path: Path, sample: int = 30) -> Path:
    """Write a blinded labeling worksheet and its harness-side join map.

    Labeler rows carry only an opaque row id, the rubric, and transcripts
    copied to neutral names, so the human cannot see the condition while
    the per-condition bias metric still joins correctly. The judge's flag
    state is deliberately not on the row: a labeler who knows Jev was
    unsure is not a blind labeler.
    """
    results_path = Path(results_path)
    all_records = _judge_records(results_path)
    flagged, unflagged = partition_flagged(all_records)
    if not flagged and not unflagged:
        raise FileNotFoundError(f"no judged cells in sidecar for {results_path}")

    selected = sorted(flagged + _stratified_sample(unflagged, sample), key=cell_key)
    generation = _generation(selected)

    legs_dir = transcripts_dir(results_path)
    neutral_dir = calibration_transcripts_dir(results_path)
    neutral_dir.mkdir(exist_ok=True)
    for stale in neutral_dir.glob("row-*.txt"):
        stale.unlink()
    ws_path = worksheet_path(results_path)
    map_path = calibration_map_path(results_path)

    join_map: dict[str, dict] = {}
    rows = []
    for i, rec in enumerate(selected):
        row_id = f"row-{i:03d}"
        legs = cell_leg_paths(legs_dir, *cell_key(rec))
        if not legs:
            continue  # a cell with no transcripts cannot be labeled
        neutral_paths = []
        for j, leg in enumerate(legs):
            dest = neutral_dir / f"{row_id}-leg{j}.txt"
            dest.write_text(
                blind_transcript(leg.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
            neutral_paths.append(str(dest))
        join_map[row_id] = dict(
            zip(("task_id", "condition", "variant"), cell_key(rec))
        )
        rows.append(
            {
                "row_id": row_id,
                "generation": generation,
                "transcripts": neutral_paths,
                "rubric": RUBRIC,
                "human_score": None,
                "human_memory_used": None,
                "human_confabulated": None,
            }
        )
    ws_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    # Map writes are atomic so a failed emit never leaves a mismatched pair.
    tmp = map_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"generation": generation, "rows": join_map}, indent=2)
    )
    tmp.replace(map_path)
    return ws_path


def score_labels(results_path: Path, labels_path: Path) -> dict:
    """Join a filled worksheet against the sidecar and score agreement.

    Returns a metrics dict; flagged cells with labels also append an
    adjudication record to the sidecar so they re-enter judge metrics.
    """
    results_path = Path(results_path)
    labels_path = Path(labels_path)
    map_path = calibration_map_path(results_path)
    map_doc = json.loads(map_path.read_text())
    join_map = map_doc.get("rows", map_doc)  # tolerate pre-generation maps
    generation = map_doc.get("generation") if "rows" in map_doc else None

    records = _judge_records(results_path)
    by_key = {cell_key(r): r for r in records if is_judged_record(r)}
    flagged_keys = {cell_key(r) for r in partition_flagged(records)[0]}

    labeled, missing, invalid = [], [], []
    for row in load_records(labels_path):
        score_v = row.get("human_score")
        mem = row.get("human_memory_used")
        confab = row.get("human_confabulated")
        if score_v is None and mem is None and confab is None:
            missing.append(row)
            continue
        cell = join_map.get(row.get("row_id"))
        if (
            cell is None
            or (generation is not None and row.get("generation") != generation)
            or not isinstance(score_v, (int, float))
            or isinstance(score_v, bool)
            or not 0 <= score_v <= 4
            or not isinstance(mem, bool)
            or not isinstance(confab, bool)
        ):
            invalid.append(row)
            continue
        rec = by_key.get(cell_key(cell))
        if rec is None:
            invalid.append(row)
            continue
        labeled.append((row, cell, rec))

    # Agreement, bias, reliability, and confabulation metrics are measured
    # over unflagged cells: flagged rows exist for adjudication, and mixing
    # Jev's low-confidence cells into the agreement metric would skew it.
    measured = [
        (row, cell, r)
        for row, cell, r in labeled
        if cell_key(r) not in flagged_keys
    ]
    n = len(measured)
    agree = sum(
        1
        for row, _c, r in measured
        if abs(row["human_score"] - r["answers"]["task_success"]["score"])
        <= AGREE_WITHIN
    )
    bias: dict[str, list[float]] = {}
    for row, cell, r in measured:
        bias.setdefault(cell["condition"], []).append(
            r["answers"]["task_success"]["score"] - row["human_score"]
        )

    # Confidence reliability: per bucket, does accuracy track confidence?
    buckets: dict[str, list[tuple[float, bool]]] = {}
    for row, _c, r in measured:
        conf = r["answers"]["task_success"]["confidence"]
        ok = (
            abs(row["human_score"] - r["answers"]["task_success"]["score"])
            <= AGREE_WITHIN
        )
        name = (
            f"0.0-{FLAG_MIN_CONFIDENCE}"
            if conf < FLAG_MIN_CONFIDENCE
            else f"{FLAG_MIN_CONFIDENCE}-0.8" if conf < 0.8 else "0.8-1.0"
        )
        buckets.setdefault(name, []).append((conf, ok))
    reliability = {
        name: {
            "n": len(vals),
            "mean_confidence": sum(c for c, _ in vals) / len(vals),
            "accuracy": sum(1 for _, ok in vals) / len(vals),
        }
        for name, vals in sorted(buckets.items())
    }

    # Confabulation precision/recall and memory-use accuracy.
    tp = fp = fn = tn = mem_ok = 0
    for row, _c, r in measured:
        pred = (
            r["answers"]["confabulated"]["probability"] > CONFAB_SUSPECT_P
        )
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
    adjudicated = 0
    version = bundle_version()
    sidecar = sidecar_path(results_path)
    with open(sidecar, "a", encoding="utf-8") as out:
        for row, cell, _r in labeled:
            key = cell_key(cell)
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
                        "bundle_version": version,
                        "judged_at": now_iso(),
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
