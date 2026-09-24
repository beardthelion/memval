"""Shared record and artifact-path helpers.

One module owns the shapes every command agrees on: the results JSONL
record, the judge sidecar record, the calibration worksheet trio, and the
transcript filenames the writer (agent.py) and readers (judge.py,
calibrate.py) must spell identically. Keeping them here rather than in
report.py breaks the judge<->report import cycle and stops each consumer
re-implementing the same parsing loop.

Stdlib only.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None


def load_records(path: Path) -> list[dict]:
    """Read a JSONL file; warn and skip malformed lines.

    Results files and sidecars are append-only, so damage realistically
    lands on the trailing line (torn write after a kill mid-flush, a
    concurrent reader catching a partial append). One bad line must not
    kill resume, report, judge, or calibrate.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"warning: {path}:{i}: skipping malformed JSONL line",
                    file=sys.stderr,
                )
    return records


def cell_key(rec: dict) -> tuple[str, str, str]:
    """The (task, condition, variant) triple identifying one scored cell."""
    return (rec["task_id"], rec["condition"], rec["variant"])


def sidecar_path(results_path: Path) -> Path:
    """The judge sidecar for a results file: ``foo.jsonl`` ->
    ``foo.judge.jsonl``."""
    return results_path.with_suffix(".judge.jsonl")


def transcripts_dir(results_path: Path) -> Path:
    """Per-cell transcripts persist beside the results file so the judge can
    run after the run root is gone."""
    return results_path.with_suffix(".transcripts")


def worksheet_path(results_path: Path) -> Path:
    return results_path.with_suffix(".calibration.jsonl")


def calibration_map_path(results_path: Path) -> Path:
    return results_path.with_suffix(".calibration.map.json")


def calibration_transcripts_dir(results_path: Path) -> Path:
    return results_path.with_suffix(".calibration-transcripts")


def load_judge_records(results_path: Path) -> list[dict] | None:
    """The sidecar for a results file, or None when no judge pass ran."""
    sidecar = sidecar_path(Path(results_path))
    if not sidecar.exists():
        return None
    return load_records(sidecar)


_LEG_RE = re.compile(r"-leg(\d+)\.txt$")


def leg_filename(task_id: str, condition: str, variant: str, leg_index: int) -> str:
    """The one place the transcript filename format lives."""
    return f"{task_id}-{condition}-{variant}-leg{leg_index}.txt"


def _leg_no(path: Path) -> int:
    m = _LEG_RE.search(path.name)
    return int(m.group(1)) if m else -1


def cell_leg_paths(transcripts_dir: Path, task_id: str, condition: str,
                   variant: str) -> list[Path]:
    """A cell's per-leg transcript paths in numeric leg order (leg10 sorts
    after leg2, not before)."""
    return sorted(
        transcripts_dir.glob(leg_filename(task_id, condition, variant, "*")),
        key=_leg_no,
    )


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def is_judged_record(rec: dict) -> bool:
    """Sidecar record carrying actual judge answers (not an error or an
    adjudication write-back)."""
    return "answers" in rec


def is_error_record(rec: dict) -> bool:
    return "judge_error" in rec


def adjudicated_keys(records: list[dict]) -> set[tuple[str, str, str]]:
    return {cell_key(r) for r in records if r.get("adjudicated")}


def partition_flagged(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split judged sidecar records into (flagged, clean).

    Adjudicated cells count as clean: a human label cleared the flag. All
    three consumers (report means, worksheet selection, adjudication
    write-back) share this partition so it cannot drift again.
    """
    adjudicated = adjudicated_keys(records)
    flagged, clean = [], []
    for r in records:
        if not is_judged_record(r):
            continue
        if r.get("flags") and cell_key(r) not in adjudicated:
            flagged.append(r)
        else:
            clean.append(r)
    return flagged, clean


@contextlib.contextmanager
def locked(path: Path):
    """Hold an advisory lock covering a read-resume-set + append cycle.

    The lock lives on a sibling `<path>.lock` file so a second `run`,
    `judge`, or `calibrate --labels` on the same results file fails fast
    instead of racing the resume scan and double-appending records.
    POSIX-only; on platforms without fcntl this is a no-op.
    """
    if fcntl is None:
        yield
        return
    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"another memval process holds {lock_path}; "
                "wait for it to finish or remove the lock file if it died"
            )
        yield
    finally:
        os.close(fd)
