"""Markdown report renderer (R12, R23).

Reads a results JSONL (one record per task x condition x variant cell) and
renders the three-way comparison, per-task control outcomes, and the
disclosure block. A condition that could not start renders "not run", never
a 0% row (R22).
"""
from __future__ import annotations

import json
from pathlib import Path

CONDITIONS = ["none", "memlawb", "signet"]

CONTROL_LABELS = {
    "collapsed": "collapsed",
    "no_drop": "INVALID (no drop)",
    "inconclusive": "inconclusive",
    "none": "-",
}


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _cell_map(records: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Last record wins per (task, condition, variant), so a resumed run's
    re-executed error cell replaces its earlier error record."""
    cells: dict[tuple[str, str, str], dict] = {}
    for r in records:
        cells[(r["task_id"], r["condition"], r["variant"])] = r
    return cells


def render_report(records: list[dict], meta: dict | None = None) -> str:
    cells = _cell_map(records)
    task_ids = sorted({r["task_id"] for r in records})
    conditions = [c for c in CONDITIONS if any(r["condition"] == c for r in records)]

    lines = ["# memval report", ""]
    if meta:
        bits = [
            f"model: `{meta['model']}`" if meta.get("model") else None,
            f"generated: {meta['generated_at']}" if meta.get("generated_at") else None,
            "fixture upstream" if meta.get("fixture") else None,
        ]
        lines.append(" | ".join(b for b in bits if b))
        lines.append("")

    # Three-way live table.
    header = "| task | " + " | ".join(conditions) + " |"
    lines += [header, "|" + "---|" * (len(conditions) + 1)]
    totals = {c: [0, 0] for c in conditions}  # [passed, scored]
    for tid in task_ids:
        row = [f"`{tid}`"]
        for c in conditions:
            r = cells.get((tid, c, "live"))
            if r is None:
                row.append("-")
            elif r["outcome"] in ("pass", "fail"):
                totals[c][1] += 1
                if r["outcome"] == "pass":
                    totals[c][0] += 1
                row.append(r["outcome"])
            else:
                row.append(r["outcome"])  # error / not_run never count as fail
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Totals", ""]
    for c in conditions:
        ran = totals[c][1]
        not_run = sum(
            1 for tid in task_ids
            if cells.get((tid, c, "live"), {}).get("outcome") in ("not_run", "error")
        )
        if ran == 0:
            label = "not run" if not_run else "no cells"
            lines.append(f"- **{c}**: {label}")
        else:
            lines.append(
                f"- **{c}**: {totals[c][0]}/{ran} passed"
                + (f" ({not_run} cells not run/error)" if not_run else "")
            )

    # Controls.
    lines += ["", "## Controls (retrieval disabled, writes verified)", ""]
    lines += ["| task | condition | control | witness |", "|---|---|---|---|"]
    for tid in task_ids:
        for c in conditions:
            if c == "none":
                continue
            r = cells.get((tid, c, "control"))
            if r is None:
                continue
            outcome = CONTROL_LABELS.get(r.get("control_outcome", ""), r.get("control_outcome", "?"))
            witness = r.get("witness_ok")
            witness_s = {True: "ok", False: "FAILED", None: "-"}.get(witness, "-")
            lines.append(f"| `{tid}` | {c} | {outcome} | {witness_s} |")

    # Tool-call witness: a memory condition with zero tool calls did not
    # actually exercise the backend.
    lines += ["", "## Tool-call witnesses", ""]
    for c in conditions:
        if c == "none":
            continue
        n = sum(
            r.get("tool_calls", 0)
            for (t, cc, v), r in cells.items()
            if cc == c and v == "live"
        )
        lines.append(f"- {c}: {n} tool calls across live cells")

    lines += [
        "",
        "## Disclosures",
        "",
        "- Tool surfaces are native per condition: memlawb advertises save/recall/search/list/delete; signet advertises read tools only, with capture done harness-side by `signet learn` at each session boundary. Backend guide text is filtered to the advertised surface.",
        "- Write paths differ by design: memlawb saves are agent-discretionary; signet captures are automatic at the boundary. This asymmetry is the design difference being measured.",
        "- Isolation tasks measure default-surface leakage only: the second fictional user's session is never told the first user's scope name, so backend options that address a sibling scope by name are untested.",
        "- `contains_none` is scored over every assistant message in the closing leg; `exact`/`contains_all` over the final message.",
        "",
    ]
    return "\n".join(lines)
