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


def load_judge_records(results_path: Path) -> list[dict] | None:
    """The sidecar for a results file, or None when no judge pass ran."""
    from .judge import sidecar_path

    sidecar = sidecar_path(Path(results_path))
    if not sidecar.exists():
        return None
    return load_records(sidecar)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "-"


def _judge_section(judge_records: list[dict]) -> list[str]:
    """Render the judge analysis block from sidecar records (R8).

    Flagged cells are excluded from the means; a run whose flag rate exceeds
    15% is headed invalid per the uncertainty policy (R6). Judge output is
    advisory -- it never feeds back into the mechanical table above.
    """
    judged = [
        r for r in judge_records if "judge_error" not in r and "answers" in r
    ]
    errors = len(judge_records) - len(judged)
    adjudicated = {
        (r["task_id"], r["condition"], r["variant"])
        for r in judge_records
        if r.get("adjudicated")
    }
    flagged = [
        r
        for r in judged
        if r.get("flags")
        and (r["task_id"], r["condition"], r["variant"]) not in adjudicated
    ]
    clean = [r for r in judged if r not in flagged]
    flag_rate = len(flagged) / len(judged) if judged else 0.0
    invalid = flag_rate > 0.15

    heading = "## Judge analysis (Jev)"
    if invalid:
        heading += f" -- INVALID: {flag_rate:.0%} of cells flagged (>15%)"
    lines = ["", heading, ""]

    conditions = sorted(
        {r["condition"] for r in clean} - {"none"}, key=CONDITIONS.index
    )
    if conditions:
        lines += [
            "### Memory-use probability (live vs control)",
            "",
            "| condition | live | control |",
            "|---|---|---|",
        ]
        for c in conditions:
            live = _mean(
                [
                    r["answers"]["memory_used"]["probability"]
                    for r in clean
                    if r["condition"] == c and r["variant"] == "live"
                ]
            )
            ctrl = _mean(
                [
                    r["answers"]["memory_used"]["probability"]
                    for r in clean
                    if r["condition"] == c and r["variant"] == "control"
                ]
            )
            lines.append(f"| {c} | {_fmt(live)} | {_fmt(ctrl)} |")
        lines += [
            "",
            "### Task-success score, 0-4 (live vs control)",
            "",
            "| condition | live | control |",
            "|---|---|---|",
        ]
        for c in conditions:
            live = _mean(
                [
                    r["answers"]["task_success"]["score"]
                    for r in clean
                    if r["condition"] == c and r["variant"] == "live"
                ]
            )
            ctrl = _mean(
                [
                    r["answers"]["task_success"]["score"]
                    for r in clean
                    if r["condition"] == c and r["variant"] == "control"
                ]
            )
            lines.append(f"| {c} | {_fmt(live)} | {_fmt(ctrl)} |")
        lines.append("")

    counts: dict[str, int] = {}
    for r in clean:
        cls = r["answers"]["failure_class"]["choice"]
        counts[cls] = counts.get(cls, 0) + 1
    if counts:
        lines += ["### Failure classes", ""]
        for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {cls}: {n}")
        lines.append("")

    suspects = [
        r
        for r in clean
        if r["answers"]["confabulated"]["probability"] > 0.65
    ]
    if suspects:
        lines += ["### Confabulation suspects", ""]
        for r in suspects:
            p = r["answers"]["confabulated"]["probability"]
            lines.append(
                f"- `{r['task_id']}` ({r['condition']}, {r['variant']}): {p:.2f}"
            )
        lines.append("")

    in_tok = sum(
        (r.get("usage") or {}).get("input_tokens", 0) for r in judge_records
    )
    out_tok = sum(
        (r.get("usage") or {}).get("output_tokens", 0) for r in judge_records
    )
    lines += [
        f"- {len(judged)} cells judged, {len(flagged)} flagged "
        f"({flag_rate:.0%}), {errors} judge errors",
        f"- judge spend: {in_tok} input / {out_tok} output tokens",
    ]
    return lines


def _cell_map(records: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Last record wins per (task, condition, variant), so a resumed run's
    re-executed error cell replaces its earlier error record."""
    cells: dict[tuple[str, str, str], dict] = {}
    for r in records:
        cells[(r["task_id"], r["condition"], r["variant"])] = r
    return cells


def render_report(
    records: list[dict],
    meta: dict | None = None,
    judge_records: list[dict] | None = None,
) -> str:
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
    if judge_records:
        lines += _judge_section(judge_records)
    return "\n".join(lines)
