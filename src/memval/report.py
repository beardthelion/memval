"""Markdown report renderer (R12, R23).

Reads a results JSONL (one record per task x condition x variant cell) and
renders the three-way comparison, per-task control outcomes, and the
disclosure block. A condition that could not start renders "not run", never
a 0% row (R22).

Control outcomes are recomputed here from the evidence fields stored on
each record rather than trusting the value frozen at emit time, so a
resumed run whose live cell re-executed cannot leave a stale verdict in
the table.
"""
from __future__ import annotations

from .judge import CONFAB_SUSPECT_P, FLAG_INVALID_RATE, QUESTIONS
from .records import (
    cell_key,
    is_error_record,
    is_judged_record,
    load_judge_records,
    load_records,
    partition_flagged,
)
from .score import control_outcome

__all__ = [
    "CONDITIONS",
    "CONTROL_LABELS",
    "load_judge_records",
    "load_records",
    "render_report",
]

CONDITIONS = ["none", "memlawb", "signet"]

CONTROL_LABELS = {
    "collapsed": "collapsed",
    "no_drop": "INVALID (no drop)",
    "inconclusive": "inconclusive",
    "none": "-",
}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "-"


def _judge_value(rec: dict, adjudications: dict, question: str, field: str):
    """A cell's value for one question: the human label when adjudicated,
    else the Jev answer."""
    adj = adjudications.get(cell_key(rec))
    if adj is not None:
        human = adj.get(f"human_{field if question == 'task_success' else question}")
        if human is not None:
            return 1.0 if human is True else 0.0 if human is False else float(human)
    return rec["answers"][question][field]


def _judge_section(judge_records: list[dict], results: list[dict]) -> list[str]:
    """Render the judge analysis block from sidecar records (R8).

    Flagged cells are excluded from the means; a run whose flag rate exceeds
    15% of judgeable cells is headed invalid per the uncertainty policy
    (R6). Judge output is advisory -- it never feeds back into the
    mechanical table above.
    """
    # Only records matching the current bundle's question set aggregate;
    # stale-generation records (a rubric/bundle change mid-sidecar) are
    # counted and skipped so they can never KeyError the section.
    judged = [
        r
        for r in judge_records
        if is_judged_record(r) and all(q in r["answers"] for q in QUESTIONS)
    ]
    stale = sum(
        1 for r in judge_records if is_judged_record(r)
    ) - len(judged)
    judged_keys = {cell_key(r) for r in judged}
    errors = len(
        {
            cell_key(r)
            for r in judge_records
            if is_error_record(r) and cell_key(r) not in judged_keys
        }
    )
    flagged, clean = partition_flagged(judged)
    adjudications = {
        cell_key(r): r for r in judge_records if r.get("adjudicated")
    }

    judgeable = {
        cell_key(r) for r in results if r.get("outcome") in ("pass", "fail")
    }
    denominator = len(judgeable) if judgeable else len(judged)
    flag_rate = len(flagged) / denominator if denominator else 0.0
    invalid = flag_rate > FLAG_INVALID_RATE

    heading = "## Judge analysis (Jev)"
    if invalid:
        heading += f" -- INVALID: {flag_rate:.0%} of judgeable cells flagged (>15%)"
    lines = ["", heading, ""]
    if len(judged) < denominator:
        lines += [
            f"*coverage: {len(judged)} of {denominator} judgeable cells "
            "scored; the rest hit judge errors*",
            "",
        ]

    conditions = sorted(
        {r["condition"] for r in clean} - {"none"},
        key=lambda c: CONDITIONS.index(c) if c in CONDITIONS else len(CONDITIONS),
    )
    if conditions:
        for title, question, field in (
            ("Memory-use probability (live vs control)", "memory_used", "probability"),
            ("Task-success score, 0-4 (live vs control)", "task_success", "score"),
        ):
            lines += [
                f"### {title}",
                "",
                "| condition | live | control |",
                "|---|---|---|",
            ]
            for c in conditions:
                live = _mean(
                    [
                        _judge_value(r, adjudications, question, field)
                        for r in clean
                        if r["condition"] == c and r["variant"] == "live"
                    ]
                )
                ctrl = _mean(
                    [
                        _judge_value(r, adjudications, question, field)
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
        if _judge_value(r, adjudications, "confabulated", "probability")
        > CONFAB_SUSPECT_P
    ]
    if suspects:
        lines += ["### Confabulation suspects", ""]
        for r in suspects:
            p = _judge_value(r, adjudications, "confabulated", "probability")
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
    footer = (
        f"- {len(judged)} cells judged, {len(flagged)} flagged "
        f"({flag_rate:.0%}), {errors} unresolved judge errors"
    )
    if stale:
        footer += f", {stale} records skipped (stale bundle version)"
    lines += [
        footer,
        f"- judge spend: {in_tok} input / {out_tok} output tokens",
    ]
    return lines


def _cell_map(records: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Last record wins per (task, condition, variant), so a resumed run's
    re-executed error cell replaces its earlier error record."""
    cells: dict[tuple[str, str, str], dict] = {}
    for r in records:
        cells[cell_key(r)] = r
    return cells


def _control_verdict(
    r: dict, live: dict | None
) -> str:
    """Recompute the control classification from stored evidence so a
    resumed run cannot leave a stale verdict (and a not_run cell renders its
    own outcome). Falls back to the emit-time value for old records that
    lack task_type."""
    outcome = r.get("outcome", "-")
    # A control cell that never ran reports its own outcome, not a verdict
    # about evidence it never produced.
    if "task_type" not in r or outcome in ("not_run", "error"):
        return r.get("control_outcome") or outcome
    live_outcome = (live or {}).get("outcome", "missing")
    return control_outcome(
        r["task_type"],
        live_outcome,
        r.get("outcome", ""),
        bool(r.get("sentinel_landed")),
        int(r.get("read_calls_blinded") or 0),
        r.get("witness_ok"),
    )


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
            live = cells.get((tid, c, "live"))
            verdict = _control_verdict(r, live)
            outcome = CONTROL_LABELS.get(verdict, verdict if verdict else "-")
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
        "- Tool surfaces are native per condition: memlawb advertises save/recall/search/list/delete; signet advertises read tools only, with capture done harness-side by `signet learn` at each session boundary. Backend guide text and server instructions are filtered to the advertised surface; calls to unadvertised tools return a tool error.",
        "- Write paths differ by design: memlawb saves are agent-discretionary; signet captures are automatic at the boundary. This asymmetry is the design difference being measured.",
        "- Isolation tasks measure default-surface leakage only: the second fictional user's session is never told the first user's scope name, so backend options that address a sibling scope by name are untested.",
        "- `contains_none` is scored over every assistant message in the closing leg; `exact`/`contains_all` over the final message. `contains_all` keywords match at a left token boundary with no trailing digit ('12pm' cannot satisfy '2pm'; '9:30am' still satisfies '9:30').",
        "",
    ]
    if judge_records:
        lines += _judge_section(judge_records, records)
    return "\n".join(lines)
