"""Run orchestration: preflight, per-cell execution, resumable JSONL, report.

One record per (task, condition, variant) cell is appended to the results
file as it completes; a re-run on a partial file skips completed cells and
re-executes `error` cells (R21).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .agent import run_cell
from .gateway import load_gateway_config, serve_gateway
from .memory_backends import make_cell_session
from .records import (
    cell_key,
    load_judge_records,
    load_records,
    locked,
    transcripts_dir,
)
from .report import CONDITIONS, render_report
from .score import control_outcome, score
from .supervisor import (
    PreflightError,
    RunRoot,
    find_free_port,
    memlawb_store_argv,
    memlawb_store_env,
    signet_store_argv,
    signet_store_env,
    spawn_store,
)
from .tasks import default_tasks_dir, load_battery

ALL_CONDITIONS = CONDITIONS
# pass/fail are terminal. not_run is deliberately absent: a cell recorded
# while its backend was down must be re-attempted on resume once the blocker
# clears, not frozen forever.
COMPLETED = {"pass", "fail"}


def _completed_cells(path: Path) -> dict[tuple[str, str, str], str]:
    if not path.exists():
        return {}
    done = {cell_key(r): r["outcome"] for r in load_records(path)}
    # error cells re-execute on resume; keep only terminal outcomes.
    return {k: v for k, v in done.items() if v in COMPLETED}


async def run_battery(
    *,
    config_path: Path,
    model: str | None,
    conditions: list[str],
    task_filter: list[str] | None,
    results_path: Path | None,
    memlawb_checkout: Path,
    signet_checkout: Path,
    seed: int | None = None,
    fixture: bool = False,
) -> Path:
    cfg = load_gateway_config(str(config_path))
    if model is None:
        if len(cfg["models"]) != 1:
            raise PreflightError("config has multiple models; pass --model")
        model = next(iter(cfg["models"]))

    tasks = load_battery(default_tasks_dir())
    if task_filter:
        wanted = set(task_filter)
        tasks = [t for t in tasks if t.id in wanted]
        missing = wanted - {t.id for t in tasks}
        if missing:
            raise PreflightError(f"--tasks matched no battery entries: {sorted(missing)}")

    results_path = results_path or Path("results") / f"{int(time.time())}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    run_root = RunRoot()
    log_dir = run_root.log_dir()
    # Transcripts are the one artifact that escapes the run root: a judge
    # pass reads them after the run, so they live beside the results file
    # that names their cells rather than under the cleaned-up temp root.
    legs_dir = transcripts_dir(results_path)
    legs_dir.mkdir(exist_ok=True)

    # Held from the resume scan through the last append so a concurrent
    # `run`/`judge`/`calibrate` on this file fails fast instead of
    # double-appending cells. Entered manually because the battery body
    # already has its own try/finally for teardown.
    results_lock = locked(results_path)
    results_lock.__enter__()

    completed = _completed_cells(results_path)
    if completed:
        # A resumed file carries the earlier invocation's model/seed; mixing
        # generations under one report header would blend two models into
        # one table.
        prior = load_records(results_path)
        prior_models = {r.get("model") for r in prior} - {None}
        prior_seeds = {r.get("seed") for r in prior} - {None}
        if prior_models - {model}:
            print(
                f"warning: resuming a results file recorded under "
                f"model(s) {sorted(prior_models)}; current model is "
                f"{model!r} and both will appear under one report header",
                file=sys.stderr,
            )
        if prior_seeds - {seed}:
            print(
                f"warning: resuming a results file recorded under "
                f"seed(s) {sorted(prior_seeds)}; current seed is {seed!r}",
                file=sys.stderr,
            )
    out = open(results_path, "a", encoding="utf-8")

    gateway = serve_gateway(cfg, port=find_free_port(8799))
    gateway_url = f"http://127.0.0.1:{gateway.server_address[1]}"

    stores = []
    urls: dict[str, str] = {}
    unavailable: dict[str, str] = {}
    try:
        if "memlawb" in conditions:
            try:
                port = find_free_port(8801)
                stores.append(
                    spawn_store(
                        "memlawb-store",
                        memlawb_store_argv(memlawb_checkout),
                        memlawb_store_env(run_root.path / "memlawb-data", port),
                        log_dir,
                        health_url=f"http://127.0.0.1:{port}/health",
                    )
                )
                urls["memlawb"] = f"http://127.0.0.1:{port}"
            except PreflightError as e:
                unavailable["memlawb"] = str(e)
        if "signet" in conditions:
            try:
                port = find_free_port(8802)
                stores.append(
                    spawn_store(
                        "signet-store",
                        signet_store_argv(signet_checkout),
                        signet_store_env(run_root.path / "signet-data", port),
                        log_dir,
                        health_url=f"http://127.0.0.1:{port}/health",
                    )
                )
                urls["signet"] = f"http://127.0.0.1:{port}"
            except PreflightError as e:
                unavailable["signet"] = str(e)

        def emit(record: dict) -> None:
            record.update({"model": model, "seed": seed})
            out.write(json.dumps(record) + "\n")
            out.flush()

        for task in tasks:
            for cond in conditions:
                variants = ["live"] + (["control"] if cond != "none" else [])
                # Seed from completed cells so a resumed control cell still
                # classifies against the earlier live outcome.
                live_outcome = completed.get((task.id, cond, "live"))
                for variant in variants:
                    key = (task.id, cond, variant)
                    if key in completed:
                        continue
                    if cond in unavailable:
                        emit(
                            {
                                "task_id": task.id,
                                "condition": cond,
                                "variant": variant,
                                "outcome": "not_run",
                                "detail": unavailable[cond],
                            }
                        )
                        continue
                    session = make_cell_session(
                        cond,
                        task,
                        control=(variant == "control"),
                        log_dir=log_dir,
                        memlawb_checkout=memlawb_checkout,
                        memlawb_url=urls.get("memlawb", ""),
                        signet_checkout=signet_checkout,
                        signet_url=urls.get("signet", ""),
                        run_root=run_root.path,
                    )
                    res = await run_cell(
                        task,
                        session,
                        gateway_url,
                        model,
                        seed=seed,
                        transcripts_dir=legs_dir,
                        condition=cond,
                        variant=variant,
                    )
                    evidence = res.evidence
                    record: dict = {
                        "task_id": task.id,
                        "task_type": task.type,
                        "condition": cond,
                        "variant": variant,
                        "tool_calls": res.tool_calls,
                        "read_calls_blinded": evidence.read_calls_blinded,
                        "sentinel_landed": evidence.sentinel_landed,
                        "witness_ok": evidence.witness_ok,
                        "guide_injected": evidence.guide_injected,
                        "teardown_error": evidence.teardown_error,
                        "detail": res.detail,
                    }
                    if res.outcome == "error":
                        record["outcome"] = "error"
                    elif res.outcome == "run":
                        record["outcome"] = score(task, res.final_answer, res.closing_messages)
                        record["final_answer"] = res.final_answer[:500]
                    else:
                        record["outcome"] = res.outcome
                    if variant == "live":
                        live_outcome = record["outcome"]
                    else:
                        record["control_outcome"] = control_outcome(
                            task.type,
                            live_outcome or "missing",
                            record["outcome"],
                            evidence.sentinel_landed,
                            evidence.read_calls_blinded,
                            evidence.witness_ok,
                        )
                    emit(record)
    finally:
        out.close()
        results_lock.__exit__(None, None, None)
        for s in stores:
            s.stop()
        gateway.shutdown()
        run_root.cleanup()

    meta = {
        "model": model,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "fixture": fixture,
    }
    report_path = results_path.with_suffix(".md")
    report_path.write_text(
        render_report(
            load_records(results_path),
            meta,
            judge_records=load_judge_records(results_path),
        )
    )
    return report_path
