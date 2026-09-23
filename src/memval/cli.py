"""memval CLI: run / report / check-isolation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anyio

from .isolation import check_isolation
from .report import load_judge_records, render_report
from .run import ALL_CONDITIONS, run_battery
from .supervisor import PreflightError

DEFAULT_MEMLAWB = Path.home() / "projects" / "memlawb"
DEFAULT_SIGNET = Path.home() / "projects" / "signet"


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memval",
        description="Reproducible eval harness comparing encrypted memory backends.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="replay the battery and render a report")
    run.add_argument("--config", default="memval.config.json",
                     help="gateway config: models -> {base, key_env}")
    run.add_argument("--model", default=None, help="model name from the config")
    run.add_argument("--conditions", default=",".join(ALL_CONDITIONS),
                     help="comma-separated subset of none,memlawb,signet")
    run.add_argument("--tasks", default=None, help="comma-separated task ids")
    run.add_argument("--results", default=None, help="results JSONL to append/resume")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--memlawb-checkout", type=Path, default=DEFAULT_MEMLAWB)
    run.add_argument("--signet-checkout", type=Path, default=DEFAULT_SIGNET)
    run.add_argument("--fixture", action="store_true",
                     help="mark the report as produced against a fake upstream")

    rep = sub.add_parser("report", help="render a report from a results JSONL")
    rep.add_argument("results", type=Path)

    jud = sub.add_parser(
        "judge", help="judge a run's persisted transcripts with the Jev API"
    )
    jud.add_argument("results", type=Path)
    jud.add_argument("--tasks-dir", type=Path, default=None)
    jud.add_argument("--endpoint", default=None,
                     help="override the System One endpoint")

    cal = sub.add_parser(
        "calibrate",
        help="emit a blinded labeling worksheet, or score completed labels",
    )
    cal.add_argument("results", type=Path)
    cal.add_argument("--sample", type=int, default=30)
    cal.add_argument("--labels", type=Path, default=None)

    chk = sub.add_parser("check-isolation", help="enable/disable artifact check per backend")
    chk.add_argument("backend", choices=["memlawb", "signet"])
    chk.add_argument("--memlawb-checkout", type=Path, default=DEFAULT_MEMLAWB)
    chk.add_argument("--signet-checkout", type=Path, default=DEFAULT_SIGNET)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "report":
        records = [
            json.loads(l)
            for l in args.results.read_text().splitlines()
            if l.strip()
        ]
        print(
            render_report(records, judge_records=load_judge_records(args.results))
        )
        return 0
    if args.cmd == "judge":
        from .judge import JEV_ENDPOINT, JevClient, judge_results

        try:
            client = JevClient(endpoint=args.endpoint or JEV_ENDPOINT)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 2
        sidecar = judge_results(args.results, tasks_dir=args.tasks_dir, client=client)
        print(f"judge sidecar: {sidecar}")
        return 0
    if args.cmd == "calibrate":
        from .calibrate import emit_worksheet, score_labels

        try:
            if args.labels:
                metrics = score_labels(args.results, args.labels)
                print(json.dumps(metrics, indent=2))
                print(
                    f"agreement {metrics['agreement']:.2f} vs bar "
                    f"{metrics['bar']} -> {'PASS' if metrics['passed'] else 'FAIL'}"
                )
                return 0 if metrics["passed"] else 1
            ws = emit_worksheet(args.results, sample=args.sample)
            print(f"worksheet: {ws}")
            return 0
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            return 2
    if args.cmd == "check-isolation":
        checkout = args.memlawb_checkout if args.backend == "memlawb" else args.signet_checkout
        result = anyio.run(check_isolation, args.backend, checkout)
        print(f"{result.backend}: {'ok' if result.ok else 'FAILED'} - {result.detail}")
        return 0 if result.ok else 1
    if args.cmd == "run":
        conditions = _split_csv(args.conditions) or ALL_CONDITIONS
        bad = [c for c in conditions if c not in ALL_CONDITIONS]
        if bad:
            print(f"unknown conditions: {bad} (choose from {ALL_CONDITIONS})", file=sys.stderr)
            return 2
        async def _go():
            return await run_battery(
                config_path=Path(args.config),
                model=args.model,
                conditions=conditions,
                task_filter=_split_csv(args.tasks),
                results_path=Path(args.results) if args.results else None,
                memlawb_checkout=args.memlawb_checkout,
                signet_checkout=args.signet_checkout,
                seed=args.seed,
                fixture=args.fixture,
            )

        try:
            report_path = anyio.run(_go)
        except PreflightError as e:
            print(f"preflight failed: {e}", file=sys.stderr)
            return 2
        except FileNotFoundError as e:
            print(f"missing config or file: {e}", file=sys.stderr)
            return 2
        print(f"report: {report_path}")
        transcripts = report_path.with_suffix(".transcripts")
        if transcripts.is_dir():
            print(
                f"transcripts kept at {transcripts}; "
                f"`memval judge {report_path.with_suffix('.jsonl')}` runs the "
                "Jev analysis layer"
            )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
