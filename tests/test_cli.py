"""CLI-level tests: the fake-upstream subset run is the end-to-end check
that needs no real model key and no real backends."""
import json
import threading
from pathlib import Path

import pytest

from memval.cli import main
from tests.fixtures import fake_upstream


@pytest.fixture
def gw_config(tmp_path):
    server = fake_upstream.serve(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cfg = tmp_path / "memval.config.json"
    cfg.write_text(
        json.dumps(
            {
                "models": {
                    "fake": {
                        "base": f"http://127.0.0.1:{server.server_address[1]}",
                        "key_env": None,
                    }
                }
            }
        )
    )
    yield cfg
    server.shutdown()


def test_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


def test_subset_run_none_completes(gw_config, tmp_path):
    results = tmp_path / "run.jsonl"
    rc = main(
        [
            "run",
            "--config",
            str(gw_config),
            "--model",
            "fake",
            "--conditions",
            "none",
            "--tasks",
            "recall-01",
            "--results",
            str(results),
            "--fixture",
        ]
    )
    assert rc == 0
    records = [json.loads(l) for l in results.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["condition"] == "none"
    assert records[0]["outcome"] in ("pass", "fail")
    # no-memory has no tools: the floor fails closed.
    assert records[0]["tool_calls"] == 0
    report = results.with_suffix(".md")
    assert report.exists()
    assert "fixture upstream" in report.read_text()


def test_missing_backend_marks_not_run(gw_config, tmp_path):
    results = tmp_path / "run.jsonl"
    rc = main(
        [
            "run",
            "--config",
            str(gw_config),
            "--model",
            "fake",
            "--conditions",
            "memlawb",
            "--tasks",
            "recall-01",
            "--results",
            str(results),
            "--memlawb-checkout",
            str(tmp_path / "no-such-checkout"),
            "--fixture",
        ]
    )
    assert rc == 0
    records = [json.loads(l) for l in results.read_text().splitlines()]
    assert all(r["outcome"] == "not_run" for r in records)
    report = results.with_suffix(".md").read_text()
    assert "not run" in report


def test_resume_skips_completed_and_reruns_error(gw_config, tmp_path):
    results = tmp_path / "run.jsonl"
    results.write_text(
        json.dumps(
            {
                "task_id": "recall-01",
                "condition": "none",
                "variant": "live",
                "outcome": "pass",
            }
        )
        + "\n"
        + json.dumps(
            {
                "task_id": "recall-02",
                "condition": "none",
                "variant": "live",
                "outcome": "error",
            }
        )
        + "\n"
    )
    rc = main(
        [
            "run",
            "--config",
            str(gw_config),
            "--model",
            "fake",
            "--conditions",
            "none",
            "--tasks",
            "recall-01,recall-02",
            "--results",
            str(results),
            "--fixture",
        ]
    )
    assert rc == 0
    records = [json.loads(l) for l in results.read_text().splitlines()]
    by_task = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r["outcome"])
    assert by_task["recall-01"] == ["pass"]  # untouched
    assert by_task["recall-02"] == ["error", "pass"] or by_task["recall-02"][0] == "error"
    assert by_task["recall-02"][-1] in ("pass", "fail")  # re-executed


def test_bad_config_exits_nonzero(tmp_path):
    rc = main(
        [
            "run",
            "--config",
            str(tmp_path / "missing.json"),
            "--conditions",
            "none",
        ]
    )
    assert rc == 2
