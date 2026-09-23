"""Transcript persistence (R3): every cell's leg transcripts land in
<results-stem>.transcripts/ beside the results file, namespaced by
condition and variant so no cell overwrites another's, and they survive
the run for a later judge pass."""
import json
import threading

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


def _run_cli(gw_config, results, *extra):
    return main(
        [
            "run",
            "--config",
            str(gw_config),
            "--model",
            "fake",
            "--results",
            str(results),
            "--fixture",
            *extra,
        ]
    )


def test_none_run_leaves_named_transcripts(gw_config, tmp_path):
    """One transcript per leg, named with condition and variant, surviving
    after the run (run_root cleanup) completes."""
    results = tmp_path / "run.jsonl"
    rc = _run_cli(gw_config, results, "--conditions", "none", "--tasks", "recall-01")
    assert rc == 0
    transcripts = tmp_path / "run.transcripts"
    # recall-01 has two legs; a none cell has only the live variant.
    assert sorted(p.name for p in transcripts.iterdir()) == [
        "recall-01-none-live-leg0.txt",
        "recall-01-none-live-leg1.txt",
    ]
    for p in transcripts.iterdir():
        assert p.read_text().strip()


def test_live_and_control_transcripts_are_distinct(gw_config, tmp_path, monkeypatch):
    """Regression: the control cell used to overwrite the live cell's
    transcripts because filenames carried no variant."""
    import memval.run as run_mod
    from tests.test_agent import FakeSession, FakeStore

    class DummyStore:
        def stop(self):
            pass

    store = FakeStore()
    monkeypatch.setattr(run_mod, "spawn_store", lambda *a, **k: DummyStore())
    monkeypatch.setattr(
        run_mod,
        "make_cell_session",
        lambda cond, t, control, log_dir, **kw: FakeSession(t, control, log_dir, store),
    )

    results = tmp_path / "run.jsonl"
    rc = _run_cli(gw_config, results, "--conditions", "memlawb", "--tasks", "recall-01")
    assert rc == 0
    transcripts = tmp_path / "run.transcripts"
    assert sorted(p.name for p in transcripts.iterdir()) == [
        "recall-01-memlawb-control-leg0.txt",
        "recall-01-memlawb-control-leg1.txt",
        "recall-01-memlawb-live-leg0.txt",
        "recall-01-memlawb-live-leg1.txt",
    ]
    live_leg1 = (transcripts / "recall-01-memlawb-live-leg1.txt").read_text()
    control_leg1 = (transcripts / "recall-01-memlawb-control-leg1.txt").read_text()
    assert live_leg1 != control_leg1
    # Live recall surfaces the plant; the control's reads are blinded.
    assert "pepperjack" in live_leg1
    assert "pepperjack" not in control_leg1


def test_resume_preserves_completed_transcripts(gw_config, tmp_path):
    """A resume must not delete or rewrite transcripts of cells already
    recorded as completed in the results file."""
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
    )
    transcripts = tmp_path / "run.transcripts"
    transcripts.mkdir()
    for leg in (0, 1):
        (transcripts / f"recall-01-none-live-leg{leg}.txt").write_text(
            f"sentinel leg {leg}\n"
        )

    rc = _run_cli(
        gw_config, results, "--conditions", "none", "--tasks", "recall-01,recall-02"
    )
    assert rc == 0
    # The completed cell's transcripts are untouched and unduplicated.
    assert sorted(p.name for p in transcripts.iterdir()) == [
        "recall-01-none-live-leg0.txt",
        "recall-01-none-live-leg1.txt",
        "recall-02-none-live-leg0.txt",
        "recall-02-none-live-leg1.txt",
    ]
    for leg in (0, 1):
        assert (transcripts / f"recall-01-none-live-leg{leg}.txt").read_text() == (
            f"sentinel leg {leg}\n"
        )


def test_bare_results_path_resolves_sibling_transcripts(gw_config, tmp_path, monkeypatch):
    """A results path with no directory component still gets a sibling
    transcripts dir, resolved against the cwd."""
    monkeypatch.chdir(tmp_path)
    rc = _run_cli(gw_config, "run.jsonl", "--conditions", "none", "--tasks", "recall-01")
    assert rc == 0
    transcripts = tmp_path / "run.transcripts"
    assert (transcripts / "recall-01-none-live-leg0.txt").exists()
    assert (transcripts / "recall-01-none-live-leg1.txt").exists()
