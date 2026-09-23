import sys
import time
from pathlib import Path

import pytest

from memval.supervisor import (
    PreflightError,
    RunRoot,
    base_env,
    find_free_port,
    spawn_store,
)


def test_find_free_port_returns_usable():
    port = find_free_port()
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_find_free_port_skips_squatted():
    import socket

    squatter = socket.socket()
    squatter.bind(("127.0.0.1", 0))
    squatter.listen()
    taken = squatter.getsockname()[1]
    try:
        got = find_free_port(taken)
        assert got != taken
    finally:
        squatter.close()


def test_base_env_is_allowlist(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-propagate")
    monkeypatch.setenv("MEMVAL_KEEP_ME", "x")
    env = base_env()
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "MEMVAL_KEEP_ME" not in env
    assert "PATH" in env


def test_spawn_store_health_timeout_is_preflight_error(tmp_path):
    with pytest.raises(PreflightError):
        spawn_store(
            "dead",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            {},
            tmp_path,
            health_url="http://127.0.0.1:1/health",
            health_timeout=0.5,
        )


def test_teardown_kills_child(tmp_path):
    mp = spawn_store(
        "sleeper",
        [sys.executable, "-c", "import time; time.sleep(30)"],
        {},
        tmp_path,
    )
    assert mp.alive()
    mp.stop(timeout=2)
    assert not mp.alive()


def test_run_root_cleanup(tmp_path):
    root = RunRoot()
    (root.path / "x").write_text("y")
    root.cleanup()
    assert not root.path.exists()
