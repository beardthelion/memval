"""Process supervision for memval runs.

Owns the per-run temporary root, the gateway process (in-process thread, see
gateway.py), the backend store subprocesses, and teardown. Store children get
an explicit environment allow-list rather than the inherited environment:
ambient variables like S3_* or a stray PORT could otherwise redirect a
"per-run tmp" store at an operator's real data, which R24 forbids.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


class PreflightError(RuntimeError):
    """A condition cannot run; the cell records not_run, never a crash."""


def find_free_port(preferred: int | None = None) -> int:
    """Return preferred if free, else an OS-assigned free loopback port."""
    if preferred is not None:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def base_env() -> dict[str, str]:
    """Minimal ambient variables a child needs to run at all.

    Everything else (backend settings, secrets) is passed explicitly by the
    caller, so nothing unexpected rides into a child environment.
    """
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "BUN_INSTALL", "XDG_CACHE_HOME")
    return {k: os.environ[k] for k in keep if k in os.environ}


def wait_for_health(url: str, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.1)
    return False


@dataclass
class ManagedProcess:
    name: str
    proc: subprocess.Popen
    log_path: Path

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        if self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=timeout)


@dataclass
class RunRoot:
    """Per-run temporary root; everything the run creates lives under it."""

    path: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="memval-")))

    def log_dir(self) -> Path:
        d = self.path / "logs"
        d.mkdir(exist_ok=True)
        return d

    def cleanup(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def spawn_store(
    name: str,
    argv: list[str],
    env_block: dict[str, str],
    log_dir: Path,
    health_url: str | None = None,
    health_timeout: float = 15.0,
) -> ManagedProcess:
    """Spawn a store child with an explicit env block; stderr goes to a log."""
    env = base_env() | env_block
    log_path = log_dir / f"{name}.log"
    log = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    finally:
        # Popen dup'd the fd into the child; the parent's copy would leak.
        log.close()
    mp = ManagedProcess(name=name, proc=proc, log_path=log_path)
    if health_url and not wait_for_health(health_url, health_timeout):
        tail = log_path.read_bytes()[-2000:].decode(errors="replace")
        mp.stop()
        raise PreflightError(f"{name} did not become healthy at {health_url}: {tail.strip()}")
    if not mp.alive():
        raise PreflightError(f"{name} exited immediately; see {log_path}")
    return mp


def memlawb_store_argv(checkout: Path) -> list[str]:
    # The checkout's bin/ points at an unbuilt dist/; run the source entrypoint.
    return ["bun", "run", str(checkout / "bin" / "memlawb.ts"), "serve"]


def signet_store_argv(checkout: Path) -> list[str]:
    return ["bun", "run", str(checkout / "bin" / "signet.ts"), "serve"]


def memlawb_store_env(data_dir: Path, port: int) -> dict[str, str]:
    return {
        "ALLOW_UNAUTHENTICATED": "true",
        "STORE": "fs",
        "DATA_DIR": str(data_dir),
        "PORT": str(port),
        # A high ceiling rather than 0: 0 would deny every request.
        "RATE_LIMIT_PER_MINUTE": "100000",
    }


def signet_store_env(data_dir: Path, port: int) -> dict[str, str]:
    return {
        "SIGNET_DATA_DIR": str(data_dir),
        "PORT": str(port),
        "SIGNET_MODE": "local",
        "SIGNET_HOST": "127.0.0.1",
    }
