"""Backend enable/disable isolation checks (R15).

For each backend: stand up a store under a fresh run root, open an MCP
session, write and read back a fact, tear everything down, then prove two
things: no artifact survives (store dir, custody, passphrase file,
transcripts), and a fresh session against a fresh root recalls nothing.
"""
from __future__ import annotations

import secrets as _secrets
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .mcp_client import McpConnection
from .supervisor import (
    PreflightError,
    RunRoot,
    base_env,
    find_free_port,
    memlawb_store_argv,
    memlawb_store_env,
    signet_store_argv,
    signet_store_env,
    spawn_store,
)
import subprocess


@dataclass
class IsolationResult:
    backend: str
    ok: bool
    detail: str


async def _memlawb_roundtrip(checkout: Path, root: RunRoot, token: str) -> None:
    port = find_free_port(8801)
    store = spawn_store(
        "memlawb-store",
        memlawb_store_argv(checkout),
        memlawb_store_env(root.path / "memlawb-data", port),
        root.log_dir(),
        health_url=f"http://127.0.0.1:{port}/health",
    )
    passfile = root.path / "iso-pass"
    passfile.write_text(_secrets.token_hex(24))
    passfile.chmod(0o600)
    conn = McpConnection(
        argv=["bun", "run", str(checkout / "bin" / "memlawb.ts"), "mcp"],
        env={
            "MEMLAWB_URL": f"http://127.0.0.1:{port}",
            "MEMLAWB_NAMESPACE": "user:eval/isolation",
            "MEMLAWB_PASSPHRASE_FILE": str(passfile),
            "MEMLAWB_SCAN": "block",
        },
        errlog_path=root.log_dir() / "mcp-isolation-memlawb.log",
    )
    try:
        await conn.__aenter__()
        text, is_err = await conn.call_tool(
            "memory_save", {"key": "probe.md", "content": f"isolation probe {token}"}
        )
        if is_err:
            raise PreflightError(f"memory_save failed: {text}")
        recalled, _ = await conn.call_tool("memory_search", {"query": "isolation probe"})
        if token not in recalled:
            raise PreflightError("probe fact did not recall after save")
    finally:
        # __aexit__ is idempotent: safe whether or not __aenter__ completed.
        await conn.__aexit__(None, None, None)
        store.stop()


async def _signet_roundtrip(checkout: Path, root: RunRoot, token: str) -> None:
    port = find_free_port(8802)
    store = spawn_store(
        "signet-store",
        signet_store_argv(checkout),
        signet_store_env(root.path / "signet-data", port),
        root.log_dir(),
        health_url=f"http://127.0.0.1:{port}/health",
    )
    home = root.path / "signet-home"
    home.mkdir()
    env = base_env() | {
        "SIGNET_URL": f"http://127.0.0.1:{port}",
        "SIGNET_HOME": str(home),
        "SIGNET_PASSPHRASE": _secrets.token_hex(24),
        "SIGNET_SCAN": "block",
    }
    try:
        proc = subprocess.run(
            ["bun", "run", str(checkout / "bin" / "signet.ts"), "init"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as e:
        store.stop()
        raise PreflightError("signet init timed out") from e
    if proc.returncode != 0:
        store.stop()
        raise PreflightError(f"signet init failed: {proc.stderr.strip()}")
    env.pop("SIGNET_PASSPHRASE")
    conn = McpConnection(
        argv=["bun", "run", str(checkout / "bin" / "signet.ts"), "mcp"],
        env=env,
        errlog_path=root.log_dir() / "mcp-isolation-signet.log",
    )
    try:
        await conn.__aenter__()
        text, is_err = await conn.call_tool(
            "signet_save", {"key": "memory/probe.md", "content": f"isolation probe {token}"}
        )
        if is_err:
            raise PreflightError(f"signet_save failed: {text}")
        recalled, _ = await conn.call_tool("signet_search", {"query": "isolation probe"})
        if token not in recalled:
            raise PreflightError("probe fact did not recall after save")
    finally:
        await conn.__aexit__(None, None, None)
        store.stop()


async def _fresh_recall_empty(
    backend: str, checkout: Path, root: RunRoot, token: str
) -> bool:
    """Second phase: a brand-new root and store must recall nothing."""
    port = find_free_port(8801 if backend == "memlawb" else 8802)
    if backend == "memlawb":
        store = spawn_store(
            "memlawb-store",
            memlawb_store_argv(checkout),
            memlawb_store_env(root.path / "memlawb-data", port),
            root.log_dir(),
            health_url=f"http://127.0.0.1:{port}/health",
        )
        passfile = root.path / "iso-pass"
        passfile.write_text(_secrets.token_hex(24))
        passfile.chmod(0o600)
        conn = McpConnection(
            argv=["bun", "run", str(checkout / "bin" / "memlawb.ts"), "mcp"],
            env={
                "MEMLAWB_URL": f"http://127.0.0.1:{port}",
                "MEMLAWB_NAMESPACE": "user:eval/isolation",
                "MEMLAWB_PASSPHRASE_FILE": str(passfile),
                "MEMLAWB_SCAN": "block",
            },
            errlog_path=root.log_dir() / "mcp-isolation-memlawb-2.log",
        )
    else:
        store = spawn_store(
            "signet-store",
            signet_store_argv(checkout),
            signet_store_env(root.path / "signet-data", port),
            root.log_dir(),
            health_url=f"http://127.0.0.1:{port}/health",
        )
        home = root.path / "signet-home"
        home.mkdir()
        env = base_env() | {
            "SIGNET_URL": f"http://127.0.0.1:{port}",
            "SIGNET_HOME": str(home),
            "SIGNET_PASSPHRASE": _secrets.token_hex(24),
        }
        try:
            proc = subprocess.run(
                ["bun", "run", str(checkout / "bin" / "signet.ts"), "init"],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as e:
            store.stop()
            raise PreflightError("signet init timed out") from e
        if proc.returncode != 0:
            store.stop()
            raise PreflightError(f"signet init failed: {proc.stderr.strip()}")
        env.pop("SIGNET_PASSPHRASE")
        conn = McpConnection(
            argv=["bun", "run", str(checkout / "bin" / "signet.ts"), "mcp"],
            env=env,
            errlog_path=root.log_dir() / "mcp-isolation-signet-2.log",
        )
    try:
        await conn.__aenter__()
        tool = "memory_search" if backend == "memlawb" else "signet_search"
        recalled, _ = await conn.call_tool(tool, {"query": "isolation probe"})
        # The backends echo the query in "no matches" replies, so the check
        # keys on the saved token, not the query text.
        return token not in recalled
    finally:
        await conn.__aexit__(None, None, None)
        store.stop()


def surviving_artifacts(root: RunRoot) -> list[Path]:
    """Anything left under a run root after cleanup.

    Called after teardown so a backend (or a store that recreated its data
    dir on the way down) cannot leave state behind undetected; the check is
    only meaningful because it enumerates real state rather than a list of
    paths we remember creating.
    """
    if not root.path.exists():
        return []
    return sorted(p for p in root.path.rglob("*"))


async def check_isolation(backend: str, checkout: Path) -> IsolationResult:
    if backend not in ("memlawb", "signet"):
        return IsolationResult(backend, False, f"unknown backend: {backend}")
    root_a, root_b = RunRoot(), RunRoot()
    token = _secrets.token_hex(8)
    try:
        if backend == "memlawb":
            await _memlawb_roundtrip(checkout, root_a, token)
        else:
            await _signet_roundtrip(checkout, root_a, token)
        # The strongest reachable-memory check: a fresh store on a fresh root
        # must recall nothing the earlier session wrote.
        if not await _fresh_recall_empty(backend, checkout, root_b, token):
            return IsolationResult(backend, False, "fresh recall returned prior facts")
    except PreflightError as e:
        return IsolationResult(backend, False, str(e))
    except Exception as e:
        return IsolationResult(backend, False, f"{type(e).__name__}: {e}")
    finally:
        root_a.cleanup()
        root_b.cleanup()
    leftovers = surviving_artifacts(root_a) + surviving_artifacts(root_b)
    if leftovers:
        return IsolationResult(
            backend, False, f"artifacts survived teardown: {leftovers[0]} (+{len(leftovers)-1} more)"
        )
    return IsolationResult(backend, True, "enable->session->disable clean; fresh recall empty")
