import sys
from pathlib import Path

import anyio
import pytest

from memval.mcp_client import McpConnection, ProtocolCorruptionError

FIXTURE = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


def _conn(tmp_path: Path, env: dict | None = None) -> McpConnection:
    return McpConnection(
        argv=[sys.executable, str(FIXTURE)],
        env=env or {},
        errlog_path=tmp_path / "mcp-err.log",
        read_timeout_seconds=15,
    )


def test_tool_roundtrip(tmp_path):
    async def go():
        async with _conn(tmp_path) as conn:
            names = {t.name for t in conn.tools}
            assert {"memory_save", "memory_recall", "memory_search", "memory_list"} <= names
            text, is_err = await conn.call_tool(
                "memory_save", {"key": "a.md", "content": "pepperjack fact"}
            )
            assert not is_err and "saved" in text
            text, _ = await conn.call_tool("memory_search", {"query": "pepperjack"})
            assert "pepperjack" in text
            text, _ = await conn.call_tool("memory_recall", {"query": "nothing relevant"})
            assert "No memories found" in text

    anyio.run(go)


def test_child_env_is_explicit_allowlist(tmp_path):
    async def go():
        async with _conn(tmp_path, env={"MEMVAL_MARKER": "yes"}) as conn:
            text, _ = await conn.call_tool("env_probe", {"prefix": "MEMVAL_"})
            assert "MEMVAL_MARKER" in text
            # Ambient secrets never ride in: nothing we didn't pass is set.
            text, _ = await conn.call_tool("env_probe", {"prefix": "AWS_"})
            assert text.strip() == "" or "AWS_" not in text

    anyio.run(go)


def test_instructions_and_prompt(tmp_path):
    async def go():
        async with _conn(tmp_path) as conn:
            assert conn.instructions == "Fake memory server for tests."
            guide = await conn.get_prompt_text("memory_guide")
            assert guide and "memory_save" in guide

    anyio.run(go)


def test_stray_stdout_is_fatal(tmp_path):
    async def go():
        conn = _conn(tmp_path, env={"FAKE_MCP_STDOUT_NOISE": "1"})
        with pytest.raises(ProtocolCorruptionError):
            await conn.__aenter__()

    anyio.run(go)
