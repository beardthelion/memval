"""A minimal stdio MCP server for tests: in-memory save/recall/search/list
plus a prompt named like the real backends' guides. Not a backend substitute;
it exists to exercise memval's MCP plumbing without Bun or private checkouts.

Run: python tests/fixtures/fake_mcp_server.py
Env knobs for tests:
  FAKE_MCP_STDOUT_NOISE=1   write a stray byte to stdout before serving
                            (protocol corruption check)
"""
from __future__ import annotations

import os
import sys

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent

STORE: dict[str, str] = {}

server = MCPServer("fake-memory", instructions="Fake memory server for tests.")


@server.tool()
def memory_save(key: str, content: str) -> str:
    """Save a memory entry."""
    STORE[key] = content
    return f"saved {key}"


@server.tool()
def memory_recall(query: str) -> str:
    """Recall memories relevant to a query (substring over values)."""
    hits = [f"{k}: {v}" for k, v in STORE.items() if _overlaps(query, v)]
    return "\n".join(hits) if hits else "No memories found."


@server.tool()
def memory_search(query: str) -> str:
    """Literal substring search over keys and values."""
    hits = [f"{k}: {v}" for k, v in STORE.items() if query.casefold() in (k + v).casefold()]
    return "\n".join(hits) if hits else "No memories found."


@server.tool()
def memory_list() -> str:
    """List entry keys."""
    return "\n".join(sorted(STORE)) if STORE else "No entries."


@server.tool()
def env_probe(prefix: str) -> str:
    """Report which env vars starting with prefix this child has."""
    return ",".join(sorted(k for k in os.environ if k.startswith(prefix)))


def _overlaps(query: str, value: str) -> bool:
    q = {w for w in query.casefold().split() if len(w) > 3}
    return any(w in value.casefold() for w in q)


@server.prompt(name="memory_guide")
def memory_guide() -> str:
    return "Use memory_save for durable facts. Call memory_recall before answering."


async def _main() -> None:
    if os.environ.get("FAKE_MCP_STDOUT_NOISE"):
        sys.stdout.write("noise\n")
        sys.stdout.flush()
    await server.run_stdio_async()


if __name__ == "__main__":
    anyio.run(_main)
