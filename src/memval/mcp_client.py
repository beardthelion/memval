"""MCP stdio client wrapper used by the harness.

Thin wrapper over the mcp SDK: each connection is one stdio child spawned
with an explicit env (the SDK merges it over a minimal safe allow-list, so
every MEMLAWB_*/SIGNET_* variable must be named here). Child stderr is tee'd
to a per-connection log file; stdout is the protocol channel. The SDK logs
non-protocol bytes and keeps serving, so this wrapper promotes them to fatal
via the message_handler hook.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client


class ProtocolCorruptionError(RuntimeError):
    """A non-protocol byte arrived on the MCP child's stdout."""


@dataclass
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


class McpConnection:
    def __init__(
        self,
        argv: list[str],
        env: dict[str, str],
        errlog_path: Path,
        read_timeout_seconds: float = 60.0,
    ) -> None:
        self._argv = argv
        self._env = env
        self._errlog_path = errlog_path
        self._read_timeout = read_timeout_seconds
        self._errlog: TextIO | None = None
        self._client: Client | None = None
        self._transport_errors: list[Exception] = []
        self.instructions: str | None = None
        self.tools: list[ToolInfo] = []

    async def _on_message(self, message: Any) -> None:
        if isinstance(message, Exception):
            self._transport_errors.append(message)

    async def __aenter__(self) -> "McpConnection":
        self._errlog = open(self._errlog_path, "a", encoding="utf-8", errors="replace")
        params = StdioServerParameters(
            command=self._argv[0],
            args=self._argv[1:],
            env=self._env,
        )
        # stdio_client doubles as the Transport, which lets us keep errlog
        # capture that Client(StdioServerParameters) would take on its own.
        # If anything after __aenter__ raises, __aexit__ must still run or the
        # child stays alive and asyncio blocks in waitpid at loop teardown.
        try:
            self._client = Client(
                stdio_client(params, errlog=self._errlog),
                read_timeout_seconds=self._read_timeout,
                message_handler=self._on_message,
            )
            await self._client.__aenter__()
            self.instructions = self._client.instructions
            tools = await self._client.list_tools()
            self.tools = [
                ToolInfo(
                    name=t.name,
                    description=t.description or "",
                    input_schema=dict(t.input_schema or {}),
                )
                for t in tools.tools
            ]
            # Transport exceptions reach message_handler via a spawned task, so
            # a stray byte written at spawn lands a checkpoint or two after the
            # handshake. Give delivery a bounded window before declaring clean.
            for _ in range(20):
                if self._transport_errors:
                    break
                await anyio.sleep(0.01)
            self._raise_if_corrupt()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    def _raise_if_corrupt(self) -> None:
        if self._transport_errors:
            raise ProtocolCorruptionError(
                f"non-protocol output on {self._argv[0]} stdout: "
                f"{self._transport_errors[0]}"
            )

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await client.__aexit__(*exc)
        if self._errlog is not None:
            self._errlog.close()
            self._errlog = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Return (text, is_error). Tool errors surface as data, not raises."""
        assert self._client is not None
        self._raise_if_corrupt()
        result = await self._client.call_tool(name, arguments)
        text = "\n".join(
            getattr(c, "text", "") for c in (result.content or []) if getattr(c, "text", None)
        )
        return text, bool(getattr(result, "isError", getattr(result, "is_error", False)))

    async def get_prompt_text(self, name: str) -> str | None:
        assert self._client is not None
        try:
            result = await self._client.get_prompt(name)
        except Exception:
            return None
        parts = []
        for msg in result.messages or []:
            content = getattr(msg, "content", None)
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts) or None
