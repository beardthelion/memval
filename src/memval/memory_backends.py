"""Memory condition wiring: none, memlawb, signet.

The agent loop is identical across conditions; everything that differs lives
here: which tools are advertised, how tool calls dispatch, what happens at a
session boundary, and how a control variant blinds reads.

Scope rules (KTD6): every task gets a fresh memory scope, and isolation tasks
get one scope per fictional user. memlawb scopes by MCP process env
(MEMLAWB_NAMESPACE) so a per-user scope means respawning the MCP server per
leg (KTD1's scoped exception). signet scopes by custody (SIGNET_HOME), so a
per-user scope likewise means respawning per leg with a fresh `signet init`.
"""
from __future__ import annotations

import secrets as _secrets
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .mcp_client import McpConnection, ToolInfo
from .supervisor import PreflightError, base_env
from .tasks import Task

MEMLAWB_TOOLS = [
    "memory_save",
    "memory_recall",
    "memory_search",
    "memory_list",
    "memory_delete",
]
MEMLAWB_READ_TOOLS = {"memory_recall", "memory_search", "memory_list"}
MEMLAWB_GUIDE = "memory_guide"

# The agent-facing signet surface is read-only: learning capture runs
# harness-side via `signet learn` at each boundary (KTD2). signet_save still
# exists on the server and the harness uses it directly for the sentinel
# write, but it is never advertised to the model.
SIGNET_AGENT_TOOLS = ["signet_recall", "signet_search", "signet_list"]
SIGNET_READ_TOOLS = set(SIGNET_AGENT_TOOLS)
SIGNET_GUIDE = "signet_guide"

EMPTY_RESULT_TEXT = "No memories found."


def _openai_tool_spec(tool: ToolInfo) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _filter_guide(
    text: str, advertised: set[str], server_tools: set[str]
) -> str:
    """Drop lines that reference tools this condition does not advertise.

    A guide that tells the agent to call a tool it does not have is a prompt
    defect, not a fair test; the report discloses that filtering happened.
    A line mentioning even one unadvertised tool is dropped whole rather
    than redacted, since a half-instruction is worse than none.
    """
    known = set(server_tools) | set(MEMLAWB_TOOLS) | {
        "signet_save",
        "signet_recall",
        "signet_search",
        "signet_list",
        "signet_delete",
        "signet_config_get",
        "signet_config_set",
        "signet_grant_list",
        "signet_grant_record",
    }
    out = []
    for line in text.splitlines():
        mentioned = [t for t in known if t in line]
        if mentioned and any(t not in advertised for t in mentioned):
            continue
        out.append(line)
    return "\n".join(out)


@dataclass
class ControlEvidence:
    read_calls_blinded: int = 0
    sentinel_landed: bool = False
    witness_ok: bool | None = None  # isolation tasks only
    guide_injected: bool = False
    teardown_error: bool = False


@dataclass
class CellSession:
    """One task x condition x variant execution context."""

    task: Task
    control: bool
    log_dir: Path
    evidence: ControlEvidence = field(default_factory=ControlEvidence)
    tool_specs: list[dict] = field(default_factory=list)
    injected_text: str = ""

    async def start_leg(self, leg_index: int, user_scope: str | None) -> None:
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        raise NotImplementedError

    async def end_leg(self, transcript_path: Path) -> None:
        pass

    async def witness(self, token: str) -> None:
        """Isolation tasks: prove the plant landed under its own scope."""
        pass

    async def finish(self) -> ControlEvidence:
        return self.evidence

    async def close(self) -> None:
        pass


class NoneCellSession(CellSession):
    """No memory surface at all (KTD4): no tools, no stubs."""

    async def start_leg(self, leg_index: int, user_scope: str | None) -> None:
        self.tool_specs = []
        self.injected_text = ""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        # A hallucinated call must not error the cell; it gets the same
        # tool-error surface a real backend would return.
        return f"[tool error] unknown tool: {name}"


class _McpCellSession(CellSession):
    """Shared mechanics for the two real backends."""

    conn: McpConnection | None = None
    read_tools: set[str] = set()
    _advertised_names: set[str] = set()
    _leg_index: int = -1
    _user_scope: str | None = None

    def _conn_id(self, leg_index: int, user_scope: str | None) -> str:
        return f"{self.task.id}-leg{leg_index}-{user_scope or 'default'}"

    async def _spawn(self, leg_index: int, user_scope: str | None) -> McpConnection:
        raise NotImplementedError

    async def _guide_text(self, conn: McpConnection, guide_name: str) -> str:
        raise NotImplementedError

    async def start_leg(self, leg_index: int, user_scope: str | None) -> None:
        # One MCP process persists across legs; a scope change (isolation
        # tasks) forces a respawn because both backends bind scope at spawn.
        if self.conn is not None and self._user_scope != user_scope:
            await self._close_conn()
        if self.conn is None:
            self.conn = await self._spawn(leg_index, user_scope)
        self._leg_index = leg_index
        self._user_scope = user_scope
        advertised = self._advertised_tools()
        self._advertised_names = {t.name for t in advertised}
        self.tool_specs = [_openai_tool_spec(t) for t in advertised]
        self.injected_text = await self._injection(self.conn, advertised)
        self.evidence.guide_injected = bool(self.injected_text.strip())

    def _advertised_tools(self) -> list[ToolInfo]:
        assert self.conn is not None
        return list(self.conn.tools)

    async def _injection(self, conn: McpConnection, advertised: list[ToolInfo]) -> str:
        names = {t.name for t in advertised}
        server_names = {t.name for t in conn.tools}
        parts = []
        if conn.instructions:
            # Server instructions can name tools this condition does not
            # advertise (signet's write surface); filter them like the guide.
            parts.append(_filter_guide(conn.instructions, names, server_names))
        guide = await self._guide_text(conn, self._guide_name())
        if guide:
            parts.append(_filter_guide(guide, names, server_names))
        parts.append(
            "Memory tools available in this session: " + ", ".join(sorted(names))
        )
        return "\n\n".join(parts)

    def _guide_name(self) -> str:
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        assert self.conn is not None
        # Advertise-filter parity: a name the model was never shown is a
        # tool error, not a dispatch. Without this, signet's read-only
        # promise is advisory only and a model that hallucinates
        # signet_save would write for real.
        if name not in self._advertised_names:
            return f"[tool error] unknown tool: {name}"
        if self.control and name in self.read_tools:
            self.evidence.read_calls_blinded += 1
            return EMPTY_RESULT_TEXT
        text, is_error = await self.conn.call_tool(name, arguments)
        if is_error:
            return f"[tool error] {text}"
        return text

    async def _call_unblinded(self, name: str, arguments: dict[str, Any]) -> str:
        assert self.conn is not None
        text, _ = await self.conn.call_tool(name, arguments)
        return text

    async def _close_conn(self) -> None:
        if self.conn is not None:
            conn, self.conn = self.conn, None
            await conn.__aexit__(None, None, None)

    async def close(self) -> None:
        await self._close_conn()


class MemlawbCellSession(_McpCellSession):
    read_tools = MEMLAWB_READ_TOOLS

    def __init__(
        self,
        task: Task,
        control: bool,
        log_dir: Path,
        checkout: Path,
        store_url: str,
        run_root: Path,
    ) -> None:
        super().__init__(task=task, control=control, log_dir=log_dir)
        self.checkout = checkout
        self.store_url = store_url
        self.run_root = run_root
        # One passphrase per task per run, shared by the live and control
        # cells: the control replays the same memory scope with reads blinded,
        # and memlawb refuses to open a namespace under a second key.
        self._passphrase_file = run_root / f"memlawb-pass-{task.id}"
        if not self._passphrase_file.exists():
            self._passphrase_file.write_text(_secrets.token_hex(24))
            self._passphrase_file.chmod(0o600)

    def _namespace(self, user_scope: str | None) -> str:
        base = f"user:eval/{self.task.id}"
        return f"{base}/{user_scope}" if user_scope else base

    async def _spawn(self, leg_index: int, user_scope: str | None) -> McpConnection:
        env = {
            "MEMLAWB_URL": self.store_url,
            "MEMLAWB_NAMESPACE": self._namespace(user_scope),
            "MEMLAWB_PASSPHRASE_FILE": str(self._passphrase_file),
            "MEMLAWB_SCAN": "block",
        }
        conn = McpConnection(
            argv=["bun", "run", str(self.checkout / "bin" / "memlawb.ts"), "mcp"],
            env=env,
            errlog_path=self.log_dir / f"mcp-memlawb-{self._conn_id(leg_index, user_scope)}.log",
        )
        try:
            await conn.__aenter__()
        except Exception as e:
            raise PreflightError(f"memlawb mcp failed to start: {e}") from e
        return conn

    def _guide_name(self) -> str:
        return MEMLAWB_GUIDE

    async def _guide_text(self, conn: McpConnection, guide_name: str) -> str:
        return await conn.get_prompt_text(guide_name) or ""

    async def finish(self) -> ControlEvidence:
        if self.control and self.conn is not None:
            # Entry keys must start with an alphanumeric.
            key = f"memval-sentinel-{_secrets.token_hex(4)}.md"
            await self._call_unblinded(
                "memory_save", {"key": key, "content": "memval control sentinel"}
            )
            listing = await self._call_unblinded("memory_list", {})
            self.evidence.sentinel_landed = key in listing
        return self.evidence

    async def witness(self, token: str) -> None:
        if self.conn is None:
            self.evidence.witness_ok = False
            return
        listing = await self._call_unblinded("memory_search", {"query": token})
        self.evidence.witness_ok = token.casefold() in listing.casefold()


class SignetCellSession(_McpCellSession):
    read_tools = SIGNET_READ_TOOLS

    def __init__(
        self,
        task: Task,
        control: bool,
        log_dir: Path,
        checkout: Path,
        store_url: str,
        run_root: Path,
    ) -> None:
        super().__init__(task=task, control=control, log_dir=log_dir)
        self.checkout = checkout
        self.store_url = store_url
        self.run_root = run_root
        self._homes: dict[str, Path] = {}

    def _home(self, user_scope: str | None) -> Path:
        key = user_scope or "default"
        if key not in self._homes:
            self._homes[key] = self.run_root / f"signet-home-{self.task.id}-{key}"
        return self._homes[key]

    def _signet_env(self, home: Path, passphrase: str | None = None) -> dict[str, str]:
        env = {
            "SIGNET_URL": self.store_url,
            "SIGNET_HOME": str(home),
            "SIGNET_SCAN": "block",
        }
        if passphrase is not None:
            env["SIGNET_PASSPHRASE"] = passphrase
        return env

    def _init_custody(self, home: Path) -> None:
        if (home / "custody.json").exists():
            return
        home.mkdir(parents=True, exist_ok=True)
        # A harness-supplied passphrase means init never prints one to stdout.
        env = base_env() | self._signet_env(home, passphrase=_secrets.token_hex(24))
        proc = subprocess.run(
            ["bun", "run", str(self.checkout / "bin" / "signet.ts"), "init"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise PreflightError(f"signet init failed: {proc.stderr.strip()}")

    def _advertised_tools(self) -> list[ToolInfo]:
        assert self.conn is not None
        return [t for t in self.conn.tools if t.name in SIGNET_AGENT_TOOLS]

    async def _spawn(self, leg_index: int, user_scope: str | None) -> McpConnection:
        home = self._home(user_scope)
        self._init_custody(home)
        conn = McpConnection(
            argv=["bun", "run", str(self.checkout / "bin" / "signet.ts"), "mcp"],
            env=self._signet_env(home),
            errlog_path=self.log_dir / f"mcp-signet-{self._conn_id(leg_index, user_scope)}.log",
        )
        try:
            await conn.__aenter__()
        except Exception as e:
            raise PreflightError(f"signet mcp failed to start: {e}") from e
        return conn

    def _guide_name(self) -> str:
        return SIGNET_GUIDE

    async def _guide_text(self, conn: McpConnection, guide_name: str) -> str:
        return await conn.get_prompt_text(guide_name) or ""

    async def end_leg(self, transcript_path: Path) -> None:
        # Learning capture is harness-side (KTD2): `signet learn` distills the
        # leg's transcript into memory/ entries under this scope's custody.
        assert self.conn is not None
        env = base_env() | self._signet_env(self._home(self._user_scope))
        proc = subprocess.run(
            [
                "bun",
                "run",
                str(self.checkout / "bin" / "signet.ts"),
                "learn",
                str(transcript_path),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise PreflightError(f"signet learn failed: {proc.stderr.strip()}")

    async def finish(self) -> ControlEvidence:
        if self.control and self.conn is not None:
            # Entry-key segments must start with an alphanumeric (SN-020).
            key = f"memory/memval-sentinel-{_secrets.token_hex(4)}.md"
            await self._call_unblinded("signet_save", {"key": key, "content": "memval sentinel"})
            listing = await self._call_unblinded("signet_list", {})
            self.evidence.sentinel_landed = key in listing
        return self.evidence

    async def witness(self, token: str) -> None:
        if self.conn is None:
            self.evidence.witness_ok = False
            return
        listing = await self._call_unblinded("signet_search", {"query": token})
        self.evidence.witness_ok = token.casefold() in listing.casefold()


def make_cell_session(
    condition: str,
    task: Task,
    control: bool,
    log_dir: Path,
    *,
    memlawb_checkout: Path | None = None,
    memlawb_url: str = "",
    signet_checkout: Path | None = None,
    signet_url: str = "",
    run_root: Path,
) -> CellSession:
    if condition == "none":
        return NoneCellSession(task=task, control=control, log_dir=log_dir)
    if condition == "memlawb":
        assert memlawb_checkout is not None
        return MemlawbCellSession(task, control, log_dir, memlawb_checkout, memlawb_url, run_root)
    if condition == "signet":
        assert signet_checkout is not None
        return SignetCellSession(task, control, log_dir, signet_checkout, signet_url, run_root)
    raise ValueError(f"unknown condition: {condition}")
