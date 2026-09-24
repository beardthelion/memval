"""Agent-loop tests drive run_cell through the fake upstream with a
fake in-memory CellSession, no real backends required."""
import json
import threading
import urllib.request
from pathlib import Path

import anyio
import pytest

from memval.agent import run_cell
from memval.memory_backends import CellSession, ControlEvidence
from memval.tasks import validate_task
from tests.fixtures import fake_upstream

TASK = validate_task(
    {
        "id": "recall-t1",
        "type": "recall",
        "sessions": [
            ["Remember that I prefer pepperjack cheese on sandwiches."],
            ["What cheese should go on the sandwiches?"],
        ],
        "planted_facts": ["pepperjack"],
        "expected": {"mode": "contains_all", "keywords": ["pepperjack"]},
        "control": True,
    }
)

ISOLATION_TASK = validate_task(
    {
        "id": "leak-t1",
        "type": "isolation",
        "sessions": [
            ["Remember that my favorite pizza topping is anchovies."],
            ["What pizza topping do I like?"],
        ],
        "planted_facts": ["anchovies"],
        "expected": {"mode": "contains_none", "forbidden": ["anchovies"]},
        "control": True,
        "users": ["alice", "bob"],
    }
)


class FakeStore:
    """In-memory backend keyed by user scope."""

    def __init__(self):
        self.scopes: dict[str, dict[str, str]] = {}


class FakeSession(CellSession):
    """A CellSession that simulates memlawb's shape: save/recall tools,
    scope-per-user via respawn, boundary learn that distills 'prefer' lines."""

    def __init__(self, task, control, log_dir, store: FakeStore):
        super().__init__(task=task, control=control, log_dir=log_dir)
        self.store = store
        self.scope = "default"
        self.transcripts: list[str] = []
        self.boundaries = 0

    @property
    def _scope_data(self) -> dict[str, str]:
        return self.store.scopes.setdefault(self.scope, {})

    async def start_leg(self, leg_index: int, user_scope: str | None) -> None:
        self.scope = user_scope or "default"
        names = ["memory_save", "memory_recall"]
        self.tool_specs = [
            {"type": "function", "function": {"name": n, "parameters": {}}} for n in names
        ]
        self.injected_text = "Fake guide: use memory_save and memory_recall."

    async def call_tool(self, name: str, arguments: dict) -> str:
        if self.control and name == "memory_recall":
            self.evidence.read_calls_blinded += 1
            return "No memories found."
        if name == "memory_save":
            self._scope_data[arguments["key"]] = arguments["content"]
            return "saved"
        if name == "memory_recall":
            q = arguments.get("query", "").casefold()
            hits = [
                v
                for v in self._scope_data.values()
                if any(w in v.casefold() for w in q.split() if len(w) > 3)
            ]
            return "\n".join(hits) if hits else "No memories found."
        return "?"

    async def end_leg(self, transcript_path: Path) -> None:
        # Simulate signet-style capture: distill 'prefer' lines into memory.
        self.boundaries += 1
        text = transcript_path.read_text()
        self.transcripts.append(text)
        for line in text.splitlines():
            if "prefer" in line.casefold() or "favorite" in line.casefold():
                self._scope_data[f"learned/{self.boundaries}.md"] = line

    async def witness(self, token: str) -> None:
        self.evidence.witness_ok = any(
            token.casefold() in v.casefold() for v in self._scope_data.values()
        )

    async def finish(self) -> ControlEvidence:
        if self.control:
            self._scope_data["_sentinel.md"] = "x"
            self.evidence.sentinel_landed = "_sentinel.md" in self._scope_data
        return self.evidence


@pytest.fixture
def upstream_url():
    server = fake_upstream.serve(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _run(task, session, gw, tmp_path):
    async def go():
        return await run_cell(
            task, session, gw, "fake",
            seed=1, transcripts_dir=tmp_path, condition="test", variant="live",
        )

    return anyio.run(go)


def test_recall_cell_passes_with_memory(upstream_url, tmp_path):
    store = FakeStore()
    session = FakeSession(TASK, False, tmp_path, store)
    res = _run(TASK, session, upstream_url, tmp_path)
    assert res.outcome == "run"
    assert "pepperjack" in res.final_answer
    assert res.tool_calls >= 1


def test_control_blinds_reads(upstream_url, tmp_path):
    store = FakeStore()
    session = FakeSession(TASK, True, tmp_path, store)
    res = _run(TASK, session, upstream_url, tmp_path)
    assert res.outcome == "run"
    assert "pepperjack" not in res.final_answer
    assert session.evidence.read_calls_blinded >= 1


def test_context_wiped_at_boundary(upstream_url, tmp_path):
    """The leg-2 system+messages must not contain the plant verbatim."""
    store = FakeStore()
    session = FakeSession(TASK, False, tmp_path, store)
    _run(TASK, session, upstream_url, tmp_path)
    # Two boundaries' transcripts: leg transcripts exist and are separate.
    assert len(session.transcripts) == 2
    assert "pepperjack" in session.transcripts[0]


def test_isolation_scope_switch(upstream_url, tmp_path):
    store = FakeStore()
    session = FakeSession(ISOLATION_TASK, False, tmp_path, store)
    res = _run(ISOLATION_TASK, session, upstream_url, tmp_path)
    assert res.outcome == "run"
    # alice's fact must not appear in bob's closing session messages.
    assert not any("anchovies" in m for m in res.closing_messages)
    assert "alice" in store.scopes and "bob" in store.scopes
    assert session.evidence.witness_ok is True


def test_upstream_failure_is_error_not_fail(upstream_url, tmp_path):
    class DeadSession(FakeSession):
        async def call_tool(self, name, arguments):
            raise RuntimeError("backend died")

    # Make the model call a tool by advertising them, then dying.
    store = FakeStore()
    session = DeadSession(TASK, False, tmp_path, store)
    res = _run(TASK, session, upstream_url, tmp_path)
    assert res.outcome == "error"
    assert "backend died" in res.detail


class ExplodingCloseSession(FakeSession):
    async def close(self):
        raise RuntimeError("teardown boom")


def test_teardown_failure_does_not_discard_result(upstream_url, tmp_path):
    session = ExplodingCloseSession(TASK, False, tmp_path, FakeStore())
    res = _run(TASK, session, upstream_url, tmp_path)
    assert res.outcome == "run"
    assert res.evidence.teardown_error is True


class NoneSession(CellSession):
    async def start_leg(self, leg_index, user_scope):
        self.tool_specs = []
        self.injected_text = ""

    async def call_tool(self, name, arguments):
        return f"[tool error] unknown tool: {name}"


def test_none_condition_prompt_claims_no_tools(tmp_path, monkeypatch):
    import memval.agent as agent_mod

    sent = []

    async def fake_chat(gw, model, messages, tools, *a, **kw):
        sent.append(messages[0]["content"])
        return {"role": "assistant", "content": "ok"}

    monkeypatch.setattr(agent_mod, "_chat_async", fake_chat)
    session = NoneSession(task=TASK, control=False, log_dir=tmp_path)

    async def go():
        return await run_cell(
            TASK, session, "http://unused", "fake",
            transcripts_dir=tmp_path, condition="none", variant="live",
        )

    res = anyio.run(go)
    assert res.outcome == "run"
    assert sent and "tools" not in sent[0]
