"""Backend tool-surface tests: advertised-set enforcement and guide
filtering, exercised without real backends."""
import pytest

from memval.memory_backends import (
    MEMLAWB_TOOLS,
    SIGNET_AGENT_TOOLS,
    NoneCellSession,
    SignetCellSession,
    _filter_guide,
)
from memval.tasks import validate_task

TASK = validate_task(
    {
        "id": "recall-t1",
        "type": "recall",
        "sessions": [["Remember x."], ["What is x?"]],
        "planted_facts": ["x"],
        "expected": {"mode": "contains_all", "keywords": ["x"]},
        "control": True,
    }
)


def test_filter_guide_drops_lines_naming_unadvertised_tools():
    advertised = set(SIGNET_AGENT_TOOLS)
    server = set(SIGNET_AGENT_TOOLS) | {"signet_save", "signet_grant_record"}
    guide = "\n".join(
        [
            "Use signet_recall to find stored facts.",
            "Call signet_save after every user turn.",
            "Mixed line: signet_recall and signet_save together.",
            "No tools mentioned here.",
        ]
    )
    out = _filter_guide(guide, advertised, server)
    assert "signet_recall to find" in out
    assert "signet_save" not in out
    assert "No tools mentioned" in out


def test_filter_guide_keeps_all_lines_when_fully_advertised():
    advertised = set(MEMLAWB_TOOLS)
    guide = "Call memory_save, then memory_recall."
    assert _filter_guide(guide, advertised, set(MEMLAWB_TOOLS)) == guide


class _FakeConn:
    tools = []
    instructions = ""

    async def call_tool(self, name, arguments):
        return "ok", False


@pytest.mark.anyio
async def test_unadvertised_tool_call_is_a_tool_error(tmp_path):
    session = SignetCellSession(
        TASK, False, tmp_path, checkout=tmp_path, store_url="", run_root=tmp_path
    )
    session.conn = _FakeConn()
    session._advertised_names = {"signet_recall"}
    out = await session.call_tool("signet_save", {"key": "k", "content": "c"})
    assert out.startswith("[tool error]")
    # Advertised names still dispatch.
    assert await session.call_tool("signet_recall", {"query": "q"}) == "ok"


@pytest.mark.anyio
async def test_none_session_tool_call_is_a_tool_error_not_assertion(tmp_path):
    session = NoneCellSession(task=TASK, control=False, log_dir=tmp_path)
    out = await session.call_tool("memory_recall", {})
    assert out.startswith("[tool error]")
