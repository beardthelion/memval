"""The shared agent loop.

One loop drives every condition (R6): scripted user turns per leg, a context
wipe at each boundary, OpenAI-shaped chat calls through the local gateway,
tool dispatch to the condition's CellSession. Nothing in here knows which
backend is attached; that asymmetry lives entirely in memory_backends.py.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .memory_backends import CellSession, ControlEvidence
from .tasks import Task

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the tools shown. "
    "Answer concisely. When a tool is available, prefer it over guessing."
)

MAX_TURNS_PER_PROMPT = 8


class UpstreamError(RuntimeError):
    """Gateway or upstream failure; the cell records `error`, not `fail`."""


@dataclass
class LegResult:
    assistant_messages: list[str] = field(default_factory=list)
    tool_calls: int = 0


@dataclass
class CellResult:
    outcome: str  # pass | fail | error | not_run
    final_answer: str = ""
    closing_messages: list[str] = field(default_factory=list)
    tool_calls: int = 0
    detail: str = ""
    evidence: ControlEvidence = field(default_factory=ControlEvidence)


def _chat(
    gateway_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
    temperature: float,
    seed: int | None,
    max_tokens: int,
) -> dict:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        body["seed"] = seed
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(
        gateway_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise UpstreamError(f"gateway returned {e.code}: {e.read()[:300]!r}") from e
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise UpstreamError(f"gateway request failed: {e}") from e
    try:
        return payload["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        raise UpstreamError(f"malformed upstream response: {payload!r:.300}") from e


def _write_transcript(path: Path, messages: list[dict]) -> None:
    """Plain speaker-prefixed lines so signet's distiller sees clean text."""
    lines = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            if content:
                lines.append(f"Assistant: {content}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                lines.append(f"Assistant called {fn.get('name')}({fn.get('arguments', '')})")
        elif role == "tool":
            lines.append(f"Tool: {content}")
    path.write_text("\n".join(lines) + "\n")


async def run_cell(
    task: Task,
    session: CellSession,
    gateway_url: str,
    model: str,
    *,
    temperature: float = 0.0,
    seed: int | None = None,
    max_tokens: int = 4096,
    transcripts_dir: Path,
) -> CellResult:
    """Replay one task under one cell session; score nothing here."""
    tool_calls = 0
    try:
        for leg_index, leg in enumerate(task.sessions):
            user_scope = task.users[leg_index] if leg_index < len(task.users) else None
            await session.start_leg(leg_index, user_scope)
            messages: list[dict] = [
                {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + session.injected_text}
                if session.injected_text
                else {"role": "system", "content": SYSTEM_PROMPT}
            ]
            for turn in leg:
                messages.append({"role": "user", "content": turn})
                for _ in range(MAX_TURNS_PER_PROMPT):
                    msg = await _chat_async(
                        gateway_url, model, messages, session.tool_specs,
                        temperature, seed, max_tokens,
                    )
                    messages.append(msg)
                    tcs = msg.get("tool_calls") or []
                    if not tcs:
                        break
                    for tc in tcs:
                        fn = tc.get("function", {})
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except ValueError:
                            args = {}
                        out = await session.call_tool(fn.get("name", ""), args)
                        tool_calls += 1
                        messages.append(
                            {"role": "tool", "tool_call_id": tc.get("id", ""), "content": out}
                        )
                else:
                    raise UpstreamError(f"task {task.id} leg {leg_index}: turn budget exhausted")
            transcript = transcripts_dir / f"{task.id}-leg{leg_index}.txt"
            _write_transcript(transcript, messages)
            await session.end_leg(transcript)
            # Isolation witness: prove the plant landed in this leg's scope
            # before moving on. Only meaningful on the planting leg.
            if task.type == "isolation" and leg_index == 0 and task.planted_facts:
                token = task.planted_facts[-1]
                await session.witness(token)
        closing = [
            m.get("content") or ""
            for m in messages
            if m.get("role") == "assistant" and m.get("content")
        ]
        # finish() (control sentinel write, evidence collection) needs the
        # MCP connection still open, so it runs before the finally's close().
        evidence = await session.finish()
        return CellResult(
            outcome="run",
            final_answer=closing[-1] if closing else "",
            closing_messages=closing,
            tool_calls=tool_calls,
            evidence=evidence,
        )
    except Exception as e:
        return CellResult(
            outcome="error",
            detail=f"{type(e).__name__}: {e}",
            tool_calls=tool_calls,
            evidence=session.evidence,
        )
    finally:
        await session.close()


async def _chat_async(*args) -> dict:
    import anyio

    return await anyio.to_thread.run_sync(lambda: _chat(*args))
