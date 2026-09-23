"""Fake OpenAI-compatible upstream for tests and fixture runs.

A deterministic heuristic stand-in for a real model:
- if the last message is a tool result, answer with its content so saved
  facts surface in the final answer,
- if tools are advertised and the latest user turn looks like a plant
  (preference/decision phrasing), emit a save call when a save tool exists,
- if tools are advertised and the latest user turn is a question, emit a
  recall/search call,
- otherwise answer with a canned non-answer.

Run standalone: python -m tests.fixtures.fake_upstream --port 8901
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SAVE_TOOLS = {"memory_save"}
READ_TOOLS = {"memory_recall", "signet_recall"}
SEARCH_TOOLS = {"memory_search", "signet_search"}

PLANT_HINTS = ("i prefer", "remember that", "we decided", "my favorite", "i want to book", "i'm planning")


def decide(messages: list[dict], tools: list[dict]) -> dict:
    """Return an OpenAI-shaped assistant message."""
    tool_names = {
        t.get("function", {}).get("name") for t in tools if t.get("type") == "function"
    }
    last = messages[-1] if messages else {}

    if last.get("role") == "tool":
        return {"role": "assistant", "content": last.get("content") or ""}

    last_user = next(
        (m for m in reversed(messages) if m.get("role") == "user"), {}
    )
    text = (last_user.get("content") or "").casefold()

    if any(h in text for h in PLANT_HINTS):
        save = next(iter(SAVE_TOOLS & tool_names), None)
        if save:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": save,
                            "arguments": json.dumps(
                                {"key": "facts.md", "content": last_user.get("content", "")}
                            ),
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "Noted."}

    if text.rstrip().endswith("?"):
        read = next(iter(READ_TOOLS & tool_names), None) or next(
            iter(SEARCH_TOOLS & tool_names), None
        )
        if read:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": read,
                            "arguments": json.dumps({"query": last_user.get("content", "")}),
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "I don't have that information."}

    return {"role": "assistant", "content": "Understood."}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": {"message": "unsupported path"}})
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, UnicodeDecodeError):
            self._json(400, {"error": {"message": "invalid json"}})
            return
        msg = decide(body.get("messages") or [], body.get("tools") or [])
        self._json(
            200,
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": msg, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )


def serve(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8901)
    args = ap.parse_args()
    server = serve(args.port)
    print(f"fake upstream on 127.0.0.1:{args.port}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
