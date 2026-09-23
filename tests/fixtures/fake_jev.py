"""Fake TypeSafe System One endpoint for judge tests.

A scripted stand-in for ``POST /v1/systemone``: each request body is captured
on ``server.jev_state.requests`` and handed to a responder that returns
``(status, payload)``. The default responder answers with a well-formed
four-question bundle mirroring the spike-verified API shape (score and choice
answers carry confidence + distribution; noul answers carry a bare
probability).

Run standalone: python -m tests.fixtures.fake_jev --port 8902
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

DEFAULT_ANSWERS = {
    "task_success": {
        "score": 3.6,
        "confidence": 0.82,
        "distribution": {"0": 0.0, "1": 0.0, "2": 0.05, "3": 0.55, "4": 0.4},
    },
    "memory_used": {"probability": 0.9},
    "confabulated": {"probability": 0.08},
    "failure_class": {
        "choice": "no-failure",
        "confidence": 0.9,
        "distribution": {
            "no-failure": 0.9,
            "retrieval-miss": 0.04,
            "wrong-memory": 0.02,
            "confabulation": 0.02,
            "tool-error": 0.01,
            "refused": 0.01,
        },
    },
}

Responder = Callable[[dict], "tuple[int, object]"]


def default_response(body: dict) -> dict:
    return {
        "model": "jev-latest",
        "answers": DEFAULT_ANSWERS,
        "usage": {"input_tokens": 1234, "output_tokens": 0},
    }


class _State:
    def __init__(self, responder: Responder) -> None:
        self.responder = responder
        self.requests: list[dict] = []


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

    def do_POST(self):
        state: _State = self.server.jev_state
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, UnicodeDecodeError):
            self._json(400, {"error": "invalid json"})
            return
        state.requests.append({"headers": dict(self.headers), "body": body})
        status, payload = state.responder(body)
        self._json(status, payload)


def serve(port: int = 0, responder: Responder | None = None) -> ThreadingHTTPServer:
    """Build a fake Jev server. ``responder(body) -> (status, payload)``;
    captured requests live on ``server.jev_state.requests``."""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.jev_state = _State(responder or (lambda body: (200, default_response(body))))
    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8902)
    args = ap.parse_args()
    server = serve(args.port)
    print(f"fake jev on 127.0.0.1:{server.server_address[1]}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
