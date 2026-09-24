"""Local OpenAI-compatible gateway, a minimal port of beardbot's hop.py.

A config file maps model names to {"base": <upstream url>, "key_env": <env var
holding the key>}; the gateway injects `Authorization: Bearer` from the named
env var so keys never live in the config or the client. It binds 127.0.0.1
only and exposes /health plus /chat/completions; upstream status codes
propagate verbatim. Upstream bases must be https, except loopback (the fake
upstream fixture).
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_MAX_BODY = 8 * 1024 * 1024
_TIMEOUT = 300.0


class GatewayConfigError(ValueError):
    pass


def is_loopback_host(host: str) -> bool:
    """The http exception allowed for fixtures: any loopback bind."""
    return host in {"127.0.0.1", "localhost", "::1"} or host.startswith("127.")


def load_gateway_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    models = cfg.get("models")
    if not isinstance(models, dict) or not models:
        raise GatewayConfigError("gateway config needs a non-empty 'models' object")
    for name, route in models.items():
        base = route.get("base")
        if not isinstance(base, str) or not base:
            raise GatewayConfigError(f"model {name!r}: missing 'base'")
        parsed = urllib.parse.urlparse(base)
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and is_loopback_host(parsed.hostname or "")
        ):
            raise GatewayConfigError(
                f"model {name!r}: upstream base must be https (loopback http allowed): {base!r}"
            )
    return cfg


def make_handler(config: dict):
    models = config["models"]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "memval-gateway/1"

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
                self._json(200, {"ok": True, "models": sorted(models)})
            else:
                self._json(404, {"error": {"message": "not found", "type": "memval"}})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._json(404, {"error": {"message": "unsupported path", "type": "memval"}})
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n > _MAX_BODY:
                self._json(413, {"error": {"message": "body too large", "type": "memval"}})
                return
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, UnicodeDecodeError):
                self._json(400, {"error": {"message": "invalid json", "type": "memval"}})
                return

            model = body.get("model")
            route = models.get(model)
            if route is None:
                self._json(
                    404,
                    {"error": {"message": f"unknown model: {model!r}", "type": "memval"}},
                )
                return
            body.pop("stream", None)
            self._relay(route, body)

        def _relay(self, route, body):
            url = route["base"].rstrip("/") + "/chat/completions"
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
            req.add_header("Content-Type", "application/json")
            # urllib's default UA is bot-filtered by some providers' edge (cf 1010).
            req.add_header("User-Agent", "curl/8.5.0")
            key_env = route.get("key_env")
            if key_env:
                key = os.environ.get(key_env, "")
                if not key:
                    self._json(
                        500,
                        {"error": {"message": f"{key_env} not set", "type": "memval"}},
                    )
                    return
                req.add_header("Authorization", "Bearer " + key)

            try:
                resp = urllib.request.urlopen(req, timeout=_TIMEOUT)
                payload = resp.read()
                code = resp.status
                ctype = resp.headers.get("Content-Type") or "application/json"
            except urllib.error.HTTPError as e:
                payload = e.read() or b""
                code = e.code
                ctype = (e.headers.get("Content-Type") if e.headers else None) or "application/json"
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                self._json(
                    502,
                    {"error": {"message": f"upstream unreachable: {e}", "type": "memval"}},
                )
                return
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def serve_gateway(config: dict, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    if not is_loopback_host(host):
        raise GatewayConfigError(f"gateway binds loopback only, got {host!r}")
    server = ThreadingHTTPServer((host, port), make_handler(config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
