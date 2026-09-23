import json
import urllib.error
import urllib.request

import pytest

from memval.gateway import (
    GatewayConfigError,
    load_gateway_config,
    serve_gateway,
)
from tests.fixtures import fake_upstream


@pytest.fixture
def upstream():
    server = fake_upstream.serve(0)
    import threading

    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_health_and_completion_roundtrip(upstream):
    base = f"http://127.0.0.1:{upstream.server_address[1]}"
    cfg = {"models": {"fake": {"base": base, "key_env": None}}}
    gw = serve_gateway(cfg, port=0)
    try:
        port = gw.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as r:
            assert json.loads(r.read())["ok"] is True
        code, body = _post(
            f"http://127.0.0.1:{port}/chat/completions",
            {"model": "fake", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert code == 200
        assert body["choices"][0]["message"]["content"] == "Understood."
    finally:
        gw.shutdown()


def test_unknown_model_404(upstream):
    base = f"http://127.0.0.1:{upstream.server_address[1]}"
    gw = serve_gateway({"models": {"fake": {"base": base, "key_env": None}}}, port=0)
    try:
        code, _ = _post(
            f"http://127.0.0.1:{gw.server_address[1]}/chat/completions",
            {"model": "nope", "messages": []},
        )
        assert code == 404
    finally:
        gw.shutdown()


def test_missing_key_env_returns_500_without_secret(upstream):
    base = f"http://127.0.0.1:{upstream.server_address[1]}"
    cfg = {"models": {"fake": {"base": base, "key_env": "MEMVAL_DEFINITELY_UNSET_KEY"}}}
    gw = serve_gateway(cfg, port=0)
    try:
        code, body = _post(
            f"http://127.0.0.1:{gw.server_address[1]}/chat/completions",
            {"model": "fake", "messages": []},
        )
        assert code == 500
        assert "MEMVAL_DEFINITELY_UNSET_KEY" in body["error"]["message"]
    finally:
        gw.shutdown()


def test_bearer_injected_from_env(upstream, monkeypatch):
    seen = {}

    class Sniffing(fake_upstream.Handler):
        def do_POST(self):
            seen["auth"] = self.headers.get("Authorization")
            super().do_POST()

    import threading
    from http.server import ThreadingHTTPServer

    sniffer = ThreadingHTTPServer(("127.0.0.1", 0), Sniffing)
    threading.Thread(target=sniffer.serve_forever, daemon=True).start()
    monkeypatch.setenv("MEMVAL_TEST_KEY", "sekrit-token")
    try:
        cfg = {
            "models": {
                "fake": {
                    "base": f"http://127.0.0.1:{sniffer.server_address[1]}",
                    "key_env": "MEMVAL_TEST_KEY",
                }
            }
        }
        gw = serve_gateway(cfg, port=0)
        try:
            code, _ = _post(
                f"http://127.0.0.1:{gw.server_address[1]}/chat/completions",
                {"model": "fake", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert code == 200
            assert seen["auth"] == "Bearer sekrit-token"
        finally:
            gw.shutdown()
    finally:
        sniffer.shutdown()


def test_rejects_non_https_upstream(tmp_path):
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(
        json.dumps({"models": {"x": {"base": "http://example.com/v1", "key_env": None}}})
    )
    with pytest.raises(GatewayConfigError, match="https"):
        load_gateway_config(str(cfg_path))


def test_rejects_non_loopback_bind():
    with pytest.raises(GatewayConfigError, match="loopback"):
        serve_gateway({"models": {}}, host="0.0.0.0", port=0)


def test_upstream_status_propagates(upstream):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Busy(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    busy = ThreadingHTTPServer(("127.0.0.1", 0), Busy)
    threading.Thread(target=busy.serve_forever, daemon=True).start()
    try:
        cfg = {"models": {"fake": {"base": f"http://127.0.0.1:{busy.server_address[1]}", "key_env": None}}}
        gw = serve_gateway(cfg, port=0)
        try:
            code, _ = _post(
                f"http://127.0.0.1:{gw.server_address[1]}/chat/completions",
                {"model": "fake", "messages": []},
            )
            assert code == 429
        finally:
            gw.shutdown()
    finally:
        busy.shutdown()
