"""Judge-layer tests.

The Jev client is exercised end to end against the fake System One fixture;
blinding, answer normalization, and the flag policy are unit-tested directly.
No real API calls happen here.
"""
import json
import re
import threading
from pathlib import Path

import pytest

from memval.cli import main
from memval.judge import (
    QUESTIONS,
    RUBRIC,
    JevClient,
    JudgeError,
    blind_transcript,
    bundle_version,
    judge_results,
    judge_state,
    parse_answers,
)
from memval.tasks import validate_task
from tests.fixtures import fake_jev, fake_upstream

TASK = validate_task(
    {
        "id": "recall-t1",
        "type": "recall",
        "sessions": [
            ["Remember that I prefer pepperjack cheese on sandwiches."],
            ["What cheese should go on the sandwiches?"],
        ],
        "planted_facts": ["I prefer pepperjack cheese", "pepperjack"],
        "expected": {"mode": "contains_all", "keywords": ["pepperjack"]},
        "control": True,
    }
)

TRANSCRIPT = (
    "User: Remember that I prefer pepperjack cheese on sandwiches.\n"
    'Assistant called memory_save({"key": "facts.md", "content": "prefers pepperjack"})\n'
    "Tool: Saved to memlawb namespace user:eval/recall-t1\n"
    "Assistant: Noted.\n"
    "--- new session ---\n"
    "User: What cheese should go on the sandwiches?\n"
    'Assistant called signet_recall({"query": "cheese preference"})\n'
    "Tool: signet recall hit: prefers pepperjack\n"
    "Assistant: Pepperjack.\n"
)


@pytest.fixture
def jev_server():
    server = fake_jev.serve(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def _client(server, **kwargs):
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/systemone"
    return JevClient(api_key="test-key", endpoint=url, **kwargs)


def test_missing_api_key_raises_at_construction(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        JevClient()


def test_happy_path_parses_all_four_answers(jev_server):
    resp = _client(jev_server).evaluate(judge_state(TASK, TRANSCRIPT))
    assert resp["model"] == "jev-latest"
    assert resp["usage"]["input_tokens"] > 0

    parsed = parse_answers(resp)
    score = parsed["task_success"]
    assert score["type"] == "score"
    assert score["score"] == 3.6
    assert score["confidence"] == 0.82
    assert score["distribution"]["3"] == 0.55
    assert score["flagged"] is False

    assert parsed["memory_used"] == {
        "type": "noul",
        "probability": 0.9,
        "flagged": False,
    }
    assert parsed["confabulated"]["probability"] == 0.08

    fc = parsed["failure_class"]
    assert fc["type"] == "choice"
    assert fc["choice"] == "no-failure"
    assert fc["confidence"] == 0.9
    assert fc["flagged"] is False


def test_request_body_shape(jev_server):
    """Spike-verified: score.criteria is a list, choice.criteria a dict."""
    _client(jev_server).evaluate("some state")
    assert len(jev_server.jev_state.requests) == 1
    req = jev_server.jev_state.requests[0]
    body = req["body"]
    assert body["model"] == "jev-latest"
    assert body["state"] == "some state"
    qs = body["questions"]
    assert isinstance(qs["task_success"]["criteria"], list)
    assert len(qs["task_success"]["criteria"]) == 5
    assert isinstance(qs["failure_class"]["criteria"], dict)
    assert set(qs["failure_class"]["criteria"]) == {
        "no-failure",
        "retrieval-miss",
        "wrong-memory",
        "confabulation",
        "tool-error",
        "refused",
    }
    assert "criteria" not in qs["memory_used"]
    assert "criteria" not in qs["confabulated"]
    # Auth travels in the header only; the key never enters the payload.
    assert req["headers"]["Authorization"] == "Bearer test-key"
    assert "test-key" not in json.dumps(body)


def _answers(score_conf=None, score=2.0, noul_p=0.9, choice_conf=None):
    return {
        "answers": {
            "task_success": {
                "score": score,
                **({"confidence": score_conf} if score_conf is not None else {}),
            },
            "memory_used": {"probability": noul_p},
            "confabulated": {"probability": 0.1},
            "failure_class": {
                "choice": "retrieval-miss",
                **({"confidence": choice_conf} if choice_conf is not None else {}),
            },
        }
    }


@pytest.mark.parametrize(
    "p,flagged",
    [(0.35, True), (0.5, True), (0.65, True), (0.34, False), (0.66, False), (0.9, False)],
)
def test_noul_flag_band(p, flagged):
    parsed = parse_answers(_answers(noul_p=p))
    assert parsed["memory_used"]["flagged"] is flagged


@pytest.mark.parametrize(
    "conf,flagged", [(0.5, True), (0.59, True), (0.6, False), (0.7, False)]
)
def test_score_confidence_flag(conf, flagged):
    parsed = parse_answers(_answers(score_conf=conf, choice_conf=0.9))
    assert parsed["task_success"]["flagged"] is flagged


def test_choice_confidence_flag():
    parsed = parse_answers(_answers(score_conf=0.9, choice_conf=0.4))
    assert parsed["failure_class"]["flagged"] is True


def test_missing_confidence_flags_conservatively():
    """An answer with no confidence field must not read as trusted."""
    parsed = parse_answers(_answers(score_conf=None, choice_conf=0.9))
    assert parsed["task_success"]["flagged"] is True
    assert parsed["task_success"]["confidence"] == 0.0


def test_score_from_distribution_when_no_interpolated_score():
    resp = _answers(score_conf=0.9)
    resp["answers"]["task_success"] = {
        "confidence": 0.9,
        "distribution": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0},
    }
    parsed = parse_answers(resp)
    assert parsed["task_success"]["score"] == 4.0


def test_choice_falls_back_to_distribution_argmax():
    resp = _answers(score_conf=0.9, choice_conf=0.9)
    del resp["answers"]["failure_class"]["choice"]
    resp["answers"]["failure_class"]["distribution"] = {
        "no-failure": 0.1,
        "tool-error": 0.9,
    }
    parsed = parse_answers(resp)
    assert parsed["failure_class"]["choice"] == "tool-error"


def test_blind_transcript_normalizes_tool_names():
    out = blind_transcript(TRANSCRIPT)
    assert "called save(" in out
    assert "called recall(" in out
    lowered = out.casefold()
    for banned in ("memlawb", "signet", "memory_save", "memory_recall", "signet_recall"):
        assert banned not in lowered


def test_blind_transcript_scrubs_all_known_tool_names():
    text = "\n".join(
        f"Assistant called {name}({{}})"
        for name in (
            "memory_save",
            "memory_recall",
            "memory_search",
            "memory_list",
            "memory_delete",
            "memory_guide",
            "signet_save",
            "signet_recall",
            "signet_search",
            "signet_list",
            "signet_delete",
            "signet_config_get",
            "signet_config_set",
            "signet_grant_list",
            "signet_grant_record",
        )
    )
    out = blind_transcript(text).casefold()
    assert not re.search(r"(memory|signet)_", out)
    for verb in ("save", "recall", "search", "list", "delete"):
        assert f"called {verb}(" in out


def test_blind_transcript_scrubs_backend_strings_in_args_and_results():
    text = (
        'Assistant called memory_recall({"namespace": "memlawb:user:eval", '
        '"backend": "SIGNET"})\n'
        "Tool: memlawb stored under signet custody at MEMLAWB_URL\n"
    )
    out = blind_transcript(text)
    assert not re.search(r"(?i)memlawb|signet", out)


def test_blind_transcript_preserves_generic_content():
    out = blind_transcript("User: my favorite memory is the beach\n")
    assert "my favorite memory is the beach" in out


def test_judge_state_carries_context_and_blinds():
    legs = [
        "User: Remember that I prefer pepperjack cheese on sandwiches.\n"
        'Assistant called memory_save({"key": "facts.md"})\n'
        "Tool: saved\n",
        "User: What cheese should go on the sandwiches?\n"
        'Assistant called signet_recall({"query": "cheese"})\n'
        "Tool: prefers pepperjack\n"
        "Assistant: Pepperjack.\n",
    ]
    state = judge_state(TASK, legs)
    assert "\n--- new session ---\n" in state
    assert "What cheese should go on the sandwiches?" in state
    assert "pepperjack" in state  # planted fact + expected keyword surface
    assert not re.search(r"(?i)memlawb|signet|memory_save", state)


def test_judge_state_accepts_single_string():
    state = judge_state(TASK, "User: hi\nAssistant: hello\n")
    # The preamble names the marker inline; a real separator is a bare line.
    assert "\n--- new session ---\n" not in state
    assert "User: hi" in state


def test_judge_state_describes_expected_modes():
    exact = validate_task(
        {
            "id": "exact-t1",
            "type": "recall",
            "sessions": [["q"], ["answer?"]],
            "planted_facts": ["x"],
            "expected": {"mode": "exact", "value": "42"},
            "control": True,
        }
    )
    assert '"42"' in judge_state(exact, "t")
    none_task = validate_task(
        {
            "id": "iso-t1",
            "type": "isolation",
            "sessions": [["q"], ["answer?"]],
            "planted_facts": ["secret-a"],
            "expected": {"mode": "contains_none", "forbidden": ["secret-a"]},
            "control": True,
            "users": ["a", "b"],
        }
    )
    state = judge_state(none_task, "t")
    assert "secret-a" in state
    assert "never" in state.casefold()


def test_judge_state_handles_empty_planted_facts():
    task = validate_task(
        {
            "id": "free-t1",
            "type": "follow-through",
            "sessions": [["Set up the thing."], ["Did it work?"]],
            "planted_facts": [],
            "expected": {"mode": "contains_all", "keywords": ["done"]},
            "control": False,
        }
    )
    state = judge_state(task, "User: hi\n")
    assert "none planted" in state


def test_bundle_version_is_stable_hex():
    v1 = bundle_version()
    v2 = bundle_version()
    assert v1 == v2
    assert re.fullmatch(r"[0-9a-f]+", v1)
    # The rubric text feeds the hash: a different bundle hashes differently.
    import hashlib

    other = hashlib.sha256(
        json.dumps({"rubric": "other", "questions": QUESTIONS}, sort_keys=True).encode()
    ).hexdigest()[:16]
    assert v1 != other
    assert RUBRIC  # rubric text shipped, not a file reference


def test_4xx_raises_without_retry(jev_server):
    jev_server.jev_state.responder = lambda body: (403, {"error": "forbidden"})
    client = _client(jev_server)
    with pytest.raises(RuntimeError, match="403"):
        client.evaluate("state", retries=3, backoff=0)
    assert len(jev_server.jev_state.requests) == 1


def test_429_retries_then_raises(jev_server):
    jev_server.jev_state.responder = lambda body: (429, {"error": "slow down"})
    client = _client(jev_server)
    with pytest.raises(RuntimeError, match="429"):
        client.evaluate("state", retries=2, backoff=0)
    assert len(jev_server.jev_state.requests) == 3


def test_500_retries_then_raises(jev_server):
    jev_server.jev_state.responder = lambda body: (500, {"error": "boom"})
    client = _client(jev_server)
    with pytest.raises(RuntimeError, match="500"):
        client.evaluate("state", retries=2, backoff=0)
    assert len(jev_server.jev_state.requests) == 3


def test_500_recovers_on_later_attempt(jev_server):
    calls = {"n": 0}

    def flaky(body):
        calls["n"] += 1
        if calls["n"] < 3:
            return 500, {"error": "boom"}
        return 200, fake_jev.default_response(body)

    jev_server.jev_state.responder = flaky
    resp = _client(jev_server).evaluate("state", retries=3, backoff=0)
    assert resp["model"] == "jev-latest"
    assert calls["n"] == 3


def test_network_error_retries_then_raises():
    # Nothing listens on this port: every attempt is a connection error.
    client = JevClient(api_key="k", endpoint="http://127.0.0.1:1/v1/systemone")
    with pytest.raises(RuntimeError, match="failed"):
        client.evaluate("state", retries=1, backoff=0)


def test_non_dict_body_raises(jev_server):
    jev_server.jev_state.responder = lambda body: (200, ["not", "a", "dict"])
    with pytest.raises(RuntimeError, match="non-dict"):
        _client(jev_server).evaluate("state", retries=3, backoff=0)
    # Malformed bodies fail fast: no point retrying a shape that will not heal.
    assert len(jev_server.jev_state.requests) == 1


def test_parse_answers_rejects_malformed_response():
    with pytest.raises(JudgeError, match="not a JSON object"):
        parse_answers(["answers"])
    with pytest.raises(JudgeError, match="answers"):
        parse_answers({"model": "jev-latest"})
    with pytest.raises(JudgeError, match="memory_used"):
        parse_answers({"answers": {"task_success": {"score": 1, "confidence": 0.9}}})
    with pytest.raises(JudgeError, match="probability"):
        parse_answers(
            {
                "answers": {
                    "task_success": {"score": 1, "confidence": 0.9},
                    "memory_used": {},
                    "confabulated": {"probability": 0.1},
                    "failure_class": {"choice": "no-failure", "confidence": 0.9},
                }
            }
        )


# --- U3: judge_results orchestration ---------------------------------------


def _seed_cell(tmp_path: Path, task_id: str, condition: str, variant: str,
               outcome: str = "pass", transcripts: int = 2) -> None:
    results = tmp_path / "run.jsonl"
    with open(results, "a") as f:
        f.write(
            json.dumps(
                {
                    "task_id": task_id,
                    "condition": condition,
                    "variant": variant,
                    "outcome": outcome,
                }
            )
            + "\n"
        )
    transcripts_dir = tmp_path / "run.transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    for leg in range(transcripts):
        (transcripts_dir / f"{task_id}-{condition}-{variant}-leg{leg}.txt").write_text(
            f"User: prompt leg {leg}\nAssistant called recall({{}})\n"
            "Tool: pepperjack\nAssistant: Pepperjack.\n"
        )


def test_judge_results_covers_judgeable_cells_only(jev_server, tmp_path):
    _seed_cell(tmp_path, "recall-01", "none", "live", "pass")
    _seed_cell(tmp_path, "recall-01", "memlawb", "live", "fail")
    _seed_cell(tmp_path, "recall-02", "none", "live", "error")
    _seed_cell(tmp_path, "recall-02", "memlawb", "live", "not_run")

    sidecar = judge_results(
        tmp_path / "run.jsonl", client=_client(jev_server), progress=None
    )
    recs = [json.loads(l) for l in sidecar.read_text().splitlines() if l.strip()]
    keys = {(r["task_id"], r["condition"], r["variant"]) for r in recs}
    assert keys == {
        ("recall-01", "none", "live"),
        ("recall-01", "memlawb", "live"),
    }
    for r in recs:
        assert r["model"] == "jev-latest"
        assert r["bundle_version"] == bundle_version()
        assert r["usage"]["input_tokens"] == 1234
        assert isinstance(r["latency_ms"], int)
        assert set(r["answers"]) == set(QUESTIONS)


def test_judge_results_resume_is_noop(jev_server, tmp_path):
    _seed_cell(tmp_path, "recall-01", "none", "live")
    client = _client(jev_server)
    judge_results(tmp_path / "run.jsonl", client=client, progress=None)
    n = len(jev_server.jev_state.requests)
    judge_results(tmp_path / "run.jsonl", client=client, progress=None)
    assert len(jev_server.jev_state.requests) == n


def test_judge_results_missing_transcript_marks_error(jev_server, tmp_path):
    _seed_cell(tmp_path, "recall-01", "none", "live")
    _seed_cell(tmp_path, "recall-02", "none", "live", transcripts=0)

    sidecar = judge_results(
        tmp_path / "run.jsonl", client=_client(jev_server), progress=None
    )
    recs = {r["task_id"]: r for r in
            (json.loads(l) for l in sidecar.read_text().splitlines())}
    assert "judge_error" in recs["recall-02"]
    assert "judge_error" not in recs["recall-01"]


def test_judge_results_unknown_task_marks_error(jev_server, tmp_path):
    _seed_cell(tmp_path, "nope-99", "none", "live")
    sidecar = judge_results(
        tmp_path / "run.jsonl", client=_client(jev_server), progress=None
    )
    (rec,) = [json.loads(l) for l in sidecar.read_text().splitlines() if l.strip()]
    assert "battery" in rec["judge_error"]


def test_judge_results_does_not_mutate_results(jev_server, tmp_path):
    _seed_cell(tmp_path, "recall-01", "none", "live")
    before = (tmp_path / "run.jsonl").read_bytes()
    judge_results(tmp_path / "run.jsonl", client=_client(jev_server), progress=None)
    assert (tmp_path / "run.jsonl").read_bytes() == before


def test_judge_results_warns_on_bundle_version_mismatch(jev_server, tmp_path, capsys):
    _seed_cell(tmp_path, "recall-01", "none", "live")
    sidecar = tmp_path / "run.judge.jsonl"
    sidecar.write_text(
        json.dumps(
            {
                "task_id": "recall-02",
                "condition": "none",
                "variant": "live",
                "bundle_version": "stale0000version",
                "answers": {},
            }
        )
        + "\n"
    )
    judge_results(tmp_path / "run.jsonl", client=_client(jev_server), progress=None)
    assert "bundle versions" in capsys.readouterr().err


def test_judge_cli_end_to_end(jev_server, tmp_path, monkeypatch):
    """Fixture run -> persisted transcripts -> `memval judge` -> sidecar."""
    upstream = fake_upstream.serve(0)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    cfg = tmp_path / "memval.config.json"
    cfg.write_text(
        json.dumps(
            {
                "models": {
                    "fake": {
                        "base": f"http://127.0.0.1:{upstream.server_address[1]}",
                        "key_env": None,
                    }
                }
            }
        )
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    endpoint = f"http://127.0.0.1:{jev_server.server_address[1]}/v1/systemone"
    try:
        results = tmp_path / "run.jsonl"
        rc = main(
            [
                "run", "--config", str(cfg), "--model", "fake",
                "--results", str(results), "--fixture",
                "--conditions", "none", "--tasks", "recall-01",
            ]
        )
        assert rc == 0
        rc = main(["judge", str(results), "--endpoint", endpoint])
        assert rc == 0
        sidecar = tmp_path / "run.judge.jsonl"
        (rec,) = [json.loads(l) for l in sidecar.read_text().splitlines()]
        assert rec["task_id"] == "recall-01"
        assert rec["condition"] == "none"
        assert rec["answers"]["task_success"]["score"] == 3.6
    finally:
        upstream.shutdown()


def test_judge_cli_fails_closed_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    results = tmp_path / "run.jsonl"
    results.write_text("")
    assert main(["judge", str(results)]) == 2


def test_endpoint_rejects_non_loopback_http():
    JevClient(api_key="k", endpoint="https://api.typesafe.ai/v1/systemone")
    JevClient(api_key="k", endpoint="http://127.0.0.1:9/x")
    JevClient(api_key="k", endpoint="http://localhost:9/x")
    with pytest.raises(RuntimeError, match="https"):
        JevClient(api_key="k", endpoint="http://evil.example.com/v1/systemone")
    with pytest.raises(RuntimeError, match="https"):
        JevClient(api_key="k", endpoint="ftp://127.0.0.1/x")


def test_terminal_errors_not_reappended(jev_server, tmp_path):
    _seed_cell(tmp_path, "nope-99", "none", "live")
    client = _client(jev_server)
    sidecar = judge_results(tmp_path / "run.jsonl", client=client, progress=None)
    judge_results(tmp_path / "run.jsonl", client=client, progress=None)
    recs = [
        json.loads(l) for l in sidecar.read_text().splitlines() if l.strip()
    ]
    assert len(recs) == 1
    assert "judge_error" in recs[0]


def test_transient_errors_retry_on_next_run(tmp_path):
    bad = fake_jev.serve(0, responder=lambda b: (500, {"error": "boom"}))
    threading.Thread(target=bad.serve_forever, daemon=True).start()
    try:
        _seed_cell(tmp_path, "recall-01", "none", "live")
        client = _client(bad)
        sidecar = judge_results(
            tmp_path / "run.jsonl", client=client, progress=None
        )
        judge_results(tmp_path / "run.jsonl", client=client, progress=None)
        recs = [
            json.loads(l)
            for l in sidecar.read_text().splitlines()
            if l.strip()
        ]
        # The API error is retryable, so the second run re-judged and
        # appended a fresh error record.
        assert len(recs) == 2
        assert all("judge_error" in r for r in recs)
    finally:
        bad.shutdown()


def test_out_of_range_answer_records_judge_error(tmp_path):
    answers = dict(fake_jev.DEFAULT_ANSWERS)
    answers["task_success"] = {"score": 9.0, "confidence": 0.9}
    srv = fake_jev.serve(
        0,
        responder=lambda b: (
            200, {"model": "jev-latest", "answers": answers, "usage": {}}
        ),
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        _seed_cell(tmp_path, "recall-01", "none", "live")
        sidecar = judge_results(
            tmp_path / "run.jsonl", client=_client(srv), progress=None
        )
        (rec,) = [
            json.loads(l)
            for l in sidecar.read_text().splitlines()
            if l.strip()
        ]
        assert "outside" in rec["judge_error"]
    finally:
        srv.shutdown()
