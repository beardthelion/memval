"""Judge-layer tests.

The Jev client is exercised end to end against the fake System One fixture;
blinding, answer normalization, and the flag policy are unit-tested directly.
No real API calls happen here.
"""
import json
import re
import threading

import pytest

from memval.judge import (
    QUESTIONS,
    RUBRIC,
    JevClient,
    JudgeError,
    blind_transcript,
    bundle_version,
    judge_state,
    parse_answers,
)
from memval.tasks import validate_task
from tests.fixtures import fake_jev

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
