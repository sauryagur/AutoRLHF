"""OpenRouterClient tests: no network, the SDK is faked."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from autorlhf.core.errors import (
    CredentialError,
    GenerationError,
    JudgeError,
    RateLimitedError,
    RemoteAPIError,
)
from autorlhf.core.types import ChatMessage
from autorlhf.llm import client as client_module
from autorlhf.llm.client import (
    DECISIONS_SERVER_URL,
    OpenRouterClient,
    extract_json_object,
)
from autorlhf.llm.schemas import DecisionsRequest, ScoreQuestion
from autorlhf.storage.cache import ResponseCache


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.raw_response = SimpleNamespace(status_code=status_code, headers=headers or {})


def chat_payload(text: str = "hello there", *, finish_reason: str = "stop") -> dict:
    return {
        "id": "gen-1",
        "model": "openai/gpt-4o-mini",
        "provider": "OpenAI",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "cost": 0.00042,
        },
    }


class FakeChat:
    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []

    def send(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, BaseException):
            raise result
        return result


class FakeDecisions:
    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, BaseException):
            raise result
        return result


class FakeApiKeys:
    def __init__(self, result=None, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error

    def get_current_key_metadata(self, **kwargs):
        if self.error is not None:
            raise self.error
        return self.result


class FakeSDK:
    def __init__(self, *, chat=None, decisions=None, api_keys=None) -> None:
        self.chat = chat or FakeChat([chat_payload()])
        self.alpha = SimpleNamespace(decisions=decisions or FakeDecisions([]))
        self.api_keys = api_keys or FakeApiKeys(SimpleNamespace(data=None))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda *_: None)


def make_client(*, cache=None, max_retries=2, sdk=None, trace=None) -> OpenRouterClient:
    client = OpenRouterClient(
        "sk-or-v1-test",
        max_retries=max_retries,
        cache=cache,
        run_id="run-1",
        trace=trace,
    )
    client._sdk = sdk or FakeSDK()
    return client


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #


def test_chat_maps_tokens_cost_and_truncation():
    client = make_client(sdk=FakeSDK(chat=FakeChat([chat_payload(finish_reason="length")])))
    result = client.chat(
        model="openai/gpt-4o-mini",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=16,
    )
    assert result.text == "hello there"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert result.truncated is True
    assert result.source == "openai/gpt-4o-mini"
    assert result.latency_s is not None
    summary = client.usage_summary()
    assert summary["calls"] == 1
    assert summary["total_tokens"] == 18
    assert summary["cost_usd"] == pytest.approx(0.00042)


def test_chat_sends_decoding_parameters_and_normalises_messages():
    sdk = FakeSDK()
    client = make_client(sdk=sdk)
    client.chat(
        model="m",
        messages=[{"role": "system", "content": "s"}, ChatMessage(role="user", content="u")],
        temperature=0.3,
        top_p=0.5,
        max_tokens=9,
        seed=4,
        response_format={"type": "json_object"},
    )
    call = sdk.chat.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert call["temperature"] == 0.3
    assert call["top_p"] == 0.5
    assert call["max_tokens"] == 9
    assert call["seed"] == 4
    assert call["response_format"] == {"type": "json_object"}


def test_chat_verbose_trace_contains_request_and_response_payloads():
    traces: list[str] = []
    client = make_client(
        sdk=FakeSDK(chat=FakeChat([chat_payload("response body")])),
        trace=traces.append,
    )
    client.chat(model="m", messages=[ChatMessage(role="user", content="private question")])

    trace = "\n".join(traces)
    assert "API request POST https://openrouter.ai/api/v1/chat/completions" in trace
    assert "private question" in trace
    assert "API response POST /api/v1/chat/completions" in trace
    assert "response body" in trace
    assert "sk-or-v1-test" not in trace


def test_chat_rejects_empty_completion():
    client = make_client(sdk=FakeSDK(chat=FakeChat([chat_payload("")])))
    with pytest.raises(GenerationError):
        client.chat(model="m", messages=[ChatMessage(role="user", content="hi")])


def test_chat_rejects_no_choices():
    payload = chat_payload()
    payload["choices"] = []
    client = make_client(sdk=FakeSDK(chat=FakeChat([payload])))
    with pytest.raises(GenerationError):
        client.chat(model="m", messages=[ChatMessage(role="user", content="hi")])


def test_chat_json_parses_fenced_json():
    body = 'Sure:\n```json\n{"prompts": [{"text": "hi"}]}\n```'
    client = make_client(sdk=FakeSDK(chat=FakeChat([chat_payload(body)])))
    data, result = client.chat_json(
        model="m", messages=[ChatMessage(role="user", content="x")], schema_hint="{}"
    )
    assert data == {"prompts": [{"text": "hi"}]}
    assert result.text == body


def test_chat_json_appends_schema_hint_and_requests_json_mode():
    sdk = FakeSDK(chat=FakeChat([chat_payload('{"a": 1}')]))
    client = make_client(sdk=sdk)
    client.chat_json(
        model="m", messages=[ChatMessage(role="user", content="x")], schema_hint="be brief"
    )
    call = sdk.chat.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"][-1] == {"role": "system", "content": "be brief"}


def test_chat_json_malformed_body_raises_generation_error():
    client = make_client(sdk=FakeSDK(chat=FakeChat([chat_payload("no json here")])))
    with pytest.raises(GenerationError):
        client.chat_json(
            model="m", messages=[ChatMessage(role="user", content="x")], schema_hint=""
        )


def test_extract_json_object_handles_embedded_and_nested_braces():
    text = 'prefix {"a": {"b": "}"}, "c": 2} suffix'
    assert extract_json_object(text) == {"a": {"b": "}"}, "c": 2}


# --------------------------------------------------------------------------- #
# retries / error mapping
# --------------------------------------------------------------------------- #


def test_rate_limit_then_success_retries():
    sdk = FakeSDK(chat=FakeChat([FakeHTTPError(429, {"retry-after": "0"}), chat_payload()]))
    traces: list[str] = []
    client = make_client(sdk=sdk, trace=traces.append)
    result = client.chat(model="m", messages=[ChatMessage(role="user", content="hi")])
    assert result.text == "hello there"
    assert len(sdk.chat.calls) == 2
    assert any("API retry chat completion (m): rate limited" in entry for entry in traces)


def test_exhausted_retries_raise_rate_limited():
    sdk = FakeSDK(chat=FakeChat([FakeHTTPError(429)]))
    client = make_client(sdk=sdk, max_retries=2)
    with pytest.raises(RateLimitedError) as excinfo:
        client.chat(model="m", messages=[ChatMessage(role="user", content="hi")])
    assert excinfo.value.retry_after is None
    assert len(sdk.chat.calls) == 3  # initial attempt + 2 retries


def test_server_error_is_retried_then_succeeds():
    sdk = FakeSDK(chat=FakeChat([FakeHTTPError(503), FakeHTTPError(502), chat_payload()]))
    client = make_client(sdk=sdk, max_retries=3)
    assert client.chat(model="m", messages=[ChatMessage(role="user", content="hi")]).text


def test_unauthorized_raises_credential_error_without_retry():
    sdk = FakeSDK(chat=FakeChat([FakeHTTPError(401)]))
    client = make_client(sdk=sdk)
    with pytest.raises(CredentialError) as excinfo:
        client.chat(model="m", messages=[ChatMessage(role="user", content="hi")])
    assert len(sdk.chat.calls) == 1
    assert excinfo.value.hint


def test_non_retryable_error_raises_remote_api_error():
    sdk = FakeSDK(chat=FakeChat([FakeHTTPError(400)]))
    client = make_client(sdk=sdk)
    with pytest.raises(RemoteAPIError):
        client.chat(model="m", messages=[ChatMessage(role="user", content="hi")])


def test_no_api_key_raises_credential_error():
    with pytest.raises(CredentialError):
        OpenRouterClient("   ")


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #


def test_cache_second_call_makes_no_sdk_call(store):
    cache = ResponseCache(store)
    sdk = FakeSDK()
    client = make_client(cache=cache, sdk=sdk)
    first = client.chat(model="m", messages=[ChatMessage(role="user", content="hi")])
    second = client.chat(model="m", messages=[ChatMessage(role="user", content="hi")])
    assert len(sdk.chat.calls) == 1
    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text


def test_cache_key_includes_decoding_parameters(store):
    cache = ResponseCache(store)
    sdk = FakeSDK()
    client = make_client(cache=cache, sdk=sdk)
    client.chat(model="m", messages=[ChatMessage(role="user", content="hi")], temperature=0.2)
    client.chat(model="m", messages=[ChatMessage(role="user", content="hi")], temperature=0.9)
    assert len(sdk.chat.calls) == 2


def test_costs_accumulate_across_distinct_calls(store):
    client = make_client(cache=ResponseCache(store))
    client.chat(model="m", messages=[ChatMessage(role="user", content="a")])
    client.chat(model="m", messages=[ChatMessage(role="user", content="b")])
    assert client.usage_summary()["cost_usd"] == pytest.approx(0.00084)


def test_cache_entries_counts_stored_rows_not_this_process_lookups(store):
    cache = ResponseCache(store)
    client = make_client(cache=cache)
    for index in range(4):
        cache.put("chat.completion", f"k{index}", {"text": "x"})

    summary = client.usage_summary()

    assert cache.hits == 0 and cache.misses == 0
    assert summary["cache_entries"] == 4.0


# --------------------------------------------------------------------------- #
# decisions
# --------------------------------------------------------------------------- #


def _decisions_request() -> DecisionsRequest:
    return DecisionsRequest(
        model="typesafe/jev-1.13",
        state={"task": "x"},
        questions={
            "urgency": ScoreQuestion(instructions="how urgent", criteria=["low", "middle", "high"])
        },
        session_id="run-1",
    )


def _response(answers: dict) -> dict:
    return {
        "id": "gen-dec-1",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": answers,
        "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 0.00001},
    }


def test_decide_uses_the_decisions_server_url_and_validates():
    decisions = FakeDecisions([_response({"urgency": {"type": "score", "score": 2.0}})])
    traces: list[str] = []
    client = make_client(sdk=FakeSDK(decisions=decisions), trace=traces.append)
    response = client.decide(_decisions_request())
    trace = "\n".join(traces)
    assert "API request POST https://openrouter.ai/api/alpha/decisions" in trace
    assert '"task": "x"' in trace
    assert "API response POST /api/alpha/decisions" in trace
    assert "gen-dec-1" in trace
    call = decisions.calls[0]
    assert call["server_url"] == DECISIONS_SERVER_URL
    assert call["model"] == "typesafe/jev-1.13"
    assert call["state"] == {"task": "x"}
    assert call["questions"] == {
        "urgency": {
            "type": "score",
            "instructions": "how urgent",
            "criteria": ["low", "middle", "high"],
        }
    }
    assert call["session_id"] == "run-1"
    assert response.model == "typesafe/jev-1.13-20260917"
    assert response.usage.input_tokens == 10


def test_decide_missing_answer_key_raises():
    decisions = FakeDecisions([_response({})])
    client = make_client(sdk=FakeSDK(decisions=decisions))
    with pytest.raises(JudgeError):
        client.decide(_decisions_request())


def test_decide_unexpected_answer_type_raises():
    decisions = FakeDecisions([_response({"urgency": {"type": "noul", "noul": 0.5}})])
    client = make_client(sdk=FakeSDK(decisions=decisions))
    with pytest.raises(JudgeError):
        client.decide(_decisions_request())


def test_decide_caches_and_counts_usage(store):
    cache = ResponseCache(store)
    decisions = FakeDecisions([_response({"urgency": {"type": "score", "score": 1.0}})])
    client = make_client(cache=cache, sdk=FakeSDK(decisions=decisions))
    client.decide(_decisions_request())
    client.decide(_decisions_request())
    assert len(decisions.calls) == 1
    assert client.usage_summary()["cost_usd"] == pytest.approx(0.00001)


def test_decide_many_preserves_order_and_reports_progress():
    requests = [
        DecisionsRequest(
            model="typesafe/jev-1.13",
            state={"i": i},
            questions={"urgency": ScoreQuestion(instructions="x", criteria=["a", "b"])},
        )
        for i in range(3)
    ]
    decisions = FakeDecisions(
        [
            _response({"urgency": {"type": "score", "score": 0.0}}),
            _response({"urgency": {"type": "score", "score": 1.0}}),
            _response({"urgency": {"type": "score", "score": 2.0}}),
        ]
    )
    client = make_client(sdk=FakeSDK(decisions=decisions))
    seen: list[tuple[int, int]] = []
    responses = client.decide_many(
        requests, on_progress=lambda done, total: seen.append((done, total))
    )
    assert [r.answers["urgency"].score for r in responses] == [0.0, 1.0, 2.0]
    assert len(decisions.calls) == 3
    assert seen[-1] == (3, 3)


def test_decide_many_cancel_event_skips_remaining():
    requests = [
        DecisionsRequest(
            model="typesafe/jev-1.13",
            state={"i": i},
            questions={"urgency": ScoreQuestion(instructions="x", criteria=["a", "b"])},
        )
        for i in range(3)
    ]
    decisions = FakeDecisions([_response({"urgency": {"type": "score", "score": 1.0}})])
    client = make_client(sdk=FakeSDK(decisions=decisions))
    import threading

    cancel = threading.Event()
    cancel.set()
    assert client.decide_many(requests, cancel_event=cancel) == []
    assert decisions.calls == []


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #


def test_verify_credentials_success():
    sdk = FakeSDK(api_keys=FakeApiKeys(SimpleNamespace(data=SimpleNamespace(label="autorlhf"))))
    client = make_client(sdk=sdk)
    ok, message = client.verify_credentials()
    assert ok is True
    assert "accepted" in message
    assert "autorlhf" in message


def test_verify_credentials_failure_is_actionable():
    sdk = FakeSDK(api_keys=FakeApiKeys(error=FakeHTTPError(401)))
    client = make_client(sdk=sdk)
    ok, message = client.verify_credentials()
    assert ok is False
    assert "credential" in message.lower()


def test_close_is_idempotent():
    client = make_client()
    client.close()
    client.close()
    assert client._sdk is None


def test_json_import_is_available_for_message_payloads():
    assert json.loads(json.dumps({"role": "user"})) == {"role": "user"}
