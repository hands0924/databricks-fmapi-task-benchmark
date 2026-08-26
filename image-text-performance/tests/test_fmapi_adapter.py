"""FMAPI 어댑터 테스트 (src/adapters/fmapi.py) — 네트워크·CLI 없이 스텁으로 검증."""

import json
import subprocess

import httpx
import pytest
from src.adapters import fmapi as fmapi_mod
from src.adapters.fmapi import (
    FMAPIClient,
    FMAPIError,
    _normalize_content,
    build_image_message,
    build_text_message,
)

AUTH_ENV = json.dumps({"env": {"DATABRICKS_HOST": "https://ws.databricks.com/"}})
AUTH_TOKEN = json.dumps({"access_token": "tok-123"})


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json


class FakeHTTPClient:
    """httpx.Client 대역: 미리 정한 응답(또는 예외)을 순서대로 돌려준다."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def close(self):
        self.closed = True


@pytest.fixture
def client_factory(monkeypatch):
    """인증·HTTP·sleep을 대체한 FMAPIClient를 만드는 팩토리."""
    monkeypatch.setattr(fmapi_mod, "_get_workspace_auth",
                        lambda profile: ("https://ws.databricks.com", "tok-123"))
    monkeypatch.setattr(fmapi_mod.time, "sleep", lambda s: None)

    def make(responses, **kwargs):
        client = FMAPIClient("prof", **kwargs)
        client._client = FakeHTTPClient(responses)
        return client

    return make


def ok_response(message, *, usage=None, finish_reason="stop", headers=None, body_id=None):
    data = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage if usage is not None else {"prompt_tokens": 7, "completion_tokens": 3},
    }
    if body_id:
        data["id"] = body_id
    return FakeResponse(json_data=data, headers=headers)


# --------------------------------------------------------- _normalize_content ---
def test_string_content_is_stripped():
    assert _normalize_content({"content": "  답변  "}) == "답변"


def test_list_content_keeps_only_text_parts():
    message = {"content": [
        {"type": "reasoning", "summary": "생각"},
        {"type": "text", "text": "최종 "},
        {"type": "text", "text": "답"},
    ]}
    assert _normalize_content(message) == "최종 답"


def test_list_content_without_text_falls_back_to_reasoning():
    message = {"content": [{"type": "reasoning", "summary": "생각 중"}],
               "reasoning_content": " 잘린 답 "}
    assert _normalize_content(message) == "잘린 답"


def test_empty_content_falls_back_to_reasoning_content():
    assert _normalize_content({"content": "", "reasoning_content": "glm 답"}) == "glm 답"


def test_missing_content_and_reasoning_yields_empty_string():
    assert _normalize_content({}) == ""


def test_list_content_ignores_non_dict_parts():
    assert _normalize_content({"content": ["문자열", {"type": "text", "text": "ok"}]}) == "ok"


# ------------------------------------------------------ _get_workspace_auth ---
def test_workspace_auth_strips_trailing_slash(monkeypatch):
    outputs = {("auth", "env"): AUTH_ENV, ("auth", "token"): AUTH_TOKEN}
    calls = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs[(cmd[1], cmd[2])], stderr="")

    monkeypatch.setattr(fmapi_mod.subprocess, "run", fake_run)
    assert fmapi_mod._get_workspace_auth("myprof") == ("https://ws.databricks.com", "tok-123")
    assert calls[0][-2:] == ["--profile", "myprof"]


def test_workspace_auth_raises_on_cli_failure(monkeypatch):
    monkeypatch.setattr(fmapi_mod.subprocess, "run",
                        lambda cmd, capture_output, text: subprocess.CompletedProcess(
                            cmd, 1, stdout="", stderr="no such profile"))
    with pytest.raises(FMAPIError, match="no such profile"):
        fmapi_mod._get_workspace_auth("bad")


# ---------------------------------------------------------------- chat() ---
def test_chat_posts_to_endpoint_invocations_url(client_factory):
    client = client_factory([ok_response({"content": "hi"})])
    client.chat("databricks-claude-opus-5", build_text_message("q"))
    call = client._client.calls[0]
    assert call["url"] == "https://ws.databricks.com/serving-endpoints/databricks-claude-opus-5/invocations"
    assert call["headers"]["Authorization"] == "Bearer tok-123"
    assert call["headers"]["Content-Type"] == "application/json"


def test_chat_payload_merges_extra_reasoning_params(client_factory):
    client = client_factory([ok_response({"content": "hi"})])
    client.chat("ep", [{"role": "user", "content": "q"}], max_tokens=64,
                extra_params={"thinking": {"type": "enabled", "budget_tokens": 2048}})
    payload = client._client.calls[0]["json"]
    assert payload["max_tokens"] == 64
    assert payload["messages"] == [{"role": "user", "content": "q"}]
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_chat_returns_normalized_response(client_factory):
    client = client_factory([ok_response(
        {"content": [{"type": "text", "text": "정답"}]},
        usage={"prompt_tokens": 11, "completion_tokens": 2},
        headers={"x-request-id": "req-abc"},
    )])
    resp = client.chat("ep", build_text_message("q"))
    assert resp.text == "정답"
    assert resp.request_id == "req-abc"
    assert resp.finish_reason == "stop"
    assert resp.usage == {"prompt_tokens": 11, "completion_tokens": 2}
    assert resp.raw["choices"][0]["finish_reason"] == "stop"


def test_request_id_falls_back_to_body_id(client_factory):
    client = client_factory([ok_response({"content": "hi"}, body_id="chatcmpl-9")])
    assert client.chat("ep", build_text_message("q")).request_id == "chatcmpl-9"


def test_request_id_none_when_absent(client_factory):
    client = client_factory([ok_response({"content": "hi"})])
    assert client.chat("ep", build_text_message("q")).request_id is None


def test_missing_usage_becomes_empty_dict(client_factory):
    client = client_factory([FakeResponse(json_data={"choices": [{"message": {"content": "x"}}]})])
    resp = client.chat("ep", build_text_message("q"))
    assert resp.usage == {}
    assert resp.finish_reason is None


@pytest.mark.parametrize("data", [{}, {"choices": []}])
def test_malformed_response_raises(client_factory, data):
    client = client_factory([FakeResponse(json_data=data)])
    with pytest.raises(FMAPIError, match="예상치 못한 응답 형태"):
        client.chat("ep", build_text_message("q"))


@pytest.mark.parametrize("status", [500, 503, 429])
def test_retries_transient_statuses_then_succeeds(client_factory, status):
    client = client_factory([FakeResponse(status_code=status, text="busy"),
                             ok_response({"content": "ok"})])
    assert client.chat("ep", build_text_message("q")).text == "ok"
    assert len(client._client.calls) == 2


def test_retries_timeouts_then_succeeds(client_factory):
    client = client_factory([httpx.TimeoutException("glm은 느리다"),
                             ok_response({"content": "ok"})])
    assert client.chat("ep", build_text_message("q")).text == "ok"


def test_client_errors_fail_immediately_without_retry(client_factory):
    client = client_factory([FakeResponse(status_code=400, text="bad request"),
                             ok_response({"content": "never"})])
    with pytest.raises(FMAPIError, match="HTTP 400"):
        client.chat("ep", build_text_message("q"))
    assert len(client._client.calls) == 1


def test_gives_up_after_max_retries(client_factory):
    client = client_factory([FakeResponse(status_code=500, text="boom")] * 3,
                            max_retries=3)
    with pytest.raises(FMAPIError, match="재시도 3회 모두 실패"):
        client.chat("ep", build_text_message("q"))
    assert len(client._client.calls) == 3


def test_backoff_is_exponential(client_factory, monkeypatch):
    slept = []
    monkeypatch.setattr(fmapi_mod.time, "sleep", slept.append)
    client = client_factory([FakeResponse(status_code=500, text="boom")] * 3,
                            max_retries=3, backoff_initial_seconds=0.5)
    with pytest.raises(FMAPIError):
        client.chat("ep", build_text_message("q"))
    assert slept == [0.5, 1.0, 2.0]


# ------------------------------------------------------ context manager ---
def test_context_manager_closes_http_client(client_factory):
    client = client_factory([])
    with client as entered:
        assert entered is client
    assert client._client.closed


# --------------------------------------------------------- message builders ---
def test_build_text_message():
    assert build_text_message("질문") == [{"role": "user", "content": "질문"}]


def test_build_image_message_orders_text_then_image():
    msgs = build_image_message("설명해줘", "data:image/jpeg;base64,AAA")
    content = msgs[0]["content"]
    assert msgs[0]["role"] == "user"
    assert content[0] == {"type": "text", "text": "설명해줘"}
    assert content[1] == {"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64,AAA"}}
