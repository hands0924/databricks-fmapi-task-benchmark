"""오류 전파 및 진단 동작 테스트."""

import httpx
import pytest

from src.adapters import fmapi
from src.scoring import stats
from src.tasks import img_4, txt_3


def make_client(monkeypatch, transport):
    monkeypatch.setattr(fmapi, "_get_workspace_auth", lambda profile: ("https://example.com", "token"))
    client = fmapi.FMAPIClient("test", max_retries=3, backoff_initial_seconds=0)
    client._client.close()
    client._client = httpx.Client(transport=transport)
    monkeypatch.setattr(client, "_sleep_backoff", lambda attempt: None)
    return client


def test_fmapi_retries_request_error_then_succeeds(monkeypatch):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
            request=request,
        )

    client = make_client(monkeypatch, httpx.MockTransport(handler))
    try:
        response = client.chat("endpoint", [{"role": "user", "content": "hi"}])
    finally:
        client.close()

    assert response.text == "ok"
    assert attempts == 3


def test_fmapi_exhausted_retries_chain_request_error(monkeypatch):
    def handler(request):
        raise httpx.ReadError("read failed", request=request)

    client = make_client(monkeypatch, httpx.MockTransport(handler))
    try:
        with pytest.raises(fmapi.FMAPIError) as exc_info:
            client.chat("endpoint", [])
    finally:
        client.close()

    assert isinstance(exc_info.value.__cause__, httpx.ReadError)
    assert "read failed" in str(exc_info.value.__cause__)


def test_fmapi_non_json_success_is_fmapi_error(monkeypatch):
    def handler(request):
        return httpx.Response(200, text="not json", request=request)

    client = make_client(monkeypatch, httpx.MockTransport(handler))
    try:
        with pytest.raises(fmapi.FMAPIError, match="응답이 JSON이 아님"):
            client.chat("endpoint", [])
    finally:
        client.close()


def test_fmapi_client_error_does_not_retry(monkeypatch):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="bad request", request=request)

    client = make_client(monkeypatch, httpx.MockTransport(handler))
    try:
        with pytest.raises(fmapi.FMAPIError, match="HTTP 400"):
            client.chat("endpoint", [])
    finally:
        client.close()

    assert attempts == 1


def test_wilcoxon_failure_has_uniform_error_shape(monkeypatch, capsys):
    def fail(_diffs):
        raise ValueError("invalid data")

    monkeypatch.setattr(stats.stats, "wilcoxon", fail)
    result = stats.wilcoxon_test([1, 2], [0, 1])

    assert result == {
        "pval": None,
        "significant": False,
        "n": 2,
        "error": "ValueError: invalid data",
    }
    assert "ValueError: invalid data" in capsys.readouterr().err


def test_parse_html_table_rejects_non_string():
    with pytest.raises(TypeError):
        txt_3.parse_html_table(None)


def test_parse_html_table_parses_normal_table():
    html = "<table><tr><td>A</td><td><b>B</b></td></tr></table>"
    assert txt_3.parse_html_table(html) == [(0, 0, "A"), (0, 1, "B")]


def test_img4_requires_samples_from_both_sources(monkeypatch):
    monkeypatch.setattr(img_4, "load_registry", dict)

    def load_split(hf_id, split, n, seed, config=None):
        if "nsfw" in hf_id:
            return []
        return [{"image": object()}]

    monkeypatch.setattr(img_4, "load_hf_split", load_split)
    with pytest.raises(RuntimeError, match="requires both NSFW and SFW"):
        img_4.Img4Task({}, {}).load_samples(2, 42)
