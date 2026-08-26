"""오류 전파 및 진단 동작 테스트."""

import httpx
import pytest

from src.adapters import fmapi
from src import runner
from src.report import generate
from src.scoring import stats
from src.tasks import img_4, txt_1, txt_3
from src.tasks.base import Sample


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_load_errors", {"TXT-1": "load failed"}),
        ("score_errors", {"TXT-1": "score failed"}),
        ("report_error", "report failed"),
    ],
)
def test_runner_fatal_reasons_include_run_failures(field, value):
    errors = {
        "task_import_errors": {},
        "sample_load_errors": {},
        "score_errors": {},
        "report_error": None,
        "api_errors": 0,
    }
    errors[field] = value

    assert runner._fatal_reasons(errors, executed=1)


def test_runner_fatal_reasons_include_zero_executions():
    errors = {
        "task_import_errors": {},
        "sample_load_errors": {},
        "score_errors": {},
        "report_error": None,
        "api_errors": 0,
    }

    assert runner._fatal_reasons(errors, executed=0)


def test_runner_fatal_reasons_are_empty_for_clean_run():
    errors = {
        "task_import_errors": {},
        "sample_load_errors": {},
        "score_errors": {},
        "report_error": None,
        "api_errors": 0,
    }

    assert runner._fatal_reasons(errors, executed=1) == []


def test_txt1_judge_scores_exclude_failed_and_unparsed_results(monkeypatch):
    monkeypatch.setattr(txt_1, "load_rubrics", lambda _path: {})

    class FakeResponse:
        def __init__(self, text):
            self.text = text

    class FakeJudge:
        def __init__(self):
            self.calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return FakeResponse("Score: 4")
            if self.calls == 2:
                raise RuntimeError("judge unavailable")
            return FakeResponse("no score here")

    samples = [
        Sample(sample_id=i, inputs={"question": f"question {i}"}, reference=["answer"])
        for i in range(3)
    ]
    result = txt_1.Txt1Task({}, {}).judge_scores(
        ["prediction 1", "prediction 2", "prediction 3"],
        samples,
        FakeJudge(),
    )

    assert result["judge_scores"] == [4, None, None]
    assert result["judge_mean"] == 4
    assert result["n_judged"] == 1
    assert result["n_unparsed"] == 1
    assert len(result["judge_errors"]) == 1
    assert 3 not in result["judge_scores"]


def test_txt1_judge_mean_is_none_when_no_scores_are_valid(monkeypatch):
    monkeypatch.setattr(txt_1, "load_rubrics", lambda _path: {})

    class FakeResponse:
        text = "not a numeric score"

    class FakeJudge:
        def chat(self, **_kwargs):
            return FakeResponse()

    sample = Sample(sample_id=1, inputs={"question": "question"}, reference=["answer"])
    result = txt_1.Txt1Task({}, {}).judge_scores(["prediction"], [sample], FakeJudge())

    assert result["judge_scores"] == [None]
    assert result["judge_mean"] is None
    assert result["n_unparsed"] == 1
    assert result["judge_errors"] == []
    assert 3 not in result["judge_scores"]


def test_extract_facts_excludes_unpriced_model_from_cheapest():
    perf = {
        "priced-model": {
            "usd_complete": True,
            "total_usd": 1.0,
            "latency_ms_median": 10.0,
        },
        "unpriced-model": {
            "usd_complete": False,
            "total_usd": 0.0,
            "latency_ms_median": 5.0,
        },
    }

    facts = generate._extract_facts({}, perf)

    assert facts["cheapest_model"] == "priced-model"
    assert facts["unpriced_models"] == ["unpriced-model"]
