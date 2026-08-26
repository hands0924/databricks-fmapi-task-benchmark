"""리포트 생성기 테스트 (src/report/generate.py) — judge 호출 없이 규칙 기반 경로 검증."""

import json
from pathlib import Path

import pytest
from src.config import ModelsConfig
from src.report import generate as gen
from src.report.generate import (
    _executive_summary,
    _extract_facts,
    _p95,
    _perf_by_model,
    _perf_table,
    _quant_table,
    _rule_based_summary,
    generate_report,
)
from src.results import SampleResult

PRICING = {
    "usd_per_dbu": 0.07,
    "models": {
        "databricks-claude-opus-5": {"dbu_in": 100.0, "dbu_out": 500.0},
        "databricks-glm-5": {"dbu_in": 10.0, "dbu_out": 50.0},
    },
}

MODELS_CFG = ModelsConfig(
    profile="ai_devtools",
    judge="databricks-gemini-3-1-pro",
    models=[
        {"id": "opus", "endpoint": "databricks-claude-opus-5", "family": "claude"},
        {"id": "glm", "endpoint": "databricks-glm-5", "family": "openai"},
    ],
)

ENDPOINTS = {m.id: m.endpoint for m in MODELS_CFG.models}


@pytest.fixture(autouse=True)
def stub_pricing(monkeypatch):
    monkeypatch.setattr(gen, "load_pricing", lambda *a, **k: PRICING)


@pytest.fixture(autouse=True)
def no_judge_calls(monkeypatch):
    """judge 문단화는 네트워크가 필요하므로 기본적으로 실패시켜 규칙 기반 fallback을 쓴다."""
    import src.adapters.fmapi as fmapi_mod

    def boom(*a, **k):
        raise RuntimeError("네트워크 없음")

    monkeypatch.setattr(fmapi_mod, "FMAPIClient", boom)


def result(model_id="opus", *, latency=100.0, usage=None, finish_reason="stop",
           task_id="TXT-4", sample_id=0):
    return SampleResult(
        model_id=model_id, task_id=task_id, sample_id=sample_id,
        reasoning_mode="minimal", prompt="q", model_output="a", reference="a",
        request_id=f"req-{sample_id}", finish_reason=finish_reason,
        usage=usage if usage is not None else {"prompt_tokens": 100, "completion_tokens": 20},
        latency_ms_local=latency, timestamp="2026-07-31T14:00:00+00:00",
    )


def score_entry(task_id, model_id, mode, metrics):
    return {"task_id": task_id, "model_id": model_id, "reasoning_mode": mode,
            "metrics": metrics}


# ---------------------------------------------------------------------- _p95 ---
@pytest.mark.parametrize("xs,expected", [
    ([], 0.0),
    ([42.0], 42.0),
    ([1.0, 2.0], 2.0),
    (list(range(1, 101)), 95),
])
def test_p95(xs, expected):
    assert _p95(xs) == expected


# -------------------------------------------------------------- _perf_by_model ---
def test_perf_aggregates_latency_tokens_and_cost():
    perf = _perf_by_model([result(latency=100.0), result(latency=300.0, sample_id=1)],
                          ENDPOINTS)
    opus = perf["opus"]
    assert opus["n_calls"] == 2
    assert opus["errors"] == 0
    assert opus["latency_ms_median"] == 200.0
    assert opus["latency_ms_p95"] == 300.0
    assert opus["in_tokens"] == 200
    assert opus["out_tokens"] == 40
    assert opus["total_usd"] > 0


def test_perf_excludes_errors_from_latency_and_cost():
    perf = _perf_by_model([result(latency=100.0),
                           result(latency=9999.0, sample_id=1, finish_reason="error")],
                          ENDPOINTS)
    opus = perf["opus"]
    assert (opus["n_calls"], opus["errors"]) == (2, 1)
    assert opus["latency_ms_median"] == 100.0
    assert opus["in_tokens"] == 100


def test_perf_all_errors_leaves_latency_none():
    perf = _perf_by_model([result(finish_reason="error")], ENDPOINTS)
    assert perf["opus"]["latency_ms_median"] is None
    assert perf["opus"]["latency_ms_p95"] is None
    assert perf["opus"]["total_usd"] == 0.0


def test_perf_handles_unknown_endpoint_without_pricing():
    perf = _perf_by_model([result(model_id="mystery")], {})
    assert perf["mystery"]["total_usd"] == 0.0
    assert perf["mystery"]["n_calls"] == 1


def test_perf_tolerates_empty_usage():
    perf = _perf_by_model([result(usage={})], ENDPOINTS)
    assert perf["opus"]["in_tokens"] == 0
    assert perf["opus"]["out_tokens"] == 0


def test_perf_groups_by_model():
    perf = _perf_by_model([result("opus"), result("glm", sample_id=1)], ENDPOINTS)
    assert set(perf) == {"opus", "glm"}


# -------------------------------------------------------------- _extract_facts ---
def test_facts_pick_representative_metric_and_winner():
    scores = {
        "a": score_entry("TXT-4", "opus", "minimal", {"accuracy": 0.9, "f1": 0.1}),
        "b": score_entry("TXT-4", "glm", "minimal", {"accuracy": 0.5}),
    }
    facts = _extract_facts(scores, {})
    assert facts["per_task_scores"]["TXT-4/minimal"] == {"opus": 0.9, "glm": 0.5}
    assert facts["task_winners"]["TXT-4/minimal"] == "opus"
    assert facts["win_counts"] == {"opus": 1}


def test_facts_metric_priority_prefers_accuracy_over_f1():
    scores = {"a": score_entry("TXT-4", "opus", "full", {"f1": 0.2, "accuracy": 0.7})}
    assert _extract_facts(scores, {})["per_task_scores"]["TXT-4/full"]["opus"] == 0.7


def test_facts_skip_error_and_non_numeric_entries():
    scores = {
        "err": score_entry("TXT-4", "opus", "minimal", {"error": "timeout"}),
        "text": score_entry("TXT-5", "opus", "minimal", {"accuracy": "n/a"}),
        "none": score_entry("TXT-6", "opus", "minimal", "N/A"),
    }
    facts = _extract_facts(scores, {})
    assert facts["per_task_scores"] == {}
    assert facts["win_counts"] == {}


def test_facts_separate_reasoning_modes():
    scores = {
        "a": score_entry("TXT-4", "opus", "minimal", {"accuracy": 0.4}),
        "b": score_entry("TXT-4", "glm", "full", {"accuracy": 0.3}),
    }
    winners = _extract_facts(scores, {})["task_winners"]
    assert winners == {"TXT-4/minimal": "opus", "TXT-4/full": "glm"}


def test_facts_identify_cheapest_and_fastest():
    perf = {
        "opus": {"total_usd": 1.0, "latency_ms_median": 500.0},
        "glm": {"total_usd": 0.1, "latency_ms_median": 2000.0},
    }
    facts = _extract_facts({}, perf)
    assert facts["cheapest_model"] == "glm"
    assert facts["fastest_model"] == "opus"


def test_facts_fastest_ignores_models_without_latency():
    perf = {
        "broken": {"total_usd": 0.0, "latency_ms_median": None},
        "glm": {"total_usd": 0.5, "latency_ms_median": 900.0},
    }
    assert _extract_facts({}, perf)["fastest_model"] == "glm"


def test_facts_none_without_perf_data():
    facts = _extract_facts({}, {})
    assert facts["cheapest_model"] is None
    assert facts["fastest_model"] is None


# ------------------------------------------------------- _rule_based_summary ---
def test_rule_based_summary_mentions_winner_speed_and_cost():
    facts = {
        "win_counts": {"opus": 3, "glm": 1},
        "fastest_model": "opus",
        "cheapest_model": "glm",
        "perf": {"opus": {"latency_ms_median": 500.0, "total_usd": 1.0},
                 "glm": {"latency_ms_median": 2000.0, "total_usd": 0.1}},
    }
    summary = _rule_based_summary(facts)
    assert "**opus**가 가장 많다" in summary
    assert "opus 3회" in summary and "glm 1회" in summary
    assert "500.0ms" in summary
    assert "$0.1" in summary


def test_rule_based_summary_without_results():
    assert _rule_based_summary({}) == "집계할 결과가 없습니다."


# --------------------------------------------------------- _executive_summary ---
def test_executive_summary_falls_back_when_judge_unavailable():
    facts = _extract_facts({"a": score_entry("TXT-4", "opus", "minimal", {"accuracy": 1.0})}, {})
    assert _executive_summary(facts, MODELS_CFG) == _rule_based_summary(facts)


def test_executive_summary_uses_judge_text_and_keeps_rule_based(monkeypatch):
    import src.adapters.fmapi as fmapi_mod

    class FakeResp:
        text = "  opus가 전반적으로 우수하다.  "

    class FakeClient:
        def __init__(self, profile, timeout_seconds):
            self.profile = profile
            FakeClient.timeout_seconds = timeout_seconds

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def chat(self, endpoint, messages, max_tokens=1024):
            FakeClient.seen = {"endpoint": endpoint, "messages": messages,
                               "max_tokens": max_tokens}
            return FakeResp()

    monkeypatch.setattr(fmapi_mod, "FMAPIClient", FakeClient)

    facts = {"win_counts": {"opus": 1}}
    summary = _executive_summary(facts, MODELS_CFG)
    assert summary.startswith("opus가 전반적으로 우수하다.")
    assert "규칙 기반 요약(대조용)" in summary
    assert FakeClient.seen["endpoint"] == "databricks-gemini-3-1-pro"
    assert FakeClient.seen["max_tokens"] == 3000
    assert FakeClient.timeout_seconds >= 60


def test_executive_summary_falls_back_on_blank_judge_output(monkeypatch):
    import src.adapters.fmapi as fmapi_mod

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def chat(self, *a, **k):
            return type("R", (), {"text": "   "})()

    monkeypatch.setattr(fmapi_mod, "FMAPIClient", FakeClient)
    facts = {"win_counts": {"opus": 1}}
    assert _executive_summary(facts, MODELS_CFG) == _rule_based_summary(facts)


# ---------------------------------------------------------------- 표 렌더링 ---
def test_quant_table_lists_metrics_per_row():
    scores = {"a": score_entry("TXT-4", "opus", "minimal",
                               {"accuracy": 0.8571, "f1": 0.5, "note": "x"})}
    table = _quant_table(scores)
    assert "| TXT-4 | opus | minimal | accuracy=0.857, f1=0.5 |" in table
    assert table.startswith("| 태스크 | 모델 | reasoning | 대표 메트릭 |")


def test_quant_table_caps_metrics_at_four():
    metrics = {f"m{i}": float(i) for i in range(6)}
    cell = _quant_table({"a": score_entry("T", "opus", "minimal", metrics)}).splitlines()[-1]
    assert cell.count("=") == 4


def test_quant_table_shows_errors():
    table = _quant_table({"a": score_entry("TXT-4", "glm", "full", {"error": "timeout"})})
    assert "오류: timeout" in table


def test_quant_table_placeholder_without_numeric_metrics():
    assert "| — |" in _quant_table({"a": score_entry("T", "opus", "minimal", {"note": "x"})})
    assert "| — |" in _quant_table({"a": score_entry("T", "opus", "minimal", None)})


def test_perf_table_rows_sorted_by_model():
    perf = _perf_by_model([result("opus"), result("glm", sample_id=1)], ENDPOINTS)
    table = _perf_table(perf)
    assert table.splitlines()[2].startswith("| glm |")
    assert table.splitlines()[3].startswith("| opus |")


# ------------------------------------------------------------- generate_report ---
def test_generate_report_writes_markdown_and_facts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scores = {"a": score_entry("TXT-4", "opus", "minimal", {"accuracy": 0.9})}
    path = generate_report(Path("results/2026-07-31T14-00"), [result()], scores, MODELS_CFG)

    assert path == Path("reports/2026-07-31T14-00/report.md")
    body = path.read_text(encoding="utf-8")
    assert "# 벤치마크 리포트 — 2026-07-31T14-00" in body
    for heading in ("## Executive Summary", "## 정량 결과", "## 성능: 수행시간·비용",
                    "## Fact Sheet"):
        assert heading in body
    assert "| TXT-4 | opus | minimal | accuracy=0.9 |" in body
    assert "config/pricing.yaml" in body

    facts = json.loads((path.parent / "facts.json").read_text(encoding="utf-8"))
    assert facts["task_winners"] == {"TXT-4/minimal": "opus"}
    assert facts["perf"]["opus"]["n_calls"] == 1


def test_generate_report_embeds_fact_sheet_json_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = generate_report(Path("results/run-1"), [result()], {}, MODELS_CFG)
    body = path.read_text(encoding="utf-8")
    block = body.split("```json\n")[1].split("\n```")[0]
    assert json.loads(block)["perf"]["opus"]["n_calls"] == 1


def test_generate_report_handles_empty_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = generate_report(Path("results/empty"), [], {}, MODELS_CFG)
    assert "집계할 결과가 없습니다." in path.read_text(encoding="utf-8")
