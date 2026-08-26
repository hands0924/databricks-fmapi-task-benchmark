"""리포트 생성기 (plan §7). Executive Summary + 정량 + 정성 + 시간/비용.

- run별 markdown 리포트를 reports/<run_id>/report.md로 생성.
- Executive Summary: 하이브리드(D14) — 수치에서 fact sheet를 규칙 추출 → judge가 문단화.
  judge 실패 시 규칙 기반 템플릿 문장으로 fallback(결정론적).
- 시간·비용: SampleResult의 latency + usage를 pricing.yaml로 USD 환산(§10).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

from src.cost.pricing import compute_usd, load_pricing
from src.results import SampleResult


def generate_report(
    run_dir: Path,
    results: list[SampleResult],
    scores: dict[str, Any],
    models_cfg,
) -> Path:
    """run 디렉터리에 report.md를 생성하고 경로를 반환."""
    reports_dir = Path("reports") / run_dir.name
    reports_dir.mkdir(parents=True, exist_ok=True)

    # endpoint 매핑 (model_id → endpoint)
    ep = {m.id: m.endpoint for m in models_cfg.models}

    perf = _perf_by_model(results, ep)
    facts = _extract_facts(scores, perf)
    summary = _executive_summary(facts, models_cfg)

    md = []
    md.append(f"# 벤치마크 리포트 — {run_dir.name}\n")
    md.append("## Executive Summary\n")
    md.append(summary + "\n")

    md.append("## 정량 결과 (태스크 × 모델 × reasoning)\n")
    md.append(_quant_table(scores) + "\n")

    md.append("## 성능: 수행시간·비용 (모델별)\n")
    md.append(_perf_table(perf) + "\n")
    md.append(
        "> 비용은 `config/pricing.yaml` DBU 단가 기반 추정(usd_per_dbu 가정값). "
        "정밀 비용은 `system.ai_gateway.usage` 조인 필요(§10).\n"
    )

    md.append("## Fact Sheet (Executive Summary 근거 — 감사용)\n")
    md.append("```json\n" + json.dumps(facts, ensure_ascii=False, indent=2, default=str) + "\n```\n")

    report_path = reports_dir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    # fact sheet 별도 저장(문장-수치 대응 감사)
    (reports_dir / "facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report_path


def _perf_by_model(results: list[SampleResult], endpoints: dict[str, str]) -> dict[str, dict]:
    """모델별 latency(median/p95)·토큰·USD 비용 집계."""
    pricing = load_pricing()
    agg: dict[str, dict] = {}
    for r in results:
        a = agg.setdefault(
            r.model_id,
            {"latencies": [], "usd": 0.0, "in_tok": 0, "out_tok": 0, "n": 0, "errors": 0,
             "priced": 0, "unpriced": 0},
        )
        a["n"] += 1
        if r.finish_reason == "error":
            a["errors"] += 1
            continue
        a["latencies"].append(r.latency_ms_local)
        ep = endpoints.get(r.model_id, "")
        usd = compute_usd(ep, r.usage or {}, pricing)
        if usd is None:
            # unpriced endpoint: treating it as $0 would make the model look cheapest
            if a["unpriced"] == 0:
                print(
                    f"warning: endpoint '{ep}' missing from pricing table; "
                    f"cost for {r.model_id} is incomplete",
                    file=sys.stderr,
                )
            a["unpriced"] += 1
        else:
            a["priced"] += 1
            a["usd"] += usd
        a["in_tok"] += (r.usage or {}).get("prompt_tokens", 0) or 0
        a["out_tok"] += (r.usage or {}).get("completion_tokens", 0) or 0

    out = {}
    for mid, a in agg.items():
        lat = a["latencies"]
        out[mid] = {
            "n_calls": a["n"],
            "errors": a["errors"],
            "latency_ms_median": round(statistics.median(lat), 1) if lat else None,
            "latency_ms_p95": round(_p95(lat), 1) if lat else None,
            "total_usd": round(a["usd"], 6) if a["priced"] else None,
            "unpriced_calls": a["unpriced"],
            "usd_complete": a["unpriced"] == 0,
            "in_tokens": a["in_tok"],
            "out_tokens": a["out_tok"],
        }
    return out


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def _extract_facts(scores: dict[str, Any], perf: dict[str, dict]) -> dict[str, Any]:
    """수치에서 Executive Summary용 핵심 사실을 규칙 기반으로 추출(D14)."""
    # 태스크별 대표 점수(모델→점수) 뽑기 — 대표 메트릭 우선순위
    metric_priority = ["accuracy", "f1", "token_f1", "rouge1", "judge_mean", "judge_score_mean"]
    per_task: dict[str, dict[str, float]] = {}
    for _, entry in scores.items():
        m = entry["metrics"]
        if not isinstance(m, dict) or "error" in m:
            continue
        val = next((m[k] for k in metric_priority if isinstance(m.get(k), (int, float))), None)
        if val is None:
            continue
        key = f"{entry['task_id']}/{entry['reasoning_mode']}"
        per_task.setdefault(key, {})[entry["model_id"]] = round(float(val), 4)

    # 태스크별 1위 모델
    winners = {}
    for tk, mv in per_task.items():
        if mv:
            winners[tk] = max(mv, key=mv.get)

    # 모델별 1위 횟수
    win_counts: dict[str, int] = {}
    for w in winners.values():
        win_counts[w] = win_counts.get(w, 0) + 1

    # 비용·속도 최고/최저
    # models with unpriced calls are excluded: their total is an undercount
    priced = [k for k in perf if perf[k]["usd_complete"] and perf[k]["total_usd"] is not None]
    unpriced_models = sorted(k for k in perf if not perf[k]["usd_complete"])
    cheapest = min(priced, key=lambda k: perf[k]["total_usd"]) if priced else None
    fastest = min(
        (k for k in perf if perf[k]["latency_ms_median"] is not None),
        key=lambda k: perf[k]["latency_ms_median"],
        default=None,
    )
    return {
        "per_task_scores": per_task,
        "task_winners": winners,
        "win_counts": win_counts,
        "cheapest_model": cheapest,
        "unpriced_models": unpriced_models,
        "fastest_model": fastest,
        "perf": perf,
    }


def _executive_summary(facts: dict[str, Any], models_cfg) -> str:
    """하이브리드 요약(D14): judge로 문단화 시도, 실패 시 규칙 기반 fallback."""
    rule_based = _rule_based_summary(facts)
    # judge 문단화 시도 (실패해도 규칙 기반으로 진행)
    try:
        from src.adapters.fmapi import FMAPIClient, build_text_message

        prompt = (
            "다음은 LLM 벤치마크 집계 사실이다. 이 수치에만 근거해, 어느 모델이 무엇을 잘하고/못하는지, "
            "속도·비용 트레이드오프는 어떤지를 3~4문장 한국어 요약으로 써라. 수치를 지어내지 말 것.\n\n"
            + json.dumps(facts, ensure_ascii=False, default=str)[:3000]
        )
        # judge(gemini)는 reasoning을 완전히 못 끄는 모델 → 사고 토큰에 소진돼 요약이 잘리지
        # 않도록 max_tokens를 크게(3000) 잡는다. timeout도 넉넉히.
        with FMAPIClient(profile=models_cfg.profile, timeout_seconds=max(60, models_cfg.runtime.timeout_seconds)) as c:
            resp = c.chat(models_cfg.judge, build_text_message(prompt), max_tokens=3000)
        if resp.text.strip():
            facts["executive_summary_source"] = "judge"
            return resp.text.strip() + f"\n\n<sub>규칙 기반 요약(대조용): {rule_based}</sub>"
        reason = "judge returned an empty summary"
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
    print(
        f"warning: judge summary unavailable, falling back to rule-based summary: {reason}",
        file=sys.stderr,
    )
    facts["executive_summary_source"] = f"rule_based (judge failed: {reason})"
    return rule_based + f"\n\n<sub>judge summary unavailable ({reason}) — rule-based summary shown.</sub>"


def _rule_based_summary(facts: dict[str, Any]) -> str:
    """수치에서 직접 조립하는 결정론적 요약(fallback)."""
    wc = facts.get("win_counts", {})
    parts = []
    if wc:
        top = max(wc, key=wc.get)
        parts.append(f"태스크별 1위 횟수는 {', '.join(f'{k} {v}회' for k, v in sorted(wc.items(), key=lambda x: -x[1]))}로 **{top}**가 가장 많다.")
    if facts.get("fastest_model"):
        f = facts["fastest_model"]
        parts.append(f"응답 속도는 **{f}**가 가장 빠르다(median {facts['perf'][f]['latency_ms_median']}ms).")
    if facts.get("cheapest_model"):
        ch = facts["cheapest_model"]
        parts.append(f"비용은 **{ch}**가 가장 낮다(${facts['perf'][ch]['total_usd']}).")
    if facts.get("unpriced_models"):
        parts.append(
            f"단, {', '.join(facts['unpriced_models'])}는 pricing 표에 없는 endpoint 호출이 있어 "
            "비용 비교에서 제외했다."
        )
    return " ".join(parts) if parts else "집계할 결과가 없습니다."


def _quant_table(scores: dict[str, Any]) -> str:
    """태스크×모델×모드 점수 표(markdown). 대표 메트릭만 요약."""
    rows = ["| 태스크 | 모델 | reasoning | 대표 메트릭 |", "|---|---|---|---|"]
    for _, e in sorted(scores.items()):
        m = e["metrics"]
        if isinstance(m, dict) and "error" not in m:
            disp = {k: v for k, v in m.items() if isinstance(v, (int, float))}
            cell = ", ".join(f"{k}={round(v, 3)}" for k, v in list(disp.items())[:4]) or "—"
        else:
            cell = f"오류: {m.get('error', '?')}" if isinstance(m, dict) else "—"
        rows.append(f"| {e['task_id']} | {e['model_id']} | {e['reasoning_mode']} | {cell} |")
    return "\n".join(rows)


def _usd_cell(p: dict) -> str:
    """비용 셀: pricing 표에 없는 endpoint가 있으면 불완전함을 표시(0으로 오해 금지)."""
    if p["total_usd"] is None:
        return f"N/A (pricing 없음, {p['unpriced_calls']}건)"
    if not p["usd_complete"]:
        return f"{p['total_usd']} (불완전, pricing 없는 호출 {p['unpriced_calls']}건)"
    return str(p["total_usd"])


def _perf_table(perf: dict[str, dict]) -> str:
    rows = ["| 모델 | 호출 | 오류 | latency median(ms) | p95(ms) | 입력토큰 | 출력토큰 | 비용(USD) |",
            "|---|---|---|---|---|---|---|---|"]
    for mid, p in sorted(perf.items()):
        rows.append(
            f"| {mid} | {p['n_calls']} | {p['errors']} | {p['latency_ms_median']} | "
            f"{p['latency_ms_p95']} | {p['in_tokens']} | {p['out_tokens']} | {_usd_cell(p)} |"
        )
    return "\n".join(rows)
