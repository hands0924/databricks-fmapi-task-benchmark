"""벤치마크 실행 orchestrator (Phase 0 스캐폴딩).

실행 매트릭스 구성(모델×태스크×reasoning_mode), N/A 스킵 처리,
dry-run 지원, manifest 작성까지 구현.

실제 샘플 루프(dataset loading + task plugins)는 Phase 1에서 구현.

사용:
    python -m src.runner --config config/models.yaml --tasks config/tasks.yaml --dry-run
    python -m src.runner --config config/models.yaml --tasks config/tasks.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.adapters.fmapi import FMAPIClient
from src.config import load_models_config
from src.results import RunManifest, git_commit, make_run_id, write_manifest


def load_tasks_config(path: str | Path = "config/tasks.yaml") -> dict[str, Any]:
    """태스크 설정을 YAML에서 로드.

    구조: {image_tasks: [...], text_tasks: [...], defaults: {...}}
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_execution_matrix(
    models_cfg,
    tasks_cfg: dict[str, Any],
    models_filter: list[str] | None = None,
    reasoning_override: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """실행 매트릭스 구성: (모델 × 태스크 × reasoning_mode).

    이미지 태스크 중 vision 미지원 모델은 N/A로 마킹하고 스킵.

    Args:
        models_cfg: ModelsConfig 객체
        tasks_cfg: tasks.yaml dict
        models_filter: 포함할 모델 ID list (None이면 전부)
        reasoning_override: 사용할 reasoning_mode list (None이면 config에서)

    Returns:
        (실행 항목 list, N/A 스킵된 항목 수)
    """
    # 실제 모델·모드 결정
    models = models_cfg.models
    if models_filter:
        models = [m for m in models if m.id in models_filter]

    reasoning_modes = reasoning_override or models_cfg.reasoning_modes

    # 태스크 로드
    image_tasks = tasks_cfg.get("image_tasks", [])
    text_tasks = tasks_cfg.get("text_tasks", [])
    all_tasks = image_tasks + text_tasks

    # 매트릭스 구성
    matrix: list[dict[str, Any]] = []
    na_count = 0

    for model in models:
        for task in all_tasks:
            task_id = task["id"]
            is_image_task = task_id.startswith("IMG-")

            # N/A 검사: 이미지 태스크 + vision 미지원
            if is_image_task and not model.supports("vision"):
                na_count += len(reasoning_modes)  # 이 모델-태스크 조합의 모든 모드는 N/A
                continue

            for reasoning_mode in reasoning_modes:
                matrix.append(
                    {
                        "model_id": model.id,
                        "model_endpoint": model.endpoint,
                        "task_id": task_id,
                        "reasoning_mode": reasoning_mode,
                    }
                )

    return matrix, na_count


def main() -> int:
    """메인 runner 진입점."""
    parser = argparse.ArgumentParser(
        description="LLM 벤치마크 runner (Phase 0 스캐폴딩)",
    )
    parser.add_argument(
        "--config",
        default="config/models.yaml",
        help="모델·reasoning 설정 (default: config/models.yaml)",
    )
    parser.add_argument(
        "--tasks",
        default="config/tasks.yaml",
        help="태스크 설정 (default: config/tasks.yaml)",
    )
    parser.add_argument(
        "--models",
        help="포함할 모델 ID (쉼표 구분, e.g. 'opus,sol'). 생략 시 전부.",
    )
    parser.add_argument(
        "--reasoning-modes",
        help="사용할 reasoning mode (쉼표 구분, e.g. 'minimal,full'). 생략 시 config에서.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="샘플 수 제한 (빠른 테스트용, 생략 시 config 따름).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="매트릭스만 구성하고 FMAPI 호출 안함. manifest만 저장.",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="결과 디렉터리 루트 (default: results)",
    )

    args = parser.parse_args()

    # 설정 로드
    try:
        models_cfg = load_models_config(args.config)
        tasks_cfg = load_tasks_config(args.tasks)
    except FileNotFoundError as e:
        print(f"오류: 설정 파일을 찾을 수 없음: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"오류: 설정 파일 파싱 실패: {e}", file=sys.stderr)
        return 1

    # 필터 파싱
    models_filter = None
    if args.models:
        models_filter = [m.strip() for m in args.models.split(",")]

    reasoning_override = None
    if args.reasoning_modes:
        reasoning_override = [m.strip() for m in args.reasoning_modes.split(",")]

    # 매트릭스 구성
    matrix, na_count = build_execution_matrix(
        models_cfg,
        tasks_cfg,
        models_filter=models_filter,
        reasoning_override=reasoning_override,
    )

    # 통계 계산
    total_cells = len(matrix)
    if total_cells == 0:
        print("경고: 실행할 항목이 없습니다.", file=sys.stderr)
        return 1

    # 모델별 셀 수 계산
    model_counts = {}
    for item in matrix:
        model_id = item["model_id"]
        model_counts[model_id] = model_counts.get(model_id, 0) + 1

    # Run ID 및 디렉터리 생성
    run_id = make_run_id()
    run_dir = Path(args.out) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Manifest 작성 (dry-run/real 공통)
    all_tasks = tasks_cfg.get("image_tasks", []) + tasks_cfg.get("text_tasks", [])
    task_ids = [t["id"] for t in all_tasks]

    model_list = []
    for model in models_cfg.models:
        if models_filter is None or model.id in models_filter:
            model_list.append(
                {
                    "id": model.id,
                    "endpoint": model.endpoint,
                    "family": model.family,
                }
            )

    # 재현성 메타(§12): 데이터셋 id·split, pricing 버전, 샘플·seed 스냅샷
    defaults = tasks_cfg.get("defaults", {})
    datasets_snapshot, pricing_snapshot, meta_errors = _reproducibility_meta(all_tasks)

    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        models=model_list,
        reasoning_modes=reasoning_override or models_cfg.reasoning_modes,
        task_ids=task_ids,
        git_commit=git_commit(),
        datasets=datasets_snapshot,
        pricing=pricing_snapshot,
        samples_per_task=args.samples or defaults.get("samples", 50),
        seed=defaults.get("seed", 42),
        meta_errors=meta_errors,
        notes="dry-run" if args.dry_run else "full run",
    )

    manifest_path = write_manifest(run_dir, manifest)

    # 출력 (summary)
    print("=" * 70)
    print(f"실행 매트릭스 구성 완료")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"결과 디렉터리: {run_dir}")
    print()
    print(f"모델 수: {len(model_list)}")
    print(f"태스크 수: {len(task_ids)}")
    print(f"Reasoning 모드: {len(reasoning_override or models_cfg.reasoning_modes)}")
    print()
    print(f"총 실행 셀: {total_cells}")
    print(f"N/A 스킵(vision 미지원): {na_count}")
    print()
    print("모델별 셀 수:")
    for model_id, count in sorted(model_counts.items()):
        print(f"  {model_id}: {count}")
    print()
    print(f"Manifest 저장: {manifest_path}")
    print()

    if args.dry_run:
        print("DRY-RUN 모드: FMAPI 호출하지 않음.")
        print("=" * 70)
        return 0

    # 실제 실행 경로 (Phase 1)
    print("실제 샘플 루프 시작...")
    print("=" * 70)

    try:
        fmapi = FMAPIClient(
            profile=models_cfg.profile,
            timeout_seconds=models_cfg.runtime.timeout_seconds,
            max_retries=models_cfg.runtime.max_retries,
            backoff_initial_seconds=models_cfg.runtime.backoff_initial_seconds,
        )
    except Exception as e:
        print(f"오류: FMAPI 클라이언트 초기화 실패: {e}", file=sys.stderr)
        return 1

    try:
        rc = _run_samples(
            fmapi=fmapi,
            models_cfg=models_cfg,
            tasks_cfg=tasks_cfg,
            matrix=matrix,
            run_dir=run_dir,
            sample_cap=args.samples,
        )
    finally:
        fmapi.close()

    print("=" * 70)
    return rc


def _run_samples(
    fmapi: FMAPIClient,
    models_cfg,
    tasks_cfg: dict[str, Any],
    matrix: list[dict[str, Any]],
    run_dir: Path,
    sample_cap: int | None,
) -> int:
    """실행 매트릭스를 순회하며 각 (모델×태스크×reasoning×샘플)을 호출·채점·저장.

    - 태스크 플러그인을 동적 로드. 미구현 태스크는 스킵(점진 구현 허용).
    - 샘플은 태스크당 1회 로드해 (모델×모드)가 동일 subset을 공유(공정 비교·재현성).
    - 각 호출의 request_id 기록 → 나중에 ai_gateway.usage와 조인(비용·시간).

    결과·점수는 항상 저장하되, 결과를 신뢰할 수 없게 만드는 실패
    (태스크 import·샘플 로드·채점·리포트 실패, 전체 호출 실패)는
    errors.json에 남기고 exit code 1로 전파한다.
    """
    from datetime import datetime, timezone

    from src.datasets_loader import load_registry
    from src.results import SampleResult, write_sample_results
    from src.tasks.base import Sample
    from src.tasks.loader import IMPORT_ERRORS, discover_tasks

    registry = load_registry()
    task_classes = discover_tasks()
    print(f"로드된 태스크: {sorted(task_classes)}")

    errors: dict[str, Any] = {
        "task_import_errors": dict(IMPORT_ERRORS),
        "sample_load_errors": {},
        "parse_errors": 0,
        "api_errors": 0,
        "score_errors": {},
        "unregistered_tasks": [],
    }
    if IMPORT_ERRORS:
        print(f"경고: 태스크 import 실패: {IMPORT_ERRORS}", file=sys.stderr)

    defaults = tasks_cfg.get("defaults", {})
    n_samples = sample_cap or defaults.get("samples", 50)
    seed = defaults.get("seed", 42)

    # task_id → 설정 dict
    all_task_cfgs = {t["id"]: t for t in tasks_cfg.get("image_tasks", []) + tasks_cfg.get("text_tasks", [])}

    # 태스크별 (플러그인 인스턴스, 샘플) 캐시 — 모델·모드 간 재사용
    task_cache: dict[str, tuple[Any, list[Sample]]] = {}

    def _get_task(task_id: str):
        if task_id in task_cache:
            return task_cache[task_id]
        cls = task_classes.get(task_id)
        if cls is None:
            errors["unregistered_tasks"].append(task_id)
            print(f"경고: 등록된 플러그인이 없는 태스크: {task_id}", file=sys.stderr)
            task_cache[task_id] = (None, [])
            return task_cache[task_id]
        inst = cls(all_task_cfgs.get(task_id, {}), registry)
        try:
            samples = inst.load_samples(n_samples, seed)
        except Exception as e:
            errors["sample_load_errors"][task_id] = f"{type(e).__name__}: {e}"
            print(f"  [샘플 로드 실패] {task_id}: {type(e).__name__}: {e}", file=sys.stderr)
            samples = []
        task_cache[task_id] = (inst, samples)
        return task_cache[task_id]

    results: list[SampleResult] = []
    # (model_id, task_id, mode) → {"parsed": [...], "samples": [...], "outputs": [...]}
    groups: dict[tuple[str, str, str], dict[str, list]] = {}
    executed = skipped_tasks = 0

    for cell in matrix:
        task_id = cell["task_id"]
        model = models_cfg.get_model(cell["model_id"])
        mode = cell["reasoning_mode"]
        inst, samples = _get_task(task_id)
        if inst is None or not samples:
            skipped_tasks += 1
            continue

        params = model.reasoning_params(mode)
        gkey = (model.id, task_id, mode)
        groups.setdefault(gkey, {"parsed": [], "samples": [], "outputs": []})
        for s in samples:
            messages = inst.build_prompt(s)
            t0 = time.perf_counter()
            try:
                resp = fmapi.chat(
                    model.endpoint,
                    messages,
                    max_tokens=models_cfg.runtime.max_tokens,
                    extra_params=params,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                output_text = resp.text
                req_id, finish, usage = resp.request_id, resp.finish_reason, resp.usage
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                output_text = f"__ERROR__: {type(e).__name__}: {e}"
                req_id, finish, usage = None, "error", {}
                errors["api_errors"] += 1
                print(
                    f"  [호출 실패] {model.id}/{task_id}/{mode} 샘플 {s.sample_id}: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )

            # 채점용 파싱 (실패해도 실행은 계속하되 사유를 결과에 남긴다)
            parse_error = None
            try:
                parsed = inst.parse_output(output_text, s)
            except Exception as e:
                parsed = None
                parse_error = f"{type(e).__name__}: {e}"
                errors["parse_errors"] += 1
                print(
                    f"  [파싱 실패] {model.id}/{task_id}/{mode} 샘플 {s.sample_id}: {parse_error}",
                    file=sys.stderr,
                )
            groups[gkey]["parsed"].append(parsed)
            groups[gkey]["samples"].append(s)
            groups[gkey]["outputs"].append(output_text)

            results.append(
                SampleResult(
                    model_id=model.id,
                    task_id=task_id,
                    sample_id=s.sample_id,
                    reasoning_mode=mode,
                    prompt=_truncate(_prompt_text(messages), 2000),
                    model_output=output_text,
                    reference=s.reference,
                    request_id=req_id,
                    finish_reason=finish,
                    usage=usage,
                    latency_ms_local=latency_ms,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    parse_error=parse_error,
                )
            )
            executed += 1

    # 그룹별 채점 집계
    scores, score_errors = _score_groups(groups, task_cache)
    errors["score_errors"] = score_errors

    path = write_sample_results(run_dir, results)
    _write_json(run_dir / "scores.json", scores)
    print(f"\n실행 완료: {executed}개 샘플 호출, 태스크 미구현으로 스킵된 셀 {skipped_tasks}")
    print(f"결과 저장: {path}")

    # 리포트 생성 (Executive Summary + 정량 + 시간/비용)
    try:
        from src.report.generate import generate_report

        report_path = generate_report(run_dir, results, scores, models_cfg)
        print(f"리포트 생성: {report_path}")
    except Exception as e:
        errors["report_error"] = f"{type(e).__name__}: {e}"
        print(f"  [리포트 생성 실패] {errors['report_error']}", file=sys.stderr)

    # 전체 run 인덱스 재빌드 (§12 시점별 축적) — 보조 산출물이라 실패해도 run은 유효
    try:
        from src.report.index import rebuild_index

        idx = rebuild_index()
        print(f"인덱스 갱신: {idx}")
    except Exception as e:
        errors["index_error"] = f"{type(e).__name__}: {e}"
        print(f"  [인덱스 갱신 스킵] {errors['index_error']}", file=sys.stderr)

    fatal = _fatal_reasons(errors, executed)
    errors["executed"] = executed
    errors["fatal"] = fatal
    _write_json(run_dir / "errors.json", errors)

    if errors["api_errors"] or errors["parse_errors"]:
        print(
            f"호출 실패 {errors['api_errors']}건, 파싱 실패 {errors['parse_errors']}건 "
            f"(상세: {run_dir / 'errors.json'})",
            file=sys.stderr,
        )
    if fatal:
        for reason in fatal:
            print(f"실패: {reason}", file=sys.stderr)
        return 1
    return 0


def _fatal_reasons(errors: dict[str, Any], executed: int) -> list[str]:
    """결과를 신뢰할 수 없게 만드는 실패만 골라 사유 목록으로 돌려준다.

    인덱스 재빌드·개별 호출 실패는 치명적이지 않다(리포트에 건수로 드러남).
    """
    reasons = []
    if errors["task_import_errors"]:
        reasons.append(f"태스크 import 실패: {sorted(errors['task_import_errors'])}")
    if errors["sample_load_errors"]:
        reasons.append(f"샘플 로드 실패: {sorted(errors['sample_load_errors'])}")
    if errors["score_errors"]:
        reasons.append(f"채점 실패: {sorted(errors['score_errors'])}")
    if errors.get("report_error"):
        reasons.append(f"리포트 생성 실패: {errors['report_error']}")
    if executed == 0:
        reasons.append("실행된 샘플 호출이 없음")
    elif errors["api_errors"] == executed:
        reasons.append(f"모든 호출이 실패({executed}건)")
    return reasons


def _reproducibility_meta(
    all_tasks: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """manifest용 데이터셋·pricing 스냅샷 (§12 재현성).

    수집에 실패하면 빈 dict를 돌려주되, 실패 사유를 함께 반환해 manifest에 남긴다
    (스냅샷이 비어 있는 manifest를 완전한 것처럼 보이지 않게 한다).
    """
    errors: dict[str, str] = {}

    datasets_snapshot: dict[str, Any] = {}
    try:
        from src.datasets_loader import load_registry

        registry = load_registry()
        for t in all_tasks:
            for lang, key in (t.get("datasets") or {}).items():
                entry = registry.get(key, {})
                datasets_snapshot[key] = {
                    "hf_id": entry.get("hf_id"),
                    "split": entry.get("split"),
                    "config": entry.get("config"),
                }
    except Exception as e:
        errors["datasets"] = f"{type(e).__name__}: {e}"
        print(f"warning: dataset snapshot unavailable: {errors['datasets']}", file=sys.stderr)

    pricing_snapshot: dict[str, Any] = {}
    try:
        from src.cost.pricing import load_pricing

        p = load_pricing()
        pricing_snapshot = {"usd_per_dbu": p.get("usd_per_dbu"), "routing": p.get("routing")}
    except Exception as e:
        errors["pricing"] = f"{type(e).__name__}: {e}"
        print(f"warning: pricing snapshot unavailable: {errors['pricing']}", file=sys.stderr)

    return datasets_snapshot, pricing_snapshot, errors


def _score_groups(groups: dict, task_cache: dict) -> tuple[dict[str, Any], dict[str, str]]:
    """(model, task, mode) 그룹별로 태스크 score()를 호출해 집계.

    태스크마다 score() 반환 형식이 달라도 그대로 저장(리포트가 흡수).
    키는 "model_id::task_id::mode" 문자열.

    Returns: (집계 dict, 그룹키 → 채점 실패 사유)
    """
    out: dict[str, Any] = {}
    score_errors: dict[str, str] = {}
    for (model_id, task_id, mode), g in groups.items():
        inst = task_cache.get(task_id, (None, []))[0]
        if inst is None:
            continue
        key = f"{model_id}::{task_id}::{mode}"
        try:
            metrics = inst.score(g["parsed"], g["samples"])
        except Exception as e:
            metrics = {"error": f"{type(e).__name__}: {e}"}
            score_errors[key] = metrics["error"]
            print(f"  [채점 실패] {key}: {metrics['error']}", file=sys.stderr)
        out[key] = {
            "model_id": model_id,
            "task_id": task_id,
            "reasoning_mode": mode,
            "n": len(g["samples"]),
            "metrics": metrics,
        }
    return out, score_errors


def _write_json(path: Path, obj: Any) -> None:
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _prompt_text(messages: list[dict[str, Any]]) -> str:
    """저장·표시용으로 messages에서 텍스트만 추출."""
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


if __name__ == "__main__":
    sys.exit(main())
