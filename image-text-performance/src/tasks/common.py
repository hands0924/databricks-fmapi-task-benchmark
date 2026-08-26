"""태스크 플러그인 공용 유틸리티.

여러 태스크가 각자 복제해 온 패턴을 한곳에 모은다:
- 데이터셋 로드: registry 조회 → HF split 로드 (단일 언어 / 언어별 균등 분할)
- 컬럼명 자동 감지
- 이진 라벨 파싱 (yes/no, positive/negative, toxic/clean, 한국어 표현 포함)
- 채점 집계: 유효 예측 필터링, 이진 메트릭 요약, 언어별 통계
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry
from src.scoring.metrics import binary_metrics
from src.tasks.base import Sample


def mean(values: Sequence[float], default: float = 0.0) -> float:
    """평균. 빈 시퀀스면 default."""
    return sum(values) / len(values) if values else default


# ---------------------------------------------------------------- 데이터셋 로드


def load_dataset_rows(
    dataset_key: str,
    n: int,
    seed: int,
    *,
    registry: dict[str, Any] | None = None,
    default_split: str = "train",
) -> list[dict[str, Any]]:
    """registry 키로 HF split n개를 로드 (seed 고정).

    split이 없거나 "default"(일부 HF 데이터셋 호환성)면 default_split을 사용.
    """
    entry = resolve_dataset_entry(registry if registry is not None else load_registry(), dataset_key)
    split = entry.get("split") or default_split
    if split == "default":
        split = default_split
    return load_hf_split(entry["hf_id"], split, n, seed, entry.get("config"))


def single_language_rows(
    config: dict[str, Any],
    lang: str,
    n: int,
    seed: int,
    *,
    registry: dict[str, Any] | None = None,
    default_split: str = "train",
    dataset_hint: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """단일 언어 태스크용: config의 datasets.<lang> 데이터셋을 로드.

    Returns:
        (dataset_key, rows)
    """
    datasets_map = config.get("datasets") or {}
    if lang not in datasets_map:
        hint = f" ({dataset_hint})" if dataset_hint else ""
        raise ValueError(f"config에 datasets.{lang}{hint}가 없음")

    dataset_key = datasets_map[lang]
    rows = load_dataset_rows(
        dataset_key, n, seed, registry=registry, default_split=default_split
    )
    return dataset_key, rows


@dataclass
class LanguageBatch:
    """한 언어에 할당된 데이터셋 로드 결과."""

    lang: str
    dataset_key: str
    rows: list[dict[str, Any]]


def language_batches(
    config: dict[str, Any],
    n: int,
    seed: int,
    *,
    registry: dict[str, Any] | None = None,
    default_split: str = "train",
) -> Iterator[LanguageBatch]:
    """다국어 태스크용: n개를 언어별로 균등 분할해 데이터셋을 로드.

    나머지는 앞쪽 언어에 하나씩 배분한다 (n=5, 2개 언어 → 3, 2).
    """
    datasets_map = config.get("datasets")
    if not datasets_map:
        raise ValueError("config에 datasets 맵이 없음")

    registry = registry if registry is not None else load_registry()
    n_per_lang = max(1, n // len(datasets_map))
    remainder = n % len(datasets_map)

    for lang_idx, (lang, dataset_key) in enumerate(datasets_map.items()):
        n_lang = n_per_lang + (1 if lang_idx < remainder else 0)
        rows = load_dataset_rows(
            dataset_key, n_lang, seed, registry=registry, default_split=default_split
        )
        yield LanguageBatch(lang=lang, dataset_key=dataset_key, rows=rows)


def detect_column(
    rows: Sequence[dict[str, Any]],
    preferred: Sequence[str],
    *,
    exclude: Sequence[str] = (),
    what: str = "컬럼",
) -> str:
    """행 목록에서 컬럼명을 감지.

    preferred 순서로 먼저 찾고, 없으면 exclude에 없는 첫 컬럼을 사용한다.
    exclude가 비어 있으면 폴백 없이 예외를 발생시킨다.

    Args:
        rows: load_hf_split 결과 (list[dict])
        preferred: 우선순위 컬럼명
        exclude: 폴백 시 제외할 컬럼명
        what: 오류 메시지에 쓸 컬럼 설명 (예: "텍스트 컬럼")
    """
    column_names = list(rows[0].keys()) if rows else []

    for col in preferred:
        if col in column_names:
            return col

    if exclude:
        for col in column_names:
            if col not in exclude:
                return col

    raise ValueError(f"{what}을 찾을 수 없음. 컬럼: {column_names}")


# ------------------------------------------------------------------- 라벨 파싱


def parse_binary_label(
    raw_text: str,
    *,
    positive: Sequence[str],
    negative: Sequence[str],
    negative_first: bool = False,
    numeric_positive: str = r"\b1\b",
    numeric_negative: str = r"\b0\b",
) -> int | None:
    """모델 응답을 이진 라벨(0/1)로 파싱. 판정 불가 시 None.

    키워드는 소문자 변환 후 부분 일치로 검사한다(한국어 표현도 그대로 사용 가능).
    negative_first=True면 부정 표현을 먼저 검사한다("no weapon" → 0).
    키워드로 판정되지 않으면 numeric_* 정규식으로 재시도한다.
    """
    if not raw_text or not raw_text.strip():
        return None

    text_lower = raw_text.strip().lower()

    groups: tuple[tuple[Sequence[str], int], ...] = (
        ((negative, 0), (positive, 1)) if negative_first else ((positive, 1), (negative, 0))
    )
    for keywords, label in groups:
        if any(keyword.lower() in text_lower for keyword in keywords):
            return label

    if re.search(numeric_positive, text_lower):
        return 1
    if re.search(numeric_negative, text_lower):
        return 0

    return None


# -------------------------------------------------------------------- 채점 집계

MetricFn = Callable[[list[Any], list[Any]], dict[str, Any]]


def valid_indices_of(parsed: Sequence[Any]) -> list[int]:
    """파싱 성공(None이 아닌) 예측의 인덱스."""
    return [i for i, p in enumerate(parsed) if p is not None]


def preds_and_golds(
    parsed: Sequence[Any], samples: Sequence[Sample], indices: Sequence[int]
) -> tuple[list[Any], list[Any]]:
    """주어진 인덱스의 (예측, 정답) 쌍을 추출."""
    return [parsed[i] for i in indices], [samples[i].reference for i in indices]


def per_language_stats(
    parsed: Sequence[Any],
    samples: Sequence[Sample],
    *,
    metric_fn: MetricFn,
    metric_keys: Sequence[str],
    langs: Sequence[str] = ("en", "ko"),
) -> dict[str, Any]:
    """언어별 평가/미파싱 샘플 수와 메트릭 집계.

    해당 언어의 유효 예측이 없으면 메트릭 값은 None.
    """
    valid_indices = valid_indices_of(parsed)
    stats: dict[str, Any] = {}

    for lang in langs:
        lang_indices = [i for i in valid_indices if samples[i].lang == lang]
        entry: dict[str, Any] = {
            "n_evaluated": len(lang_indices),
            "n_unparsed": sum(
                1 for i, p in enumerate(parsed) if samples[i].lang == lang and p is None
            ),
        }

        if lang_indices:
            preds, golds = preds_and_golds(parsed, samples, lang_indices)
            metrics = metric_fn(preds, golds)
            entry.update({key: metrics[key] for key in metric_keys})
        else:
            entry.update({key: None for key in metric_keys})

        stats[lang] = entry

    return stats


def binary_score_summary(
    parsed: Sequence[int | None],
    samples: Sequence[Sample],
    *,
    class_balance_keys: tuple[str, str] | None = None,
    per_language: bool = False,
) -> dict[str, Any]:
    """이진 분류 채점 집계: accuracy/f1/혼동행렬 + 평가·미파싱 샘플 수.

    None 예측은 제외하고 n_unparsed로 집계한다.

    Args:
        class_balance_keys: 지정 시 정답의 (0, 1) 클래스 분포를 해당 키로 포함
        per_language: True면 언어별 accuracy/f1 통계를 per_language로 포함
    """
    valid_indices = valid_indices_of(parsed)
    preds, golds = preds_and_golds(parsed, samples, valid_indices)

    result: dict[str, Any] = {
        "accuracy": 0.0,
        "f1": 0.0,
        "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        "n_evaluated": len(valid_indices),
        "n_unparsed": len(parsed) - len(valid_indices),
    }

    if valid_indices:
        metrics = binary_metrics(preds, golds)
        result["accuracy"] = metrics["accuracy"]
        result["f1"] = metrics["f1"]
        result["confusion_matrix"] = metrics["confusion_matrix"]

    if class_balance_keys is not None:
        key_0, key_1 = class_balance_keys
        result["class_balance"] = {
            key_0: sum(1 for g in golds if g == 0),
            key_1: sum(1 for g in golds if g == 1),
        }

    if per_language:
        result["per_language"] = (
            per_language_stats(
                parsed,
                samples,
                metric_fn=binary_metrics,
                metric_keys=("accuracy", "f1"),
            )
            if valid_indices
            else {}
        )

    return result


def best_reference_score(
    pred: str,
    references: Any,
    scorer: Callable[[str, str], float],
) -> float:
    """여러 정답 중 최고 점수. references가 리스트가 아니면 단일 정답으로 취급."""
    if not isinstance(references, (list, tuple)):
        references = [references]

    best = 0.0
    for reference in references:
        best = max(best, scorer(pred, str(reference)))
    return best
