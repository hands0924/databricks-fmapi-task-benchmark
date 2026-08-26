"""데이터셋 로더 (plan §5). registry.yaml 기반, seed 고정 subset, 로컬 캐시.

- HuggingFace `datasets`로 로드. 다운로드는 .cache/(gitignore)에만.
- seed 고정 subset으로 "표준·불변" 재현성 확보(D8).
- 민감 데이터(NSFW 등 sensitive)는 캐시 전용 — 원본 미디어를 repo에 남기지 않음(D3).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

# 캐시 루트 (gitignore됨)
CACHE_DIR = Path(".cache")


def load_registry(path: str | Path = "datasets/registry.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    if not isinstance(registry, dict):
        raise ValueError(f"registry 파일이 dict가 아님: {path}")  # noqa: TRY004
    return registry


def _hf_cache_dir() -> str:
    d = CACHE_DIR / "hf"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def load_hf_split(
    hf_id: str,
    split: str,
    n: int,
    seed: int,
    config: str | None = None,
):
    """HF 데이터셋에서 seed 고정 subset n개를 로드.

    **streaming 우선**: 대형 데이터셋(COCO 19GB, PubTabNet 16GB 등)을 전체 다운로드하면
    디스크가 폭발하므로, streaming으로 필요한 만큼만 받는다(50샘플에 수십 GB 방지).
    streaming이 불가한 경우(로컬 parquet 등) 일반 로드로 폴백.

    재현성: streaming은 buffer shuffle(seed 고정) 후 take(n). buffer를 넉넉히 잡아
    앞쪽 편향을 줄인다. 반환: list[dict] (streaming) 또는 datasets.Dataset(폴백).
    """
    from datasets import load_dataset

    # 1) streaming 시도 (다운로드 최소화)
    streaming_err: Exception | None = None
    try:
        ds = load_dataset(hf_id, name=config, split=split, streaming=True)
        # buffer shuffle로 앞쪽 편향 완화(전체 셔플은 streaming서 불가). buffer는 n의 배수.
        buffer = max(1000, n * 10)
        ds = ds.shuffle(seed=seed, buffer_size=buffer)
        rows = list(ds.take(n)) if n else list(ds)
        if rows:
            return rows
        print(
            f"[데이터셋] streaming 로드 결과가 비어 있어 일반 로드로 폴백: "
            f"{hf_id} ({split})",
            file=sys.stderr,
        )
    except Exception as e:
        streaming_err = e
        print(
            f"[데이터셋] streaming 로드 실패, 일반 로드로 폴백: "
            f"{hf_id} ({split}) ({type(e).__name__}: {e})",
            file=sys.stderr,
        )

    # 2) 폴백: 일반 로드 (작은 데이터셋·mirror parquet). 캐시는 .cache/hf.
    try:
        ds = load_dataset(hf_id, name=config, split=split, cache_dir=_hf_cache_dir())
    except Exception as e:  # noqa: BLE001
        streaming_detail = (
            f"{type(streaming_err).__name__}: {streaming_err}"
            if streaming_err
            else "streaming returned zero rows"
        )
        raise RuntimeError(
            f"데이터셋 로드 실패 (streaming/일반 모두): {hf_id} "
            f"streaming 오류: {streaming_detail}; "
            f"일반 로드 오류: {type(e).__name__}: {e}"
        ) from e
    if n and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    # list[dict]로 정규화(streaming 경로와 반환 형식 통일 → 태스크 코드 단순화)
    return [dict(row) for row in ds]


def get_label_names(hf_id: str, split: str, config: str | None, column: str) -> list[str] | None:
    """분류 데이터셋의 라벨 이름 목록을 얻는다(예: COCO category names).

    streaming은 features를 안 주므로, 라벨 이름이 필요하면 이 함수로 비-streaming
    메타만 짧게 조회한다(데이터 다운로드 없이 features만). 실패 시 None.
    중첩 컬럼(예: 'objects.category')도 지원.
    """
    from datasets import load_dataset_builder

    try:
        builder = load_dataset_builder(hf_id, name=config, cache_dir=_hf_cache_dir())
        feats = builder.info.features
        if feats is None:
            return None
        # 중첩 경로 탐색
        cur: Any = feats
        for part in column.split("."):
            if hasattr(cur, "feature"):  # Sequence
                cur = cur.feature
            cur = cur[part]
        # ClassLabel or Sequence[ClassLabel]
        if hasattr(cur, "feature"):
            cur = cur.feature
        return list(getattr(cur, "names", []) or []) or None
    except Exception as e:  # noqa: BLE001 — datasets backends raise varied exceptions
        print(
            f"[데이터셋] 라벨 이름 조회 실패: {hf_id} ({column}) "
            f"({type(e).__name__}: {e})",
            file=sys.stderr,
        )
        return None


def resolve_dataset_entry(registry: dict[str, Any], key: str) -> dict[str, Any]:
    """registry에서 데이터셋 항목을 꺼낸다. 없으면 KeyError."""
    if key not in registry:
        raise KeyError(f"데이터셋 '{key}'가 registry.yaml에 없음")
    return registry[key]


def is_sensitive(registry: dict[str, Any], key: str) -> bool:
    """민감 데이터셋 여부(NSFW 등). True면 미디어 repo 미저장·갤러리 숨김(D3)."""
    entry = registry.get(key, {})
    return bool(entry.get("sensitive", False))


# 오프라인/네트워크 차단 환경 대비 플래그
def offline_mode() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("ITP_OFFLINE") == "1"
