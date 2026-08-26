"""벤치마크 결과 저장 및 메니페스트 관리.

SampleResult: 개별 샘플 실행 결과 (모델·태스크·샘플 별).
RunManifest: 벤치마크 실행 메타데이터 (구성·시점·코드 버전).

JSON 직렬화 → results/<run_id>/samples.jsonl + manifest.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SampleResult:
    """개별 샘플 실행 결과."""

    model_id: str                      # 모델 ID (opus, sol, glm 등)
    task_id: str                       # 태스크 ID (IMG-1, TXT-4 등)
    sample_id: int                     # 데이터셋 내 샘플 인덱스
    reasoning_mode: str                # minimal 또는 full
    prompt: str                        # 입력 프롬프트 (저장 위해 필요시 truncate 가능)
    model_output: str                  # 모델 응답 (평문)
    reference: Any                     # 정답 (형식은 태스크마다 다름: str, list, dict 등)
    request_id: str | None             # FMAPI request_id (시간·비용 조인 키)
    finish_reason: str | None          # stop, length, error 등
    usage: dict[str, Any]              # {prompt_tokens, completion_tokens, ...}
    latency_ms_local: float            # 클라이언트 측 벽시계 시간(ms)
    timestamp: str                     # ISO 8601 timestamp
    parse_error: str | None = None     # parse_output() failure ("Type: msg"), None if it parsed


def make_run_id(version_suffix: str = "") -> str:
    """타임스탬프 기반 run ID 생성.

    형식: YYYY-MM-DDTHH-MM[_version_suffix]
    예시: 2026-07-31T14-00 또는 2026-07-31T14-00_v1
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H-%M")
    if version_suffix:
        return f"{timestamp}_{version_suffix}"
    return timestamp


def git_commit() -> str | None:
    """현재 커밋 SHA를 얻는다 (short form).

    git이 없거나 repo가 아니면 None 반환 (경고를 stderr로 남긴다).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"warning: git commit unavailable: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    if result.returncode != 0:
        print(
            f"warning: git commit unavailable (exit {result.returncode}): "
            f"{result.stderr.strip()[:200]}",
            file=sys.stderr,
        )
        return None
    return result.stdout.strip()


@dataclass
class RunManifest:
    """벤치마크 실행 메타데이터 (재현성·감사용)."""

    run_id: str                        # 고유 run ID
    created_at: str                    # ISO 8601 타임스탬프
    models: list[dict[str, Any]]       # [{"id": "opus", "endpoint": "...", "family": "claude"}, ...]
    reasoning_modes: list[str]         # ["minimal", "full"]
    task_ids: list[str]                # ["IMG-1", "IMG-2", ..., "TXT-8"]
    git_commit: str | None             # 코드 버전 (short SHA)
    datasets: dict[str, Any] = field(default_factory=dict)   # 태스크별 데이터셋 id·split (재현성 §12)
    pricing: dict[str, Any] = field(default_factory=dict)    # usd_per_dbu 등 (비용 재현 §12)
    samples_per_task: int | None = None                       # 태스크당 샘플 수
    seed: int | None = None                                   # subset 고정 seed
    meta_errors: dict[str, str] = field(default_factory=dict)  # 스냅샷 수집 실패 사유 (재현성 감사)
    notes: str = ""                    # 추가 메모 (선택)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화용 dict로 변환."""
        return asdict(self)


def write_sample_results(run_dir: str | Path, results: list[SampleResult]) -> Path:
    """샘플 결과를 results/<run_id>/samples.jsonl에 저장.

    각 줄은 하나의 SampleResult (JSON).
    run_dir이 없으면 생성.

    Returns: samples.jsonl 파일 경로.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    samples_file = run_dir / "samples.jsonl"
    with open(samples_file, "w", encoding="utf-8") as f:
        for result in results:
            # set(태그셋·키프레이즈 reference 등)은 JSON 미지원 → list로 변환
            line = json.dumps(asdict(result), ensure_ascii=False, default=_json_default)
            f.write(line + "\n")

    return samples_file


def _json_default(o: Any) -> Any:
    """JSON 직렬화 fallback: set→정렬 list, 그 외는 str."""
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    return str(o)


def write_manifest(run_dir: str | Path, manifest: RunManifest) -> Path:
    """메니페스트를 results/<run_id>/manifest.json에 저장.

    Returns: manifest.json 파일 경로.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = run_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)

    return manifest_file
