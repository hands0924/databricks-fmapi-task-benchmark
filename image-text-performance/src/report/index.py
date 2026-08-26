"""reports/index.md 생성 — 전체 run을 시간순으로 인덱싱 (plan §12).

모델을 추가하거나 재실행할 때마다 새 run-id 리포트가 쌓이므로, 전체를 한눈에
볼 인덱스가 필요하다. reports/ 디렉터리를 스캔해 매번 재생성한다(append 아님 →
중복·순서 문제 없음). 각 run의 report 옆 manifest(있으면)에서 요약을 읽는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def rebuild_index(reports_root: str | Path = "reports", results_root: str | Path = "results") -> Path:
    """reports/ 하위 run들을 스캔해 index.md를 재생성. 경로 반환."""
    reports_root = Path(reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)

    runs = []
    for d in sorted(reports_root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        report_md = d / "report.md"
        if not report_md.exists():
            continue
        # manifest는 results/<run-id>/에 있음 (있으면 요약 사용)
        manifest = _load_manifest(Path(results_root) / d.name)
        runs.append((d.name, manifest))

    lines = ["# 벤치마크 리포트 인덱스\n", f"총 {len(runs)}개 run (최신순).\n"]
    lines.append("| run-id | 일시 | 모델 | reasoning | 태스크수 | 샘플/태스크 | git |")
    lines.append("|---|---|---|---|---|---|---|")
    for run_id, m in runs:
        models = ",".join(x.get("id", "?") for x in m.get("models", [])) or "?"
        modes = ",".join(m.get("reasoning_modes", [])) or "?"
        ntask = len(m.get("task_ids", [])) or "?"
        spt = m.get("samples_per_task", "?")
        git = m.get("git_commit", "?") or "?"
        created = m.get("created_at", "")[:19]
        lines.append(f"| [{run_id}]({run_id}/report.md) | {created} | {models} | {modes} | {ntask} | {spt} | {git} |")

    index_path = reports_root / "index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def _load_manifest(results_dir: Path) -> dict:
    p = results_dir / "manifest.json"
    if p.exists():
        try:
            with p.open(encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(
                f"[리포트] manifest 읽기 실패: {p} "
                f"({type(e).__name__}: {e})",
                file=sys.stderr,
            )
            return {}
    return {}
