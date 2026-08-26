"""태스크 플러그인 동적 로더.

src/tasks/ 아래 txt_*.py, img_*.py 모듈을 import하면 @register가 발동해
base._REGISTRY에 task_id → 클래스가 등록된다. 아직 미구현(파일 없음/에러)인 태스크는
방어적으로 스킵해, Phase별 점진 구현 중에도 runner가 동작하게 한다.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

from src.tasks import base


IMPORT_ERRORS: dict[str, str] = {}


def discover_tasks() -> dict[str, type[base.Task]]:
    """src.tasks 패키지의 모든 태스크 모듈을 import해 레지스트리를 채운다.

    import 실패(미구현·의존성 문제) 모듈은 경고만 남기고 스킵.
    반환: task_id → Task 클래스.
    """
    import src.tasks as tasks_pkg

    IMPORT_ERRORS.clear()
    skip = {"base", "loader"}
    for mod_info in pkgutil.iter_modules(tasks_pkg.__path__):
        name = mod_info.name
        if name in skip or name.startswith("_"):
            continue
        try:
            importlib.import_module(f"src.tasks.{name}")
        except Exception as e:  # noqa: BLE001 — skip modules with varied import failures
            error = f"{type(e).__name__}: {e}"
            IMPORT_ERRORS[name] = error
            print(f"  [태스크 로드 스킵] {name}: {error}", file=sys.stderr)
    return base.all_registered()
