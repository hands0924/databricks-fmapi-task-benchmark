"""Shared fixtures for the benchmark package tests.

Every helper in `benchmark` resolves task directories from the benchmark root
(``BENCHMARK_ROOT`` env var, cwd otherwise), so tests build a throwaway root in
tmp_path instead of touching the repo's real task dirs.
"""

import json

import pytest

KEYWORDS = {
    "task_id": "demo-task",
    "slide_count": {"min": 2, "max": 3},
    "keywords": {
        "lakehouse": ["lakehouse"],
        "delta": ["delta lake", "delta"],
        "spark": ["spark"],
    },
}

DESCRIPTION = "# demo-task\n\nBuild a deck.\n"


@pytest.fixture
def bench_root(tmp_path, monkeypatch):
    """A benchmark root containing one task dir with keywords + description."""
    root = tmp_path / "root"
    task = root / "demo-task"
    task.mkdir(parents=True)
    (task / "keywords.json").write_text(json.dumps(KEYWORDS), encoding="utf-8")
    (task / "TASK_DESCRIPTION.md").write_text(DESCRIPTION, encoding="utf-8")
    monkeypatch.setenv("BENCHMARK_ROOT", str(root))
    return root


def deck(n_slides: int = 2, *, text: str = "lakehouse delta lake spark",
         doctype: bool = True, external: bool = False) -> str:
    """A minimal slide deck with `n_slides` .slide divs."""
    slides = "".join(f'<div class="slide"><p>{text}</p></div>' for _ in range(n_slides))
    head = '<link rel="stylesheet" href="https://cdn.example/x.css">' if external else ""
    return (("<!DOCTYPE html>" if doctype else "")
            + f"<html><head>{head}</head><body>{slides}</body></html>")
