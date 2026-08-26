"""Benchmark CLI and grading error reporting tests."""

import json
import sys

from benchmark import grade_tasks, run_task


def test_validate_html_parse_error_for_missing_and_empty(tmp_path):
    cfg = {"slide_count": {"min": 1, "max": 2}, "keywords": {}}

    missing = grade_tasks.validate_html(tmp_path / "missing.html", cfg)
    assert missing["parse_error"] == "slides.html missing"

    empty_path = tmp_path / "slides.html"
    empty_path.write_text("")
    empty = grade_tasks.validate_html(empty_path, cfg)
    assert empty["parse_error"] == "slides.html is empty"


def test_discover_candidates_records_corrupt_metadata(tmp_path, monkeypatch, capsys):
    task_dir = tmp_path / "task"
    candidate_dir = task_dir / "candidate"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "slides.html").write_text("<html></html>")
    (candidate_dir / "run_meta.json").write_text("{not valid json")
    monkeypatch.setattr(grade_tasks.task_spec, "task_dir", lambda task: task_dir)

    candidates = grade_tasks.discover_candidates("task", None)

    assert candidates[0]["meta"]["meta_error"].startswith("JSONDecodeError:")
    assert "run_meta.json" in capsys.readouterr().err


def test_run_task_returns_failure_for_missing_artifact(tmp_path, monkeypatch):
    workdir = tmp_path / "candidate"
    workdir.mkdir()
    monkeypatch.setattr(run_task, "prepare_workdir", lambda task, candidate: workdir)
    monkeypatch.setattr(
        run_task,
        "run_subprocess",
        lambda *args, **kwargs: {
            "artifact_exists": False,
            "timed_out": False,
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run-task", "--task", "task", "--candidate", "candidate", "--harness", "codex"],
    )

    assert run_task.main() == 1
    assert json.loads((workdir / "run_meta.json").read_text())["artifact_exists"] is False
