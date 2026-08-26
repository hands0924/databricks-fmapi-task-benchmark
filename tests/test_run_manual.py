"""benchmark.run_task.run_manual — the human-as-keyboard-proxy path.

Invariant under test: once START is pressed, run_meta.json is always writable
(meta is always returned) and an aborted deck is quarantined so the grader can
never pick it up.
"""

import pytest

from benchmark import run_task, task_spec


@pytest.fixture
def workdir(bench_root, monkeypatch):
    monkeypatch.setattr(run_task.shutil, "which", lambda _name: None)  # no pbcopy
    return run_task.prepare_workdir("demo-task", "ui")


def fake_inputs(monkeypatch, responses):
    it = iter(responses)

    def fake_input(_prompt=""):
        try:
            return next(it)
        except StopIteration as e:  # pragma: no cover - defensive
            raise AssertionError("input() called more times than expected") from e

    monkeypatch.setattr("builtins.input", fake_input)


def test_run_manual_brackets_timing_and_detects_artifact(workdir, monkeypatch):
    fake_inputs(monkeypatch, ["", ""])
    (workdir / task_spec.ARTIFACT).write_text("<html></html>", encoding="utf-8")

    meta = run_task.run_manual("demo-task", "ui", "playground", "databricks-x", workdir)

    assert meta["mode"] == "manual"
    assert meta["aborted"] is False
    assert meta["artifact_exists"] is True
    assert meta["effective_model"] == "databricks-x"
    assert meta["wall_seconds"] >= 0
    assert meta["started_at"] and meta["finished_at"]
    assert "human-bracketed" in meta["note"]


def test_run_manual_warns_when_no_deck_produced(workdir, monkeypatch, capsys):
    fake_inputs(monkeypatch, ["", ""])
    meta = run_task.run_manual("demo-task", "ui", "playground", None, workdir)
    assert meta["artifact_exists"] is False
    assert "no slides.html produced" in capsys.readouterr().out


def test_run_manual_quarantines_aborted_deck(workdir, monkeypatch):
    fake_inputs(monkeypatch, ["", "q"])
    (workdir / task_spec.ARTIFACT).write_text("<html></html>", encoding="utf-8")

    meta = run_task.run_manual("demo-task", "ui", "playground", None, workdir)

    assert meta["aborted"] is True
    assert meta["artifact_exists"] is False
    assert not (workdir / task_spec.ARTIFACT).exists()
    assert (workdir / "slides.aborted.html").exists()
    assert "ABORTED" in meta["note"]
    assert "quarantined" in meta["note"]


def test_run_manual_treats_interrupt_as_abort(workdir, monkeypatch):
    def fake_input(_prompt=""):
        if fake_input.calls:
            raise KeyboardInterrupt
        fake_input.calls = True
        return ""

    fake_input.calls = False
    monkeypatch.setattr("builtins.input", fake_input)

    meta = run_task.run_manual("demo-task", "ui", "playground", None, workdir)
    assert meta["aborted"] is True


def test_run_manual_notes_failed_quarantine(workdir, monkeypatch):
    fake_inputs(monkeypatch, ["", "q"])
    (workdir / task_spec.ARTIFACT).write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(run_task.Path, "replace",
                        lambda self, target: (_ for _ in ()).throw(OSError("locked")))

    meta = run_task.run_manual("demo-task", "ui", "playground", None, workdir)
    assert "failed to quarantine" in meta["note"]
    assert meta["artifact_exists"] is True


def test_run_manual_copies_prompt_to_clipboard_when_pbcopy_exists(workdir, monkeypatch):
    fake_inputs(monkeypatch, ["", ""])
    monkeypatch.setattr(run_task.shutil, "which", lambda name: "/usr/bin/pbcopy")
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["input"] = kwargs.get("input")
        return None

    monkeypatch.setattr(run_task.subprocess, "run", fake_run)
    run_task.run_manual("demo-task", "ui", "playground", None, workdir)
    assert seen["argv"] == ["pbcopy"]
    assert seen["input"] == task_spec.COMMON_PROMPT


def test_run_manual_survives_clipboard_failure(workdir, monkeypatch, capsys):
    fake_inputs(monkeypatch, ["", ""])
    monkeypatch.setattr(run_task.shutil, "which", lambda name: "/usr/bin/pbcopy")
    monkeypatch.setattr(run_task.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no clipboard")))

    meta = run_task.run_manual("demo-task", "ui", "playground", None, workdir)
    assert meta["aborted"] is False
    assert "copy it manually" in capsys.readouterr().out
