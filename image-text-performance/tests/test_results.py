"""결과 저장·메니페스트 테스트 (src/results.py)."""

import json
import subprocess

import pytest
from src.results import (
    RunManifest,
    SampleResult,
    git_commit,
    make_run_id,
    write_manifest,
    write_sample_results,
)

from src import results as results_mod


def sample(sample_id=0, reference="정답"):
    return SampleResult(
        model_id="opus",
        task_id="TXT-4",
        sample_id=sample_id,
        reasoning_mode="minimal",
        prompt="질문",
        model_output="응답",
        reference=reference,
        request_id="req-1",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        latency_ms_local=123.4,
        timestamp="2026-07-31T14:00:00+00:00",
    )


def manifest(**overrides):
    kwargs = dict(
        run_id="2026-07-31T14-00",
        created_at="2026-07-31T14:00:00+00:00",
        models=[{"id": "opus", "endpoint": "databricks-claude-opus-5", "family": "claude"}],
        reasoning_modes=["minimal", "full"],
        task_ids=["TXT-4", "IMG-1"],
        git_commit="abc1234",
    )
    kwargs.update(overrides)
    return RunManifest(**kwargs)


# -------------------------------------------------------------- make_run_id ---
def test_make_run_id_format():
    run_id = make_run_id()
    assert len(run_id) == len("2026-07-31T14-00")
    assert run_id[4] == "-" and run_id[10] == "T" and run_id[13] == "-"


def test_make_run_id_with_version_suffix():
    assert make_run_id("v2").endswith("_v2")


# -------------------------------------------------------------- git_commit ---
def test_git_commit_returns_short_sha(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="deadbee\n"))
    assert git_commit() == "deadbee"


def test_git_commit_none_outside_repo(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 128, stdout=""))
    assert git_commit() is None


@pytest.mark.parametrize("exc", [FileNotFoundError(), subprocess.TimeoutExpired("git", 5)])
def test_git_commit_none_when_git_unusable(monkeypatch, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(subprocess, "run", boom)
    assert git_commit() is None


# ------------------------------------------------- write_sample_results ---
def test_write_sample_results_one_json_per_line(tmp_path):
    path = write_sample_results(tmp_path / "run", [sample(0), sample(1)])
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert path.name == "samples.jsonl"
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["model_id"] == "opus"
    assert first["usage"]["prompt_tokens"] == 10
    assert first["latency_ms_local"] == 123.4


def test_write_sample_results_creates_run_dir(tmp_path):
    path = write_sample_results(tmp_path / "nested" / "run", [sample()])
    assert path.exists()


def test_write_sample_results_serializes_sets_as_sorted_lists(tmp_path):
    path = write_sample_results(tmp_path / "run", [sample(reference={"b", "a"})])
    assert json.loads(path.read_text(encoding="utf-8"))["reference"] == ["a", "b"]


def test_write_sample_results_keeps_korean_readable(tmp_path):
    path = write_sample_results(tmp_path / "run", [sample()])
    assert "정답" in path.read_text(encoding="utf-8")


def test_write_sample_results_stringifies_unknown_types(tmp_path):
    class Weird:
        def __str__(self):
            return "weird-object"

    path = write_sample_results(tmp_path / "run", [sample(reference=Weird())])
    assert json.loads(path.read_text(encoding="utf-8"))["reference"] == "weird-object"


def test_write_sample_results_empty_list(tmp_path):
    path = write_sample_results(tmp_path / "run", [])
    assert path.read_text(encoding="utf-8") == ""


def test_json_default_handles_frozenset():
    assert results_mod._json_default(frozenset({"x", "y"})) == ["x", "y"]


# ------------------------------------------------------------ write_manifest ---
def test_write_manifest_round_trip(tmp_path):
    path = write_manifest(tmp_path / "run", manifest(samples_per_task=30, seed=42,
                                                    notes="첫 실행"))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "manifest.json"
    assert data["run_id"] == "2026-07-31T14-00"
    assert data["task_ids"] == ["TXT-4", "IMG-1"]
    assert data["samples_per_task"] == 30
    assert data["seed"] == 42
    assert data["notes"] == "첫 실행"


def test_manifest_optional_fields_default_empty():
    data = manifest().to_dict()
    assert data["datasets"] == {}
    assert data["pricing"] == {}
    assert data["samples_per_task"] is None
    assert data["seed"] is None
    assert data["notes"] == ""


def test_write_manifest_creates_run_dir(tmp_path):
    assert write_manifest(tmp_path / "nested" / "run", manifest()).exists()
