"""benchmark.task_spec — root/task resolution and task config loading."""

import json

import pytest

from benchmark import task_spec

from .conftest import DESCRIPTION


def test_benchmark_root_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("BENCHMARK_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert task_spec.benchmark_root() == tmp_path.resolve()


def test_benchmark_root_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv("BENCHMARK_ROOT", str(other))
    assert task_spec.benchmark_root() == other.resolve()


def test_task_dir_is_root_plus_task_name(bench_root):
    assert task_spec.task_dir("demo-task") == bench_root.resolve() / "demo-task"


def test_task_dir_defaults_to_default_task(bench_root):
    assert task_spec.task_dir().name == task_spec.DEFAULT_TASK


def test_load_task_reads_keywords_json(bench_root):
    cfg = task_spec.load_task("demo-task")
    assert cfg["task_id"] == "demo-task"
    assert cfg["slide_count"] == {"min": 2, "max": 3}
    assert set(cfg["keywords"]) == {"lakehouse", "delta", "spark"}


def test_load_task_missing_raises_with_root_in_message(bench_root):
    with pytest.raises(FileNotFoundError) as e:
        task_spec.load_task("nope")
    assert "keywords.json" in str(e.value)
    assert str(bench_root.resolve()) in str(e.value)


def test_load_description_reads_verbatim(bench_root):
    assert task_spec.load_description("demo-task") == DESCRIPTION


def test_load_description_missing_raises(bench_root):
    with pytest.raises(FileNotFoundError) as e:
        task_spec.load_description("nope")
    assert "TASK_DESCRIPTION.md" in str(e.value)


def test_load_task_propagates_invalid_json(bench_root):
    (bench_root / "demo-task" / "keywords.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        task_spec.load_task("demo-task")


def test_candidate_model_defaults_are_fmapi_names():
    """A bare claude-*/gpt-* name routes to the vendor backend, not FMAPI."""
    assert task_spec.CANDIDATE_MODELS
    assert all(m.startswith("databricks-") for m in task_spec.CANDIDATE_MODELS.values())


def test_pi_candidate_models_are_provider_model_pairs():
    for provider, model in task_spec.PI_CANDIDATE_MODELS.values():
        assert provider.startswith("databricks-")
        assert model.startswith("system.ai.")


def test_common_prompt_contract():
    assert "instructions.txt" in task_spec.COMMON_PROMPT
    assert task_spec.ARTIFACT in task_spec.COMMON_PROMPT
    assert "<!DOCTYPE html>" in task_spec.COMMON_PROMPT
