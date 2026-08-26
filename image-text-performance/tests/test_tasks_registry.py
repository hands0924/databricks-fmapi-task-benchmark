"""태스크 베이스·레지스트리·동적 로더 테스트 (src/tasks/base.py, src/tasks/loader.py)."""

import pytest
from src.tasks import base
from src.tasks.base import Sample, Task, all_registered, get_task_class, register
from src.tasks.loader import discover_tasks


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    """레지스트리는 모듈 전역이므로 테스트마다 복사본으로 교체한다."""
    monkeypatch.setattr(base, "_REGISTRY", dict(base._REGISTRY))


# ------------------------------------------------------------------- Sample ---
def test_sample_defaults():
    s = Sample(sample_id=3, inputs={"text": "질문"}, reference="정답")
    assert s.lang == "en"
    assert s.meta == {}


def test_sample_meta_is_per_instance():
    a = Sample(sample_id=0, inputs={}, reference=None)
    b = Sample(sample_id=1, inputs={}, reference=None)
    a.meta["x"] = 1
    assert b.meta == {}


# --------------------------------------------------------------------- Task ---
def test_task_defaults():
    t = Task({"id": "X"}, {"registry": True})
    assert t.config == {"id": "X"}
    assert t.registry == {"registry": True}
    assert (Task.task_id, Task.kind) == ("", "")
    assert Task.is_vision is False
    assert Task.sensitive is False


@pytest.mark.parametrize("call", [
    lambda t: t.load_samples(3, 42),
    lambda t: t.build_prompt(Sample(sample_id=0, inputs={}, reference=None)),
    lambda t: t.parse_output("raw", Sample(sample_id=0, inputs={}, reference=None)),
    lambda t: t.score([], []),
])
def test_base_task_methods_are_abstract(call):
    with pytest.raises(NotImplementedError):
        call(Task({}, {}))


# ----------------------------------------------------------------- register ---
def test_register_indexes_by_task_id():
    @register
    class Demo(Task):
        task_id = "DEMO-1"
        kind = "qa"

    assert get_task_class("DEMO-1") is Demo
    assert all_registered()["DEMO-1"] is Demo


def test_register_returns_the_class_unchanged():
    class Demo(Task):
        task_id = "DEMO-2"

    assert register(Demo) is Demo


def test_register_rejects_missing_task_id():
    class Anonymous(Task):
        pass

    with pytest.raises(ValueError):
        register(Anonymous)


def test_get_task_class_unknown_is_none():
    assert get_task_class("NOPE-9") is None


def test_all_registered_returns_a_copy():
    snapshot = all_registered()
    snapshot["INJECTED"] = Task
    assert "INJECTED" not in all_registered()


# ------------------------------------------------------------ discover_tasks ---
def test_discover_tasks_registers_all_implemented_tasks():
    registry = discover_tasks()
    assert {"IMG-1", "TXT-1", "TXT-8"} <= set(registry)
    assert all(issubclass(cls, Task) for cls in registry.values())
    assert "base" not in registry and "loader" not in registry


def test_discover_tasks_task_ids_match_registry_keys():
    for task_id, cls in discover_tasks().items():
        assert cls.task_id == task_id


def test_discover_tasks_skips_broken_modules(monkeypatch, capsys):
    """미구현·의존성 오류 모듈은 경고만 남기고 스킵해야 runner가 계속 돈다."""
    import src.tasks.loader as loader_mod

    real_import_module = loader_mod.importlib.import_module

    def flaky(name, *args, **kwargs):
        if name.endswith(".txt_1"):
            raise ImportError("pretend rouge_score is missing")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(loader_mod.importlib, "import_module", flaky)
    monkeypatch.setattr(base, "_REGISTRY", {"KEPT-1": Task})

    registry = loader_mod.discover_tasks()
    assert "TXT-1" not in registry
    assert registry["KEPT-1"] is Task
    assert "태스크 로드 스킵" in capsys.readouterr().out
