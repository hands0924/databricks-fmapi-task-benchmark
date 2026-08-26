"""데이터셋 로더 테스트 (src/datasets_loader.py) — HF 다운로드 없이 스텁으로 검증."""

import sys
import types

import pytest
import yaml

from src import datasets_loader as dl

REGISTRY = {
    "coco_captions": {"hf_id": "HuggingFaceM4/COCO", "split": "validation"},
    "nsfw_set": {"hf_id": "private/nsfw", "split": "train", "sensitive": True},
}


class FakeStreamingDataset:
    """datasets.IterableDataset 대역: shuffle/take 호출을 기록한다."""

    def __init__(self, rows):
        self.rows = rows
        self.shuffle_calls = []

    def shuffle(self, seed, buffer_size):
        self.shuffle_calls.append((seed, buffer_size))
        return self

    def take(self, n):
        return self.rows[:n]

    def __iter__(self):
        return iter(self.rows)


class FakeMapDataset:
    """비-streaming Dataset 대역."""

    def __init__(self, rows):
        self.rows = rows
        self.shuffled_with = None
        self.selected = None

    def __len__(self):
        return len(self.rows)

    def shuffle(self, seed):
        self.shuffled_with = seed
        return self

    def select(self, indices):
        self.selected = list(indices)
        return FakeMapDataset([self.rows[i] for i in self.selected])

    def __iter__(self):
        return iter(self.rows)


@pytest.fixture
def fake_datasets(monkeypatch):
    """`datasets` 모듈을 스텁으로 주입 (함수 내부 import이므로 sys.modules 교체)."""
    module = types.ModuleType("datasets")
    module.load_dataset = lambda *a, **k: pytest.fail("load_dataset가 설정되지 않음")
    module.load_dataset_builder = lambda *a, **k: pytest.fail("builder가 설정되지 않음")
    monkeypatch.setitem(sys.modules, "datasets", module)
    return module


# ----------------------------------------------------------- load_registry ---
def test_load_registry_parses_yaml(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(REGISTRY), encoding="utf-8")
    assert dl.load_registry(path)["coco_captions"]["hf_id"] == "HuggingFaceM4/COCO"


def test_load_registry_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dl.load_registry(tmp_path / "nope.yaml")


# ----------------------------------------------------- resolve_dataset_entry ---
def test_resolve_dataset_entry_returns_entry():
    assert dl.resolve_dataset_entry(REGISTRY, "coco_captions")["split"] == "validation"


def test_resolve_dataset_entry_unknown_key_raises():
    with pytest.raises(KeyError, match="registry.yaml에 없음"):
        dl.resolve_dataset_entry(REGISTRY, "ghost")


# --------------------------------------------------------------- sensitive ---
@pytest.mark.parametrize("key,expected", [
    ("nsfw_set", True),
    ("coco_captions", False),
    ("ghost", False),
])
def test_is_sensitive(key, expected):
    assert dl.is_sensitive(REGISTRY, key) is expected


# ------------------------------------------------------------ offline_mode ---
@pytest.mark.parametrize("env", ["HF_HUB_OFFLINE", "ITP_OFFLINE"])
def test_offline_mode_flags(monkeypatch, env):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("ITP_OFFLINE", raising=False)
    monkeypatch.setenv(env, "1")
    assert dl.offline_mode() is True


def test_offline_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("ITP_OFFLINE", raising=False)
    assert dl.offline_mode() is False


def test_offline_mode_ignores_other_values(monkeypatch):
    monkeypatch.delenv("ITP_OFFLINE", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert dl.offline_mode() is False


# ------------------------------------------------------------ _hf_cache_dir ---
def test_hf_cache_dir_created_under_dot_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / ".cache")
    assert dl._hf_cache_dir() == str(tmp_path / ".cache" / "hf")
    assert (tmp_path / ".cache" / "hf").is_dir()


# ------------------------------------------------------------ load_hf_split ---
def test_streaming_path_shuffles_with_seed_and_takes_n(fake_datasets):
    stream = FakeStreamingDataset([{"i": i} for i in range(100)])
    calls = []

    def load_dataset(hf_id, name=None, split=None, streaming=False, **kw):
        calls.append({"hf_id": hf_id, "name": name, "split": split, "streaming": streaming})
        return stream

    fake_datasets.load_dataset = load_dataset

    rows = dl.load_hf_split("org/ds", "validation", n=5, seed=42, config="2017")
    assert rows == [{"i": i} for i in range(5)]
    assert calls == [{"hf_id": "org/ds", "name": "2017", "split": "validation",
                      "streaming": True}]
    assert stream.shuffle_calls == [(42, 1000)]


def test_streaming_buffer_scales_with_n(fake_datasets):
    stream = FakeStreamingDataset([{"i": i} for i in range(500)])
    fake_datasets.load_dataset = lambda *a, **k: stream
    dl.load_hf_split("org/ds", "train", n=300, seed=7)
    assert stream.shuffle_calls == [(7, 3000)]


def test_streaming_with_n_zero_takes_everything(fake_datasets):
    stream = FakeStreamingDataset([{"i": 0}, {"i": 1}])
    fake_datasets.load_dataset = lambda *a, **k: stream
    assert dl.load_hf_split("org/ds", "train", n=0, seed=1) == [{"i": 0}, {"i": 1}]


def test_falls_back_to_regular_load_when_streaming_raises(fake_datasets, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / ".cache")
    mapped = FakeMapDataset([{"i": i} for i in range(10)])
    seen = []

    def load_dataset(hf_id, name=None, split=None, streaming=False, cache_dir=None):
        seen.append({"streaming": streaming, "cache_dir": cache_dir})
        if streaming:
            raise RuntimeError("streaming 미지원")
        return mapped

    fake_datasets.load_dataset = load_dataset

    rows = dl.load_hf_split("org/local", "train", n=3, seed=99)
    assert rows == [{"i": 0}, {"i": 1}, {"i": 2}]
    assert mapped.shuffled_with == 99
    assert seen[-1]["cache_dir"] == str(tmp_path / ".cache" / "hf")


def test_falls_back_when_streaming_yields_no_rows(fake_datasets, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / ".cache")

    def load_dataset(hf_id, name=None, split=None, streaming=False, cache_dir=None):
        return FakeStreamingDataset([]) if streaming else FakeMapDataset([{"i": 0}])

    fake_datasets.load_dataset = load_dataset
    assert dl.load_hf_split("org/ds", "train", n=2, seed=1) == [{"i": 0}]


def test_regular_load_keeps_all_rows_when_n_exceeds_dataset(fake_datasets, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / ".cache")
    mapped = FakeMapDataset([{"i": 0}, {"i": 1}])

    def load_dataset(hf_id, name=None, split=None, streaming=False, cache_dir=None):
        if streaming:
            raise RuntimeError("no streaming")
        return mapped

    fake_datasets.load_dataset = load_dataset
    assert dl.load_hf_split("org/ds", "train", n=50, seed=1) == [{"i": 0}, {"i": 1}]
    assert mapped.shuffled_with is None


# ---------------------------------------------------------- get_label_names ---
class FakeClassLabel:
    def __init__(self, names):
        self.names = names


class FakeSequence:
    def __init__(self, feature):
        self.feature = feature


def builder_with(features):
    info = types.SimpleNamespace(features=features)
    return types.SimpleNamespace(info=info)


def test_label_names_from_flat_class_label(fake_datasets, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / ".cache")
    fake_datasets.load_dataset_builder = lambda *a, **k: builder_with(
        {"label": FakeClassLabel(["cat", "dog"])})
    assert dl.get_label_names("org/ds", "train", None, "label") == ["cat", "dog"]


def test_label_names_from_nested_sequence_column(fake_datasets, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / ".cache")
    features = {"objects": FakeSequence({"category": FakeClassLabel(["person", "car"])})}
    fake_datasets.load_dataset_builder = lambda *a, **k: builder_with(features)
    assert dl.get_label_names("org/coco", "val", None, "objects.category") == ["person", "car"]


def test_label_names_none_when_features_missing(fake_datasets, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / ".cache")
    fake_datasets.load_dataset_builder = lambda *a, **k: builder_with(None)
    assert dl.get_label_names("org/ds", "train", None, "label") is None


def test_label_names_none_for_unknown_column(fake_datasets, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / ".cache")
    fake_datasets.load_dataset_builder = lambda *a, **k: builder_with(
        {"label": FakeClassLabel(["a"])})
    assert dl.get_label_names("org/ds", "train", None, "missing") is None


def test_label_names_none_when_column_has_no_names(fake_datasets, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / ".cache")
    fake_datasets.load_dataset_builder = lambda *a, **k: builder_with({"text": object()})
    assert dl.get_label_names("org/ds", "train", None, "text") is None


def test_label_names_none_when_builder_fails(fake_datasets, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / ".cache")

    def boom(*a, **k):
        raise RuntimeError("offline")

    fake_datasets.load_dataset_builder = boom
    assert dl.get_label_names("org/ds", "train", None, "label") is None
