"""runner orchestrator 테스트 (src/runner.py) — FMAPI 호출 없이 스텁으로 검증."""

import json
from pathlib import Path

import pytest
import yaml
from src.config import ModelsConfig
from src.runner import (
    _prompt_text,
    _reproducibility_meta,
    _score_groups,
    _truncate,
    _write_json,
    build_execution_matrix,
    load_tasks_config,
)
from src.tasks.base import Sample, Task

from src import runner

MODELS_CFG = ModelsConfig(
    profile="ai_devtools",
    judge="databricks-gemini-3-1-pro",
    reasoning_modes=["minimal", "full"],
    models=[
        {"id": "opus", "endpoint": "databricks-claude-opus-5", "family": "claude",
         "capabilities": ["text", "vision"],
         "reasoning": {"full": {"thinking": {"type": "enabled"}}}},
        {"id": "sol", "endpoint": "databricks-sol", "family": "openai",
         "capabilities": ["text"]},
    ],
)

TASKS_CFG = {
    "image_tasks": [{"id": "IMG-1", "datasets": {"en": "coco"}}],
    "text_tasks": [{"id": "TXT-4"}, {"id": "TXT-5"}],
    "defaults": {"samples": 7, "seed": 13},
}


# --------------------------------------------------------- load_tasks_config ---
def test_load_tasks_config(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text(yaml.safe_dump(TASKS_CFG), encoding="utf-8")
    assert load_tasks_config(path)["defaults"]["seed"] == 13


def test_load_tasks_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tasks_config(tmp_path / "nope.yaml")


# ---------------------------------------------------- build_execution_matrix ---
def test_matrix_covers_models_tasks_and_modes():
    matrix, na = build_execution_matrix(MODELS_CFG, TASKS_CFG)
    # opus: 3 태스크 × 2 모드 = 6, sol: 텍스트 2 태스크 × 2 모드 = 4
    assert len(matrix) == 10
    assert na == 2  # sol × IMG-1 × 2 모드
    assert {c["model_id"] for c in matrix} == {"opus", "sol"}
    assert matrix[0] == {"model_id": "opus", "model_endpoint": "databricks-claude-opus-5",
                         "task_id": "IMG-1", "reasoning_mode": "minimal"}


def test_vision_tasks_skipped_for_text_only_models():
    matrix, _ = build_execution_matrix(MODELS_CFG, TASKS_CFG)
    assert not [c for c in matrix if c["model_id"] == "sol" and c["task_id"] == "IMG-1"]


def test_models_filter_limits_matrix():
    matrix, na = build_execution_matrix(MODELS_CFG, TASKS_CFG, models_filter=["sol"])
    assert {c["model_id"] for c in matrix} == {"sol"}
    assert na == 2


def test_reasoning_override_replaces_config_modes():
    matrix, na = build_execution_matrix(MODELS_CFG, TASKS_CFG,
                                       reasoning_override=["minimal"])
    assert {c["reasoning_mode"] for c in matrix} == {"minimal"}
    assert na == 1


def test_unknown_model_filter_gives_empty_matrix():
    matrix, na = build_execution_matrix(MODELS_CFG, TASKS_CFG, models_filter=["ghost"])
    assert matrix == [] and na == 0


def test_matrix_with_no_tasks():
    assert build_execution_matrix(MODELS_CFG, {}) == ([], 0)


# ----------------------------------------------------------------- 헬퍼 함수 ---
def test_prompt_text_from_plain_and_multimodal_messages():
    messages = [
        {"role": "user", "content": "첫 질문"},
        {"role": "user", "content": [{"type": "text", "text": "설명해줘"},
                                     {"type": "image_url", "image_url": {"url": "data:..."}}]},
    ]
    assert _prompt_text(messages) == "첫 질문\n설명해줘"


def test_prompt_text_ignores_unknown_content_shapes():
    assert _prompt_text([{"role": "user"}, {"content": 42}, {"content": [None]}]) == ""


@pytest.mark.parametrize("text,limit,expected", [
    ("abc", 5, "abc"),
    ("abcde", 5, "abcde"),
    ("abcdef", 5, "abcde…"),
])
def test_truncate(text, limit, expected):
    assert _truncate(text, limit) == expected


def test_write_json_stringifies_unserializable(tmp_path):
    path = tmp_path / "scores.json"
    _write_json(path, {"metric": {1, 2}, "한글": "값"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["한글"] == "값"
    assert isinstance(data["metric"], str)


# ------------------------------------------------------------- _score_groups ---
class DummyTask(Task):
    task_id = "TXT-4"

    def __init__(self, config=None, registry=None, *, fail=False):
        super().__init__(config or {}, registry or {})
        self.fail = fail
        self.seen = None

    def load_samples(self, n, seed):
        return [Sample(sample_id=i, inputs={"text": f"q{i}"}, reference=f"a{i}")
                for i in range(n)]

    def build_prompt(self, sample):
        return [{"role": "user", "content": sample.inputs["text"]}]

    def parse_output(self, raw_text, sample):
        if raw_text.startswith("__ERROR__"):
            raise ValueError("파싱 불가")
        return raw_text.strip()

    def score(self, parsed, samples):
        if self.fail:
            raise RuntimeError("채점 실패")
        self.seen = (list(parsed), list(samples))
        correct = sum(1 for p, s in zip(parsed, samples, strict=True) if p == s.reference)
        return {"accuracy": correct / len(samples)}


def test_score_groups_calls_task_score():
    task = DummyTask()
    samples = task.load_samples(2, 0)
    groups = {("opus", "TXT-4", "minimal"): {"parsed": ["a0", "wrong"],
                                             "samples": samples, "outputs": []}}
    out = _score_groups(groups, {"TXT-4": (task, samples)})
    entry = out["opus::TXT-4::minimal"]
    assert entry["metrics"] == {"accuracy": 0.5}
    assert (entry["model_id"], entry["task_id"], entry["reasoning_mode"], entry["n"]) == \
        ("opus", "TXT-4", "minimal", 2)


def test_score_groups_records_scoring_error():
    task = DummyTask(fail=True)
    samples = task.load_samples(1, 0)
    groups = {("opus", "TXT-4", "full"): {"parsed": [None], "samples": samples,
                                          "outputs": []}}
    metrics = _score_groups(groups, {"TXT-4": (task, samples)})["opus::TXT-4::full"]["metrics"]
    assert metrics["error"].startswith("RuntimeError")


def test_score_groups_skips_unimplemented_task():
    groups = {("opus", "TXT-9", "full"): {"parsed": [], "samples": [], "outputs": []}}
    assert _score_groups(groups, {"TXT-9": (None, [])}) == {}


# -------------------------------------------------------- _reproducibility_meta ---
def test_reproducibility_meta_snapshots_datasets_and_pricing(monkeypatch):
    monkeypatch.setattr("src.datasets_loader.load_registry",
                        lambda *a, **k: {"coco": {"hf_id": "org/coco", "split": "val",
                                                  "config": "2017"}})
    monkeypatch.setattr("src.cost.pricing.load_pricing",
                        lambda *a, **k: {"usd_per_dbu": 0.07, "routing": "pay-per-token"})
    datasets, pricing = _reproducibility_meta([{"id": "IMG-1", "datasets": {"en": "coco"}}])
    assert datasets == {"coco": {"hf_id": "org/coco", "split": "val", "config": "2017"}}
    assert pricing == {"usd_per_dbu": 0.07, "routing": "pay-per-token"}


def test_reproducibility_meta_unknown_dataset_key(monkeypatch):
    monkeypatch.setattr("src.datasets_loader.load_registry", lambda *a, **k: {})
    monkeypatch.setattr("src.cost.pricing.load_pricing", lambda *a, **k: {})
    datasets, _ = _reproducibility_meta([{"id": "IMG-1", "datasets": {"en": "ghost"}}])
    assert datasets == {"ghost": {"hf_id": None, "split": None, "config": None}}


def test_reproducibility_meta_tolerates_load_failures(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("registry 없음")

    monkeypatch.setattr("src.datasets_loader.load_registry", boom)
    monkeypatch.setattr("src.cost.pricing.load_pricing", boom)
    assert _reproducibility_meta([{"id": "IMG-1", "datasets": {"en": "coco"}}]) == ({}, {})


def test_reproducibility_meta_tasks_without_datasets(monkeypatch):
    monkeypatch.setattr("src.datasets_loader.load_registry", lambda *a, **k: {})
    monkeypatch.setattr("src.cost.pricing.load_pricing", lambda *a, **k: {})
    datasets, _ = _reproducibility_meta([{"id": "TXT-4"}, {"id": "TXT-5", "datasets": None}])
    assert datasets == {}


# -------------------------------------------------------------------- main() ---
@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """config 파일을 tmp에 만들고 cwd를 옮긴다. main()은 상대경로를 쓴다."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    models_yaml = tmp_path / "config" / "models.yaml"
    models_yaml.write_text(yaml.safe_dump(MODELS_CFG.model_dump()), encoding="utf-8")
    (tmp_path / "config" / "tasks.yaml").write_text(yaml.safe_dump(TASKS_CFG),
                                                   encoding="utf-8")
    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(runner, "_reproducibility_meta", lambda tasks: ({}, {}))
    return tmp_path


def run_cli(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["runner", *args])
    return runner.main()


def test_dry_run_writes_manifest_without_calling_fmapi(cli_env, monkeypatch, capsys):
    def no_client(*a, **k):
        pytest.fail("dry-run에서 FMAPIClient를 만들면 안 됨")

    monkeypatch.setattr(runner, "FMAPIClient", no_client)
    assert run_cli(monkeypatch, "--dry-run") == 0

    out = capsys.readouterr().out
    assert "DRY-RUN 모드" in out
    assert "총 실행 셀: 10" in out
    assert "N/A 스킵(vision 미지원): 2" in out

    manifest = json.loads(next((cli_env / "results").glob("*/manifest.json")).read_text())
    assert manifest["notes"] == "dry-run"
    assert manifest["git_commit"] == "abc1234"
    assert manifest["task_ids"] == ["IMG-1", "TXT-4", "TXT-5"]
    assert manifest["samples_per_task"] == 7
    assert manifest["seed"] == 13
    assert [m["id"] for m in manifest["models"]] == ["opus", "sol"]


def test_cli_filters_reach_manifest(cli_env, monkeypatch):
    assert run_cli(monkeypatch, "--dry-run", "--models", "sol", "--reasoning-modes",
                   "minimal", "--samples", "3") == 0
    manifest = json.loads(next((cli_env / "results").glob("*/manifest.json")).read_text())
    assert [m["id"] for m in manifest["models"]] == ["sol"]
    assert manifest["reasoning_modes"] == ["minimal"]
    assert manifest["samples_per_task"] == 3


def test_cli_honors_out_directory(cli_env, monkeypatch):
    assert run_cli(monkeypatch, "--dry-run", "--out", "custom-out") == 0
    assert list((cli_env / "custom-out").glob("*/manifest.json"))


def test_missing_config_returns_error(cli_env, monkeypatch, capsys):
    assert run_cli(monkeypatch, "--dry-run", "--config", "config/absent.yaml") == 1
    assert "설정 파일을 찾을 수 없음" in capsys.readouterr().err


def test_invalid_config_returns_error(cli_env, monkeypatch, capsys):
    (cli_env / "config" / "broken.yaml").write_text("models: []", encoding="utf-8")
    assert run_cli(monkeypatch, "--dry-run", "--config", "config/broken.yaml") == 1
    assert "설정 파일 파싱 실패" in capsys.readouterr().err


def test_empty_matrix_returns_error(cli_env, monkeypatch, capsys):
    assert run_cli(monkeypatch, "--dry-run", "--models", "ghost") == 1
    assert "실행할 항목이 없습니다" in capsys.readouterr().err


def test_fmapi_init_failure_returns_error(cli_env, monkeypatch, capsys):
    def boom(**kwargs):
        raise RuntimeError("databricks CLI 없음")

    monkeypatch.setattr(runner, "FMAPIClient", boom)
    assert run_cli(monkeypatch) == 1
    assert "FMAPI 클라이언트 초기화 실패" in capsys.readouterr().err


# --------------------------------------------------------------- 전체 실행 경로 ---
class FakeChatResponse:
    def __init__(self, text):
        self.text = text
        self.request_id = "req-1"
        self.finish_reason = "stop"
        self.usage = {"prompt_tokens": 5, "completion_tokens": 2}


class FakeFMAPIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.closed = False

    def chat(self, endpoint, messages, max_tokens=1024, extra_params=None):
        self.calls.append({"endpoint": endpoint, "messages": messages,
                           "max_tokens": max_tokens, "extra_params": extra_params})
        if endpoint == "databricks-sol":
            raise RuntimeError("엔드포인트 다운")
        return FakeChatResponse(messages[0]["content"].replace("q", "a"))

    def close(self):
        self.closed = True


@pytest.fixture
def full_run(cli_env, monkeypatch):
    """실제 실행 경로용: 태스크 플러그인·레지스트리·클라이언트를 스텁으로 대체."""
    client = FakeFMAPIClient()
    monkeypatch.setattr(runner, "FMAPIClient", lambda **kwargs: client)
    monkeypatch.setattr("src.datasets_loader.load_registry", lambda *a, **k: {})
    monkeypatch.setattr("src.tasks.loader.discover_tasks", lambda: {"TXT-4": DummyTask})
    monkeypatch.setattr("src.report.generate.generate_report",
                        lambda *a, **k: Path("reports/x/report.md"))
    monkeypatch.setattr("src.report.index.rebuild_index", lambda *a, **k: Path("reports/index.md"))
    return cli_env, client


def test_full_run_writes_samples_and_scores(full_run, monkeypatch, capsys):
    root, client = full_run
    assert run_cli(monkeypatch, "--models", "opus", "--reasoning-modes", "minimal",
                   "--samples", "2") == 0

    run_dir = next((root / "results").iterdir())
    lines = (run_dir / "samples.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2  # TXT-4 × 2 샘플 (IMG-1·TXT-5는 미구현 → 스킵)
    first = json.loads(lines[0])
    assert first["model_id"] == "opus" and first["task_id"] == "TXT-4"
    assert first["model_output"] == "a0"
    assert first["request_id"] == "req-1"
    assert first["latency_ms_local"] >= 0

    scores = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
    assert scores["opus::TXT-4::minimal"]["metrics"] == {"accuracy": 1.0}
    assert scores["opus::TXT-4::minimal"]["n"] == 2

    out = capsys.readouterr().out
    assert "2개 샘플 호출" in out
    assert "리포트 생성" in out
    assert "인덱스 갱신" in out


def test_full_run_passes_reasoning_params(full_run, monkeypatch):
    _, client = full_run
    run_cli(monkeypatch, "--models", "opus", "--reasoning-modes", "full", "--samples", "1")
    assert client.calls[0]["extra_params"] == {"thinking": {"type": "enabled"}}
    assert client.calls[0]["endpoint"] == "databricks-claude-opus-5"


def test_samples_are_shared_across_modes(full_run, monkeypatch):
    root, _ = full_run
    run_cli(monkeypatch, "--models", "opus", "--samples", "2")
    run_dir = next((root / "results").iterdir())
    raw = (run_dir / "samples.jsonl").read_text(encoding="utf-8").strip()
    rows = [json.loads(line) for line in raw.split("\n")]
    minimal = [r["reference"] for r in rows if r["reasoning_mode"] == "minimal"]
    full = [r["reference"] for r in rows if r["reasoning_mode"] == "full"]
    assert minimal == full == ["a0", "a1"]


def test_call_failure_recorded_as_error_sample(full_run, monkeypatch):
    root, _ = full_run
    assert run_cli(monkeypatch, "--models", "sol", "--reasoning-modes", "minimal",
                   "--samples", "1") == 0
    run_dir = next((root / "results").iterdir())
    row = json.loads((run_dir / "samples.jsonl").read_text(encoding="utf-8").strip())
    assert row["finish_reason"] == "error"
    assert row["request_id"] is None
    assert row["usage"] == {}
    assert "__ERROR__: RuntimeError: 엔드포인트 다운" in row["model_output"]


def test_sample_load_failure_skips_task(full_run, monkeypatch, capsys):
    class BrokenTask(DummyTask):
        def load_samples(self, n, seed):
            raise RuntimeError("데이터셋 다운로드 실패")

    monkeypatch.setattr("src.tasks.loader.discover_tasks", lambda: {"TXT-4": BrokenTask})
    root, _ = full_run
    assert run_cli(monkeypatch, "--models", "opus", "--samples", "1") == 0
    out = capsys.readouterr().out
    assert "[샘플 로드 실패] TXT-4" in out
    assert "0개 샘플 호출" in out


def test_report_and_index_failures_do_not_fail_run(full_run, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("디스크 오류")

    monkeypatch.setattr("src.report.generate.generate_report", boom)
    monkeypatch.setattr("src.report.index.rebuild_index", boom)
    assert run_cli(monkeypatch, "--models", "opus", "--samples", "1") == 0
    out = capsys.readouterr().out
    assert "[리포트 생성 스킵] RuntimeError" in out
    assert "[인덱스 갱신 스킵] RuntimeError" in out


def test_parse_failure_still_scores_group(full_run, monkeypatch):
    root, _ = full_run
    run_cli(monkeypatch, "--models", "sol", "--reasoning-modes", "minimal", "--samples", "2")
    run_dir = next((root / "results").iterdir())
    scores = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
    # parse_output이 예외 → parsed=None → accuracy 0.0 이지만 실행은 계속
    assert scores["sol::TXT-4::minimal"]["metrics"] == {"accuracy": 0.0}


def test_client_is_closed_after_run(full_run, monkeypatch):
    _, client = full_run
    run_cli(monkeypatch, "--models", "opus", "--samples", "1")
    assert client.closed is True
