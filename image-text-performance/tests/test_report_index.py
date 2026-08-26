"""리포트 인덱스 재생성 테스트 (src/report/index.py)."""

import json

from src.report.index import rebuild_index


def make_run(reports_root, results_root, run_id, manifest=None, *, with_report=True):
    run_dir = reports_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if with_report:
        (run_dir / "report.md").write_text(f"# {run_id}", encoding="utf-8")
    if manifest is not None:
        res = results_root / run_id
        res.mkdir(parents=True, exist_ok=True)
        (res / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


MANIFEST = {
    "created_at": "2026-07-31T14:00:00+00:00",
    "models": [{"id": "opus"}, {"id": "glm"}],
    "reasoning_modes": ["minimal", "full"],
    "task_ids": ["TXT-4", "IMG-1"],
    "samples_per_task": 30,
    "git_commit": "abc1234",
}


def test_index_lists_runs_newest_first(tmp_path):
    reports, results = tmp_path / "reports", tmp_path / "results"
    make_run(reports, results, "2026-07-30T10-00", MANIFEST)
    make_run(reports, results, "2026-08-01T10-00", MANIFEST)

    index = rebuild_index(reports, results)
    body = index.read_text(encoding="utf-8")
    assert index.name == "index.md"
    assert "총 2개 run" in body
    assert body.index("2026-08-01T10-00") < body.index("2026-07-30T10-00")


def test_index_row_summarizes_manifest(tmp_path):
    reports, results = tmp_path / "reports", tmp_path / "results"
    make_run(reports, results, "2026-07-31T14-00", MANIFEST)

    body = rebuild_index(reports, results).read_text(encoding="utf-8")
    assert "[2026-07-31T14-00](2026-07-31T14-00/report.md)" in body
    assert "| opus,glm |" in body
    assert "| minimal,full |" in body
    assert "| 2 |" in body
    assert "| 30 |" in body
    assert "| abc1234 |" in body
    assert "2026-07-31T14:00:00" in body


def test_index_uses_placeholders_without_manifest(tmp_path):
    reports, results = tmp_path / "reports", tmp_path / "results"
    make_run(reports, results, "2026-07-31T14-00")

    body = rebuild_index(reports, results).read_text(encoding="utf-8")
    row = [line for line in body.splitlines() if "2026-07-31T14-00" in line][0]
    assert row.count("?") >= 5


def test_index_tolerates_corrupt_manifest(tmp_path):
    reports, results = tmp_path / "reports", tmp_path / "results"
    make_run(reports, results, "2026-07-31T14-00")
    res = results / "2026-07-31T14-00"
    res.mkdir(parents=True)
    (res / "manifest.json").write_text("{broken", encoding="utf-8")

    assert "2026-07-31T14-00" in rebuild_index(reports, results).read_text(encoding="utf-8")


def test_index_skips_dirs_without_report(tmp_path):
    reports, results = tmp_path / "reports", tmp_path / "results"
    make_run(reports, results, "2026-07-31T14-00", MANIFEST)
    make_run(reports, results, "2026-08-01T10-00", MANIFEST, with_report=False)

    body = rebuild_index(reports, results).read_text(encoding="utf-8")
    assert "총 1개 run" in body
    assert "2026-08-01T10-00" not in body


def test_index_ignores_stray_files(tmp_path):
    reports, results = tmp_path / "reports", tmp_path / "results"
    reports.mkdir()
    (reports / "notes.md").write_text("x", encoding="utf-8")

    assert "총 0개 run" in rebuild_index(reports, results).read_text(encoding="utf-8")


def test_index_is_regenerated_not_appended(tmp_path):
    reports, results = tmp_path / "reports", tmp_path / "results"
    make_run(reports, results, "2026-07-31T14-00", MANIFEST)
    rebuild_index(reports, results)
    body = rebuild_index(reports, results).read_text(encoding="utf-8")
    assert body.count("2026-07-31T14-00/report.md") == 1


def test_index_creates_reports_root_when_missing(tmp_path):
    index = rebuild_index(tmp_path / "reports", tmp_path / "results")
    assert index.exists()
    assert "총 0개 run" in index.read_text(encoding="utf-8")
