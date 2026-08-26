"""benchmark.grade_tasks — discovery, validation, scoring, gallery, human merge."""

import json

import pytest

from benchmark import grade_tasks, task_spec

from .conftest import KEYWORDS, deck

CFG = KEYWORDS


def make_candidate(bench_root, name, *, html=None, meta=None, task="demo-task"):
    d = bench_root / task / name
    d.mkdir(parents=True, exist_ok=True)
    if html is not None:
        (d / task_spec.ARTIFACT).write_text(html, encoding="utf-8")
    if meta is not None:
        (d / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


# ---------------------------------------------------------------- discovery ---
def test_discover_candidates_reads_meta_and_sorts(bench_root):
    make_candidate(bench_root, "opus", html=deck(), meta={"harness": "codex"})
    make_candidate(bench_root, "glm", html=deck())
    cands = grade_tasks.discover_candidates("demo-task", None)
    assert [c["name"] for c in cands] == ["glm", "opus"]
    assert cands[1]["meta"] == {"harness": "codex"}
    assert cands[0]["meta"] == {}


def test_discover_candidates_accepts_meta_only_dir(bench_root):
    make_candidate(bench_root, "aborted", meta={"harness": "codex"})
    assert [c["name"] for c in grade_tasks.discover_candidates("demo-task", None)] == ["aborted"]


def test_discover_candidates_skips_reserved_and_empty_dirs(bench_root):
    make_candidate(bench_root, "opus", html=deck())
    (bench_root / "demo-task" / "gallery").mkdir()
    (bench_root / "demo-task" / "__pycache__").mkdir()
    (bench_root / "demo-task" / "notes").mkdir()
    assert [c["name"] for c in grade_tasks.discover_candidates("demo-task", None)] == ["opus"]


def test_discover_candidates_applies_name_filter(bench_root):
    make_candidate(bench_root, "opus", html=deck())
    make_candidate(bench_root, "glm", html=deck())
    assert [c["name"] for c in grade_tasks.discover_candidates("demo-task", ["glm"])] == ["glm"]


def test_discover_candidates_tolerates_corrupt_meta(bench_root):
    d = make_candidate(bench_root, "opus", html=deck())
    (d / "run_meta.json").write_text("{not json", encoding="utf-8")
    assert grade_tasks.discover_candidates("demo-task", None)[0]["meta"] == {}


def test_discover_candidates_missing_task_dir(bench_root):
    assert grade_tasks.discover_candidates("no-such-task", None) == []


# --------------------------------------------------------------- validation ---
def test_validate_html_happy_path(bench_root):
    path = make_candidate(bench_root, "opus", html=deck(2)) / task_spec.ARTIFACT
    v = grade_tasks.validate_html(path, CFG)
    assert v["parse_ok"] and v["has_doctype"]
    assert v["slide_count"] == 2 and v["slide_count_ok"]
    assert v["keywords_found"] == 3 and v["keywords_total"] == 3
    assert v["keywords_missing"] == []
    assert v["external_refs"] == 0


def test_validate_html_counts_reveal_sections(bench_root):
    html = "<!DOCTYPE html><html><body>" + "<section>spark</section>" * 3 + "</body></html>"
    path = make_candidate(bench_root, "opus", html=html) / task_spec.ARTIFACT
    v = grade_tasks.validate_html(path, CFG)
    assert v["slide_count"] == 3 and v["slide_count_ok"]


def test_validate_html_slide_count_out_of_bounds(bench_root):
    path = make_candidate(bench_root, "opus", html=deck(5)) / task_spec.ARTIFACT
    v = grade_tasks.validate_html(path, CFG)
    assert v["slide_count"] == 5 and v["slide_count_ok"] is False


def test_validate_html_reports_missing_topics(bench_root):
    path = make_candidate(bench_root, "opus", html=deck(2, text="spark only")) / task_spec.ARTIFACT
    v = grade_tasks.validate_html(path, CFG)
    assert v["keywords_found"] == 1
    assert sorted(v["keywords_missing"]) == ["delta", "lakehouse"]


def test_validate_html_keyword_matching_is_case_insensitive(bench_root):
    path = make_candidate(bench_root, "opus",
                          html=deck(2, text="LAKEHOUSE Delta Lake SPARK")) / task_spec.ARTIFACT
    assert grade_tasks.validate_html(path, CFG)["keywords_found"] == 3


def test_validate_html_flags_external_refs(bench_root):
    html = deck(2, external=True) + '<script src="https://cdn.example/y.js"></script>'
    path = make_candidate(bench_root, "opus", html=html) / task_spec.ARTIFACT
    assert grade_tasks.validate_html(path, CFG)["external_refs"] == 2


def test_validate_html_ignores_relative_refs(bench_root):
    html = ('<!DOCTYPE html><html><head><link rel="stylesheet" href="./a.css">'
            '<script src="b.js"></script></head><body>'
            '<div class="slide">lakehouse delta spark</div>'
            '<div class="slide">x</div></body></html>')
    path = make_candidate(bench_root, "opus", html=html) / task_spec.ARTIFACT
    assert grade_tasks.validate_html(path, CFG)["external_refs"] == 0


def test_validate_html_missing_doctype(bench_root):
    path = make_candidate(bench_root, "opus", html=deck(2, doctype=False)) / task_spec.ARTIFACT
    v = grade_tasks.validate_html(path, CFG)
    assert v["has_doctype"] is False and v["parse_ok"] is True


def test_validate_html_missing_file(tmp_path):
    v = grade_tasks.validate_html(tmp_path / "nope.html", CFG)
    assert v["parse_ok"] is False and v["slide_count"] == 0


def test_validate_html_empty_file(tmp_path):
    path = tmp_path / "slides.html"
    path.write_text("", encoding="utf-8")
    assert grade_tasks.validate_html(path, CFG)["parse_ok"] is False


def test_validate_html_unparseable_content(tmp_path):
    path = tmp_path / "slides.html"
    path.write_text("   ", encoding="utf-8")
    assert grade_tasks.validate_html(path, CFG)["parse_ok"] is False


# ------------------------------------------------------------------ scoring ---
def _grade(bench_root, monkeypatch, *, html=None, meta=None, render=None, do_render=True):
    make_candidate(bench_root, "opus", html=deck(2) if html is None else html,
                   meta=meta or {})
    if render is not None:
        monkeypatch.setattr(grade_tasks, "render_and_capture", lambda *_a, **_k: render)
    cand = grade_tasks.discover_candidates("demo-task", None)[0]
    return grade_tasks.grade_one(cand, CFG, do_render=do_render)


RENDER_OK = {"rendered_ok": True, "console_errors": 0, "screenshot_paths": ["slide_01.png"],
             "render_note": ""}


def test_grade_one_perfect_score(bench_root, monkeypatch):
    row = _grade(bench_root, monkeypatch, render=RENDER_OK,
                 meta={"harness": "codex", "effective_model": "databricks-x",
                       "mode": "headless", "wall_seconds": 12.5})
    assert row["auto_score"] == 1.0
    assert row["valid"] is True
    assert row["keywords"] == "3/3"
    assert row["console_errors"] == 0
    assert row["model"] == "databricks-x"
    assert row["screenshots"] == ["slide_01.png"]
    assert row["note"] == ""


def test_grade_one_console_errors_cost_the_render_term(bench_root, monkeypatch):
    render = {**RENDER_OK, "console_errors": 3}
    row = _grade(bench_root, monkeypatch, render=render)
    assert row["auto_score"] == 0.7
    assert row["console_errors"] == 3
    assert row["valid"] is True


def test_grade_one_skipped_render_scores_zero_for_that_term(bench_root, monkeypatch):
    row = _grade(bench_root, monkeypatch, do_render=False)
    assert row["auto_score"] == 0.7
    assert row["console_errors"] is None
    assert row["valid"] is True  # rendering was skipped, not failed


def test_grade_one_render_failure_is_invalid(bench_root, monkeypatch):
    render = {"rendered_ok": False, "console_errors": 0, "screenshot_paths": [],
              "render_note": "chromium launch failed"}
    row = _grade(bench_root, monkeypatch, render=render)
    assert row["valid"] is False
    assert "chromium launch failed" in row["note"]


def test_grade_one_partial_keywords_and_bad_slide_count(bench_root, monkeypatch):
    row = _grade(bench_root, monkeypatch, html=deck(5, text="spark"), render=RENDER_OK)
    assert row["auto_score"] == round(0.4 / 3 + 0.3, 3)
    assert row["valid"] is False
    assert "missing topics" in row["note"]


def test_grade_one_notes_external_refs(bench_root, monkeypatch):
    row = _grade(bench_root, monkeypatch, html=deck(2, external=True), render=RENDER_OK)
    assert "1 external http ref(s)" in row["note"]


def test_grade_one_falls_back_to_requested_model_then_dash(bench_root, monkeypatch):
    row = _grade(bench_root, monkeypatch, render=RENDER_OK, meta={"model": "databricks-req"})
    assert row["model"] == "databricks-req"
    assert row["harness"] == "-"
    assert row["mode"] == "-"


def test_grade_one_missing_artifact(bench_root, monkeypatch):
    make_candidate(bench_root, "glm", meta={"harness": "codex"})
    monkeypatch.setattr(grade_tasks, "render_and_capture", lambda *_a, **_k: RENDER_OK)
    cand = grade_tasks.discover_candidates("demo-task", ["glm"])[0]
    row = grade_tasks.grade_one(cand, CFG, do_render=True)
    assert row["valid"] is False
    assert row["slide_count"] == 0
    assert row["auto_score"] == 0.3  # only the no-console-errors term


def test_grade_one_zero_keyword_groups_is_not_a_zero_division(bench_root, monkeypatch):
    make_candidate(bench_root, "opus", html=deck(2))
    monkeypatch.setattr(grade_tasks, "render_and_capture", lambda *_a, **_k: RENDER_OK)
    cand = grade_tasks.discover_candidates("demo-task", None)[0]
    cfg = {**CFG, "keywords": {}}
    row = grade_tasks.grade_one(cand, cfg, do_render=True)
    assert row["keywords"] == "0/0"
    assert row["auto_score"] == 0.6


# -------------------------------------------------------------- render skip ---
def test_render_and_capture_without_playwright(tmp_path, monkeypatch):
    """Missing Playwright must degrade to a note, never raise."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("playwright"):
            raise ImportError("no playwright")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = grade_tasks.render_and_capture(tmp_path / "slides.html", tmp_path / "shots")
    assert out["rendered_ok"] is False
    assert "playwright unavailable" in out["render_note"]
    assert out["screenshot_paths"] == []


# ------------------------------------------------------------------ outputs ---
ROW = {
    "candidate": "opus", "harness": "codex", "model": "databricks-x", "mode": "headless",
    "valid": True, "slide_count": 2, "keywords": "3/3", "console_errors": 0,
    "external_refs": 0, "wall_seconds": 12.5, "auto_score": 1.0,
    "screenshots": ["slide_01.png"], "note": "",
}


def test_print_table_renders_every_row(capsys):
    grade_tasks.print_table("demo-task", [ROW, {**ROW, "candidate": "glm",
                                                "console_errors": None,
                                                "wall_seconds": None,
                                                "note": "missing topics: delta"}])
    out = capsys.readouterr().out
    assert "opus" in out and "glm" in out
    assert "note[glm]: missing topics: delta" in out
    assert "note[opus]" not in out


def test_build_gallery_writes_scoreable_page(tmp_path):
    out = tmp_path / "gallery" / "index.html"
    grade_tasks.build_gallery("demo-task", [ROW], out)
    doc = out.read_text(encoding="utf-8")
    assert 'data-candidate="opus"' in doc
    assert "../opus/screenshots/slide_01.png" in doc
    assert "../opus/slides.html" in doc
    assert "human_scores.json" in doc
    assert "class=\"human-score\"" in doc


def test_build_gallery_notes_absent_screenshots(tmp_path):
    out = tmp_path / "gallery" / "index.html"
    grade_tasks.build_gallery("demo-task", [{**ROW, "screenshots": [], "note": "render failed"}],
                              out)
    doc = out.read_text(encoding="utf-8")
    assert "no screenshot" in doc
    assert "render failed" in doc


# -------------------------------------------------------------- merge-human ---
def test_merge_human_joins_scores(bench_root, capsys):
    results = bench_root / "demo-task" / "grade_results.json"
    results.write_text(json.dumps([ROW, {**ROW, "candidate": "glm"}]), encoding="utf-8")
    human = bench_root / "human_scores.json"
    human.write_text(json.dumps({"opus": {"human_score": 4, "human_note": "clean"}}),
                     encoding="utf-8")

    grade_tasks.merge_human("demo-task", human)

    merged = {r["candidate"]: r for r in json.loads(results.read_text(encoding="utf-8"))}
    assert merged["opus"]["human_score"] == 4
    assert merged["opus"]["human_note"] == "clean"
    assert merged["glm"]["human_score"] is None
    assert merged["glm"]["human_note"] == ""
    assert "merged human scores" in capsys.readouterr().out


def test_merge_human_without_prior_grading_exits(bench_root, tmp_path):
    human = tmp_path / "human_scores.json"
    human.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        grade_tasks.merge_human("demo-task", human)
    assert "run grading first" in str(e.value)


# --------------------------------------------------------------------- main ---
def _main(monkeypatch, argv):
    monkeypatch.setattr(grade_tasks, "__name__", "benchmark.grade_tasks")
    import sys
    monkeypatch.setattr(sys, "argv", ["grade-task", *argv])
    grade_tasks.main()


def test_main_grades_writes_results_and_gallery(bench_root, monkeypatch):
    make_candidate(bench_root, "opus", html=deck(2), meta={"harness": "codex"})
    _main(monkeypatch, ["--task", "demo-task", "--no-render"])

    rows = json.loads((bench_root / "demo-task" / "grade_results.json")
                      .read_text(encoding="utf-8"))
    assert [r["candidate"] for r in rows] == ["opus"]
    assert (bench_root / "demo-task" / "gallery" / "index.html").exists()


def test_main_errors_without_candidates(bench_root, monkeypatch):
    with pytest.raises(SystemExit):
        _main(monkeypatch, ["--task", "demo-task", "--no-render"])


def test_main_merge_human_short_circuits_grading(bench_root, monkeypatch):
    (bench_root / "demo-task" / "grade_results.json").write_text(json.dumps([ROW]),
                                                                 encoding="utf-8")
    human = bench_root / "human_scores.json"
    human.write_text(json.dumps({"opus": {"human_score": 5, "human_note": ""}}),
                     encoding="utf-8")
    monkeypatch.setattr(grade_tasks, "grade_one",
                        lambda *_a, **_k: pytest.fail("grading must be skipped"))
    _main(monkeypatch, ["--task", "demo-task", "--merge-human", str(human)])
    rows = json.loads((bench_root / "demo-task" / "grade_results.json").read_text(encoding="utf-8"))
    assert rows[0]["human_score"] == 5
