#!/usr/bin/env python3
"""
grade_tasks.py — Grade the slides.html produced by each candidate of a task with an
IDENTICAL grader, then build a human-review gallery.

Layout is task-centric: candidates are subdirectories of the task dir, and results are
written back into it:

    <task>/<candidate>/slides.html   ← graded
    <task>/grade_results.json         ← written
    <task>/gallery/index.html         ← written

Two phases:
  Phase A (automated, per candidate):
    - validate_html:      parse + well-formedness, slide count, keyword coverage,
                          external-ref warnings.
    - render_and_capture: headless Chromium (Playwright) renders file://slides.html,
                          collects console/page errors, screenshots each slide.
    - grade_one:          combine into a row with a transparent auto_score.
    - print_table + grade_results.json  (mirrors agent-ml/grade_agents.py).
  Phase B (human review):
    - build_gallery:      static <task>/gallery/index.html with one card per candidate,
                          screenshots, auto metrics, and a client-side 1-5 score +
                          "download human_scores.json" button (no server).
    - --merge-human:      join a downloaded human_scores.json back onto the rows and
                          re-emit grade_results.json with auto_score + human_score.

Run the CLIs from the repo root (task dirs are resolved from the working directory).

Usage:
  grade-task --task explain-databricks              # grade all candidates
  grade-task --task explain-databricks --candidates opus glm
  grade-task --task explain-databricks --no-render  # validation only
  grade-task --task explain-databricks --merge-human explain-databricks/gallery/human_scores.json
  # equivalently:  python -m benchmark.grade_tasks --task ...
"""
import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote

from lxml import html as lxml_html

from benchmark import task_spec

# Files that live in the task dir but are NOT candidate output directories.
RESERVED = {"gallery", "__pycache__"}


# ---------------------------------------------------------------- discovery ---
def discover_candidates(task: str, filter_names: list[str] | None) -> list[dict]:
    """Find candidate subdirs of the task dir (any dir with slides.html or run_meta.json),
    reading each run_meta.json."""
    tdir = task_spec.task_dir(task)
    if not tdir.exists():
        return []
    out = []
    for d in sorted(tdir.iterdir()):
        if not d.is_dir() or d.name in RESERVED:
            continue
        if not (d / task_spec.ARTIFACT).exists() and not (d / "run_meta.json").exists():
            continue
        if filter_names and d.name not in filter_names:
            continue
        meta = {}
        meta_path = d / "run_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        out.append({"dir": d, "name": d.name, "meta": meta})
    return out


# ------------------------------------------------------------- validation ----
def validate_html(html_path: Path, cfg: dict) -> dict:
    """Parse + well-formedness, slide count, keyword coverage, external refs."""
    out = {
        "parse_ok": False, "has_doctype": False, "slide_count": 0,
        "slide_count_ok": False, "keywords_found": 0, "keywords_total": 0,
        "keywords_missing": [], "external_refs": 0,
    }
    if not html_path.exists() or html_path.stat().st_size == 0:
        return out

    raw = html_path.read_text(encoding="utf-8", errors="replace")
    out["has_doctype"] = raw.lstrip().lower().startswith("<!doctype")
    try:
        doc = lxml_html.fromstring(raw)
    except Exception:
        return out
    out["parse_ok"] = True

    # Slide count: both formats allowed — vanilla .slide OR reveal.js <section>.
    n_slide = len(doc.cssselect(".slide"))
    n_section = len(doc.xpath("//section"))
    out["slide_count"] = max(n_slide, n_section)
    lo, hi = cfg["slide_count"]["min"], cfg["slide_count"]["max"]
    out["slide_count_ok"] = lo <= out["slide_count"] <= hi

    # Keyword coverage over lowercased text content. Each topic group counts if ANY of its
    # keyword variants appears.
    text = " ".join(doc.itertext()).lower()
    groups = cfg["keywords"]
    out["keywords_total"] = len(groups)
    for topic, variants in groups.items():
        if any(v.lower() in text for v in variants):
            out["keywords_found"] += 1
        else:
            out["keywords_missing"].append(topic)

    # External http(s) stylesheet/script refs break the self-contained contract.
    ext = 0
    for el in doc.xpath("//link[@rel='stylesheet'][@href] | //script[@src]"):
        src = el.get("href") or el.get("src") or ""
        if src.startswith("http://") or src.startswith("https://"):
            ext += 1
    out["external_refs"] = ext
    return out


# --------------------------------------------------------------- rendering ---
def render_and_capture(html_path: Path, out_dir: Path, allow_network: bool = False) -> dict:
    """Headless Chromium render via Playwright: collect console/page errors and screenshot
    each slide. Returns rendered_ok, console_errors, screenshot_paths.

    The deck is untrusted, model-generated HTML, so by default every request that is not
    the local file itself (or an inline data:/blob: URL) is aborted — a deck cannot phone
    home or exfiltrate anything it reads. Decks are required to be self-contained anyway;
    pass allow_network=True to grade one that legitimately needs a CDN."""
    out = {"rendered_ok": False, "console_errors": 0, "screenshot_paths": [],
           "render_note": "", "blocked_requests": 0}
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # noqa: BLE001
        out["render_note"] = f"playwright unavailable ({type(e).__name__}: {e})"
        return out

    out_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:  # noqa: BLE001 — chromium not installed
                out["render_note"] = (f"chromium launch failed ({type(e).__name__}: {e}) "
                                      "— run: uv run playwright install chromium")
                return out
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            if not allow_network:
                page.route("**/*", lambda route: _guard_route(route, out))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            page.wait_for_timeout(1500)  # settle time for CSS/JS

            # Per-slide clips if discrete DOM nodes exist; else one full-page shot.
            nodes = page.query_selector_all(".slide") or page.query_selector_all("section")
            if nodes:
                for i, node in enumerate(nodes, 1):
                    shot = out_dir / f"slide_{i:02d}.png"
                    try:
                        node.scroll_into_view_if_needed()
                        node.screenshot(path=str(shot))
                        out["screenshot_paths"].append(shot.name)
                    except Exception:
                        pass
            if not out["screenshot_paths"]:
                shot = out_dir / "slide_full.png"
                page.screenshot(path=str(shot), full_page=True)
                out["screenshot_paths"].append(shot.name)

            out["rendered_ok"] = True
            out["console_errors"] = len(errors)
            browser.close()
    except Exception as e:  # noqa: BLE001
        out["render_note"] = f"render failed ({type(e).__name__}: {e})"
    if out["blocked_requests"]:
        out["render_note"] = "; ".join(filter(None, [
            out["render_note"],
            f"{out['blocked_requests']} network request(s) blocked during render",
        ]))
    return out


def _guard_route(route, out: dict) -> None:
    """Allow only local (file:) and inline (data:/blob:) fetches; abort everything else."""
    if route.request.url.startswith(("file:", "data:", "blob:", "about:")):
        route.continue_()
    else:
        out["blocked_requests"] += 1
        route.abort()


# ----------------------------------------------------------------- scoring ---
def grade_one(cand: dict, cfg: dict, do_render: bool, allow_network: bool = False) -> dict:
    d = cand["dir"]
    meta = cand["meta"]
    html_path = d / task_spec.ARTIFACT

    v = validate_html(html_path, cfg)
    r = ({"rendered_ok": False, "console_errors": 0, "screenshot_paths": [], "render_note": "skipped"}
         if not do_render else render_and_capture(html_path, d / "screenshots", allow_network))

    kw_cov = (v["keywords_found"] / v["keywords_total"]) if v["keywords_total"] else 0.0
    no_console_err = 1.0 if (r["rendered_ok"] and r["console_errors"] == 0) else 0.0
    # Transparent weighted blend. Rendering failure/skip contributes 0 to its term.
    auto_score = round(0.4 * kw_cov + 0.3 * (1.0 if v["slide_count_ok"] else 0.0)
                       + 0.3 * no_console_err, 3)

    valid = bool(v["parse_ok"] and v["slide_count_ok"]
                 and (r["rendered_ok"] or not do_render))

    notes = []
    if v["keywords_missing"]:
        notes.append(f"missing topics: {', '.join(v['keywords_missing'])}")
    if v["external_refs"]:
        notes.append(f"{v['external_refs']} external http ref(s) (not self-contained)")
    if r.get("render_note"):
        notes.append(r["render_note"])

    return {
        "candidate": cand["name"],
        "harness": meta.get("harness", "-"),
        "model": meta.get("effective_model") or meta.get("model") or "-",
        "mode": meta.get("mode", "-"),
        "valid": valid,
        "slide_count": v["slide_count"],
        "keywords": f"{v['keywords_found']}/{v['keywords_total']}",
        "console_errors": r["console_errors"] if r["rendered_ok"] else None,
        "external_refs": v["external_refs"],
        "wall_seconds": meta.get("wall_seconds"),
        "auto_score": auto_score,
        "screenshots": r["screenshot_paths"],
        "note": "; ".join(notes),
    }


def print_table(task: str, rows: list[dict]) -> None:
    hdr = (f"{'candidate':<12}{'harness':<14}{'model':<26}{'valid':<7}{'slides':<8}"
           f"{'keywords':<10}{'cons_err':<9}{'auto':<7}{'wall(s)':<9}")
    print("\n" + "=" * len(hdr))
    print(f"Task: {task}   auto_score = 0.4*keywords + 0.3*slide_count_ok + 0.3*no_console_errors")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ce = "-" if r["console_errors"] is None else str(r["console_errors"])
        wt = f"{r['wall_seconds']:.1f}" if isinstance(r["wall_seconds"], (int, float)) else "-"
        print(f"{r['candidate']:<12}{str(r['harness'])[:13]:<14}{str(r['model'])[:25]:<26}"
              f"{str(r['valid']):<7}{r['slide_count']:<8}{r['keywords']:<10}{ce:<9}"
              f"{r['auto_score']:<7}{wt:<9}")
    print("=" * len(hdr))
    for r in rows:
        if r["note"]:
            print(f"note[{r['candidate']}]: {r['note']}")


# ----------------------------------------------------------------- gallery ---
def build_gallery(task: str, rows: list[dict], out_path: Path) -> None:
    """Static human-review gallery: one card per candidate + client-side scoring/export."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for r in rows:
        # Everything below is agent-influenced (dir names, run_meta.json fields, error
        # text), so it is escaped before landing in the gallery markup.
        e = {k: html.escape(str(v), quote=True) for k, v in r.items() if k != "screenshots"}
        cand_url = f"../{quote(str(r['candidate']))}"
        thumbs = "".join(
            f'<a href="{cand_url}/screenshots/{quote(str(s))}" target="_blank">'
            f'<img src="{cand_url}/screenshots/{quote(str(s))}" loading="lazy"></a>'
            for s in r["screenshots"]
        ) or '<p class="muted">(no screenshot — render skipped or failed)</p>'
        note = f'<p class="note">{e["note"]}</p>' if r["note"] else ""
        cards.append(f"""
    <div class="card" data-candidate="{e['candidate']}">
      <h2>{e['candidate']} <span class="muted">/ {e['harness']} / {e['model']}</span></h2>
      <p class="metrics">valid={e['valid']} · slides={e['slide_count']} ·
         keywords={e['keywords']} · console_err={e['console_errors']} ·
         auto={e['auto_score']} · wall={e['wall_seconds']}s</p>
      {note}
      <div class="thumbs">{thumbs}</div>
      <p><a href="{cand_url}/slides.html" target="_blank">open raw slides.html →</a></p>
      <label>Human score (1-5):
        <select class="human-score"><option value="">—</option>
          <option>1</option><option>2</option><option>3</option>
          <option>4</option><option>5</option></select></label>
      <label>Notes: <input class="human-note" type="text" size="40"></label>
    </div>""")

    task_esc = html.escape(task, quote=True)
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{task_esc} — review gallery</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background:#fafafa; color:#222; }}
  h1 {{ margin-bottom: .25rem; }}
  .card {{ background:#fff; border:1px solid #ddd; border-radius:10px; padding:1rem 1.25rem;
           margin:1rem 0; box-shadow:0 1px 3px rgba(0,0,0,.06); }}
  .card h2 {{ margin:.2rem 0; font-size:1.15rem; }}
  .muted {{ color:#888; font-weight:normal; }}
  .metrics {{ color:#555; font-size:.9rem; }}
  .note {{ color:#b45; font-size:.85rem; }}
  .thumbs {{ display:flex; flex-wrap:wrap; gap:.5rem; margin:.5rem 0; }}
  .thumbs img {{ height:150px; border:1px solid #ccc; border-radius:4px; }}
  label {{ display:inline-block; margin:.4rem 1rem .2rem 0; font-size:.9rem; }}
  button {{ font-size:1rem; padding:.5rem 1rem; border-radius:8px; border:1px solid #0a7;
            background:#0a7; color:#fff; cursor:pointer; }}
</style></head>
<body>
<h1>{task_esc} — review gallery</h1>
<p class="muted">Score each deck 1-5, add notes, then download human_scores.json and run
<code>python grade_tasks.py --task {task_esc} --merge-human {task_esc}/gallery/human_scores.json</code>.</p>
<button onclick="dl()">⬇ download human_scores.json</button>
{''.join(cards)}
<script>
function dl() {{
  const out = {{}};
  document.querySelectorAll('.card').forEach(c => {{
    const s = c.querySelector('.human-score').value;
    const n = c.querySelector('.human-note').value;
    if (s || n) out[c.dataset.candidate] = {{ human_score: s ? Number(s) : null, human_note: n }};
  }});
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'human_scores.json'; a.click();
}}
</script>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    print(f"[grade] gallery -> {out_path}")


# ------------------------------------------------------------- merge-human ---
def merge_human(task: str, human_path: Path) -> None:
    """Join a downloaded human_scores.json onto the existing grade_results.json."""
    results_path = task_spec.task_dir(task) / "grade_results.json"
    if not results_path.exists():
        raise SystemExit(f"ERROR: {results_path} not found — run grading first.")
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    human = json.loads(human_path.read_text(encoding="utf-8"))
    for r in rows:
        h = human.get(r["candidate"], {})
        r["human_score"] = h.get("human_score")
        r["human_note"] = h.get("human_note", "")
    results_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    print(f"\n{'candidate':<16}{'auto':<8}{'human':<8}")
    print("-" * 32)
    for r in rows:
        print(f"{r['candidate']:<16}{r['auto_score']:<8}{str(r.get('human_score', '-')):<8}")
    print(f"\n[grade] merged human scores -> {results_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade a task's candidate slide decks.")
    ap.add_argument("--task", default=task_spec.DEFAULT_TASK,
                    help=f"task id / directory (default: {task_spec.DEFAULT_TASK})")
    ap.add_argument("--candidates", nargs="*", default=None,
                    help="specific candidate names (default: all under the task dir)")
    ap.add_argument("--no-render", action="store_true",
                    help="skip Playwright rendering (validation only)")
    ap.add_argument("--merge-human", type=Path, default=None,
                    help="merge a downloaded human_scores.json into grade_results.json")
    ap.add_argument("--allow-render-network", action="store_true",
                    help="let the rendered deck make network requests (off by default: "
                         "decks are untrusted model output and must be self-contained)")
    args = ap.parse_args()

    task_spec.validate_name(args.task, "task")
    for c in args.candidates or []:
        task_spec.validate_name(c, "candidate")

    if args.merge_human is not None:
        merge_human(args.task, args.merge_human)
        return

    cfg = task_spec.load_task(args.task)
    cands = discover_candidates(args.task, args.candidates)
    if not cands:
        ap.error(f"no candidates found under {args.task}/ — run run_task.py first")

    rows = [grade_one(c, cfg, do_render=not args.no_render,
                      allow_network=args.allow_render_network) for c in cands]
    print_table(args.task, rows)

    tdir = task_spec.task_dir(args.task)
    out = tdir / "grade_results.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\n[grade] wrote {out}")

    build_gallery(args.task, rows, tdir / "gallery" / "index.html")


if __name__ == "__main__":
    main()
