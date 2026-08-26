#!/usr/bin/env python3
"""
task_spec.py — SINGLE SOURCE OF TRUTH for a benchmark task.

Layout is task-centric: each task lives in its own directory under the benchmark root:

    <task_id>/
    ├── README.md            # task overview
    ├── TASK_DESCRIPTION.md   # canonical brief + hard format contract (the instructions)
    ├── keywords.json         # machine grading config (slide bounds + required topics)
    └── <candidate>/          # one output dir per compared model/candidate (opus, sol, glm, …)

This package is reusable across tasks and may be pip-installed, so task directories are
resolved relative to the BENCHMARK ROOT — the current working directory by default (run
the CLIs from the repo root), overridable with the BENCHMARK_ROOT env var. It is NOT
resolved from this file's install location.

Both run_task.py (the runner) and grade_tasks.py (the grader) import from here so the
prompt, the task description, and the grading contract can NEVER drift apart. This mirrors
the fairness rule from the sibling agent-ml/ benchmark: everything except "the step that
produces the artifact" is byte-identical across every candidate.

Nothing in this module has side effects — it is pure constants + functions.
"""
import json
import os
import re
from pathlib import Path

DEFAULT_TASK = "explain-databricks"
ARTIFACT = "slides.html"

# Task and candidate names become directory names under the benchmark root, and the
# runner deletes/recreates <task>/<candidate>/ — so they must stay single path segments.
NAME_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")

# Default candidate -> FMAPI model mapping. The candidate name (opus/sol/glm) is a stable
# label; the model it points at can be refreshed here. Override per-run with --model.
# These are FMAPI serving-endpoint names (must start with 'databricks-').
CANDIDATE_MODELS = {
    "opus": "databricks-claude-opus-4-8",
    "sol": "databricks-gpt-5-6-sol",
    "glm": "databricks-glm-5-2",
}

# For the `pi` agent harness (Pi driven via ucode's Databricks AI Gateway config), the
# candidate maps to a (provider, model) pair. pi reaches the gateway, not the serving-
# endpoint path, so model ids are the gateway `system.ai.*` names — different from
# CANDIDATE_MODELS above. Only candidates with a natural gateway equivalent get a default;
# anything else must pass --model (+ optionally --pi-provider). ucode's pi gateway offers
# databricks-claude / databricks-openai / databricks-gemini (no GLM), so glm has no default.
PI_CANDIDATE_MODELS = {
    "opus": ("databricks-claude", "system.ai.claude-opus-5"),
    "sol": ("databricks-openai", "system.ai.gpt-5-5"),
}

# ---- The COMMON PROMPT — MUST stay byte-identical for every candidate. ------
# Kept short on the CLI; the heavy task detail lives in TASK_DESCRIPTION.md (written
# into each candidate dir as instructions.txt), which the agent is told to read.
# direct-fmapi concatenates COMMON_PROMPT + the full instructions text into one message,
# so a non-agent model gets the same brief.
COMMON_PROMPT = (
    "Read ./instructions.txt and write a slide deck to ./slides.html that "
    "explains what Databricks is. Produce ONE self-contained HTML file: all CSS "
    "inline in a <style> tag (or an allowed reveal.js CDN), no local build step, "
    "and it must render when opened directly via file:// with no network access "
    "required for the core content. Follow the exact format contract, slide "
    "count, and required topics described in instructions.txt. Iterate until "
    "./slides.html exists and is a valid, complete, self-contained deck.\n\n"
    "Output requirement: respond with ONLY the complete HTML document — start at "
    "`<!DOCTYPE html>` and end at `</html>`, with NO explanation, reasoning, "
    "preamble, or markdown code fences before or after it."
)


def validate_name(name: str, kind: str = "name") -> str:
    """Reject anything that is not a plain single path segment (no separators, no '..')."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise SystemExit(
            f"ERROR: invalid {kind} {name!r} — use letters, digits, '.', '_', '-' only "
            "(a single directory name, no path separators)"
        )
    return name


def benchmark_root() -> Path:
    """Root directory that holds the per-task directories.

    Defaults to the current working directory (run the CLIs from the repo root);
    override with the BENCHMARK_ROOT env var."""
    return Path(os.environ.get("BENCHMARK_ROOT", Path.cwd())).resolve()


def task_dir(task: str = DEFAULT_TASK) -> Path:
    """The directory holding a task's description, config, and candidate outputs."""
    return benchmark_root() / validate_name(task, "task")


def load_task(task: str = DEFAULT_TASK) -> dict:
    """Load the machine-readable task config (keywords + slide-count bounds)."""
    kw_path = task_dir(task) / "keywords.json"
    if not kw_path.exists():
        raise FileNotFoundError(
            f"task config missing: {kw_path} — expected {task}/keywords.json "
            f"under the benchmark root ({benchmark_root()}). Run from the repo root "
            f"or set BENCHMARK_ROOT."
        )
    return json.loads(kw_path.read_text(encoding="utf-8"))


def load_description(task: str = DEFAULT_TASK) -> str:
    """Load the canonical task description (TASK_DESCRIPTION.md). This is copied
    verbatim into each candidate's instructions.txt so every candidate reads the
    exact same brief."""
    desc_path = task_dir(task) / "TASK_DESCRIPTION.md"
    if not desc_path.exists():
        raise FileNotFoundError(
            f"task description missing: {desc_path} — expected {task}/TASK_DESCRIPTION.md "
            f"under the benchmark root ({benchmark_root()}). Run from the repo root "
            f"or set BENCHMARK_ROOT."
        )
    return desc_path.read_text(encoding="utf-8")
