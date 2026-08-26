"""benchmark.run_task — workdir prep, harness argv, html extraction, meta."""

import json
import subprocess

import pytest

from benchmark import run_task, task_spec

from .conftest import DESCRIPTION


# ------------------------------------------------------------------ workdir ---
def test_candidate_dir_under_task_dir(bench_root):
    expected = bench_root.resolve() / "demo-task" / "opus"
    assert run_task.candidate_dir("demo-task", "opus") == expected


def test_prepare_workdir_writes_identical_prompt_and_instructions(bench_root):
    a = run_task.prepare_workdir("demo-task", "opus")
    b = run_task.prepare_workdir("demo-task", "glm")

    assert (a / "instructions.txt").read_text(encoding="utf-8") == DESCRIPTION
    # the fairness rule: byte-identical inputs for every candidate
    assert (a / "prompt.txt").read_bytes() == (b / "prompt.txt").read_bytes()
    assert (a / "prompt.txt").read_bytes() == task_spec.COMMON_PROMPT.encode("utf-8")
    assert (a / "instructions.txt").read_bytes() == (b / "instructions.txt").read_bytes()


def test_prepare_workdir_wipes_previous_run(bench_root):
    workdir = run_task.prepare_workdir("demo-task", "opus")
    (workdir / "slides.html").write_text("stale", encoding="utf-8")
    again = run_task.prepare_workdir("demo-task", "opus")
    assert not (again / "slides.html").exists()


# --------------------------------------------------------------------- argv ---
def test_build_argv_claude_code_carries_prompt_and_turn_cap():
    argv = run_task.build_argv("claude-code", None, "claude", 7)
    assert argv[:2] == ["claude", "-p"]
    assert argv[2] == task_spec.COMMON_PROMPT
    assert argv[-2:] == ["--max-turns", "7"]


def test_build_argv_codex_is_non_interactive():
    argv = run_task.build_argv("codex", "databricks-x", "claude", 40)
    assert argv[:2] == ["codex", "exec"]
    assert argv[-1] == task_spec.COMMON_PROMPT
    assert "--skip-git-repo-check" in argv


def test_build_argv_pi_includes_provider_and_model():
    argv = run_task.build_argv("pi", "system.ai.claude-opus-5", "claude", 40,
                               pi_provider="databricks-claude")
    assert argv[:3] == ["pi", "--print", "--no-session"]
    assert argv[3:7] == ["--provider", "databricks-claude", "--model", "system.ai.claude-opus-5"]
    assert argv[-1] == task_spec.COMMON_PROMPT


def test_build_argv_pi_omits_missing_provider_and_model():
    argv = run_task.build_argv("pi", None, "claude", 40)
    assert argv == ["pi", "--print", "--no-session", task_spec.COMMON_PROMPT]


def test_build_argv_omnigent_not_wired_yet():
    with pytest.raises(NotImplementedError):
        run_task.build_argv("omnigent", "databricks-x", "claude", 40)


def test_build_argv_unknown_harness_exits():
    with pytest.raises(SystemExit):
        run_task.build_argv("direct-fmapi", "databricks-x", "claude", 40)


# ------------------------------------------------------------------- pi_env ---
def test_pi_env_points_at_ucode_agent_dir(tmp_path, monkeypatch):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "models.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_task, "UCODE_PI_AGENT_DIR", agent_dir)
    assert run_task.pi_env() == {"PI_CODING_AGENT_DIR": str(agent_dir)}


def test_pi_env_exits_when_pi_unconfigured(tmp_path, monkeypatch):
    monkeypatch.setattr(run_task, "UCODE_PI_AGENT_DIR", tmp_path / "missing")
    with pytest.raises(SystemExit) as e:
        run_task.pi_env()
    assert "ucode configure --agent pi" in str(e.value)


# ------------------------------------------------------------- extract_html ---
DOC = "<!DOCTYPE html><html><body>hi</body></html>"


def test_extract_html_empty():
    assert run_task.extract_html("") == ""


def test_extract_html_strips_prose_around_document():
    assert run_task.extract_html(f"Sure! Here you go:\n{DOC}\nHope that helps.") == DOC


def test_extract_html_strips_markdown_fence():
    assert run_task.extract_html(f"```html\n{DOC}\n```") == DOC


def test_extract_html_without_doctype_starts_at_html_tag():
    text = "preamble <html><body>x</body></html> trailer"
    assert run_task.extract_html(text) == "<html><body>x</body></html>"


def test_extract_html_uses_last_closing_tag():
    out = run_task.extract_html(f"{DOC} and also </html>")
    assert out.endswith("</html>")
    assert out.count("</html>") == 2


def test_extract_html_fence_fallback_when_no_html_tags():
    assert run_task.extract_html("blah ```\n<div>x</div>\n``` blah") == "<div>x</div>"


def test_extract_html_returns_stripped_text_as_last_resort():
    assert run_task.extract_html("  I refuse.  ") == "I refuse."


def test_extract_html_ignores_unclosed_document():
    """No </html> at all → falls through to the text-as-is branch."""
    assert run_task.extract_html("<!DOCTYPE html><html><body>truncated") == \
        "<!DOCTYPE html><html><body>truncated"


# --------------------------------------------------------------------- meta ---
def test_base_meta_schema(tmp_path):
    meta = run_task.base_meta("demo-task", "opus", "codex", "databricks-x", tmp_path,
                              ["codex", "exec"], "headless")
    assert meta["task"] == "demo-task"
    assert meta["candidate"] == "opus"
    assert meta["mode"] == "headless"
    assert meta["common_prompt"] == task_spec.COMMON_PROMPT
    assert meta["artifact_path"] == str(tmp_path / task_spec.ARTIFACT)
    assert meta["artifact_exists"] is False
    assert meta["effective_model"] is None
    assert meta["wall_seconds"] == 0.0
    assert meta["timed_out"] is False


def test_write_meta_serializes_run_meta_json(tmp_path, capsys):
    meta = run_task.base_meta("demo-task", "opus", "codex", None, tmp_path, None, "headless")
    run_task.write_meta(tmp_path, meta)
    written = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert written == meta
    assert "run_meta.json" in capsys.readouterr().out


# --------------------------------------------------------------- subprocess ---
def _run_subprocess(tmp_path, monkeypatch, fake_run):
    monkeypatch.setattr(run_task.subprocess, "run", fake_run)
    return run_task.run_subprocess("demo-task", "opus", "codex", "databricks-x", tmp_path,
                                   ["codex", "exec"], 5, 40)


def test_run_subprocess_records_exit_code_and_artifact(tmp_path, monkeypatch):
    def fake_run(argv, **kwargs):
        (tmp_path / task_spec.ARTIFACT).write_text("<html></html>", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    meta = _run_subprocess(tmp_path, monkeypatch, fake_run)
    assert meta["exit_code"] == 0
    assert meta["artifact_exists"] is True
    assert meta["timed_out"] is False
    assert (tmp_path / "agent_output.log").exists()


def test_run_subprocess_flags_timeout(tmp_path, monkeypatch):
    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 5)

    meta = _run_subprocess(tmp_path, monkeypatch, fake_run)
    assert meta["timed_out"] is True
    assert meta["exit_code"] is None
    assert meta["artifact_exists"] is False


def test_run_subprocess_notes_missing_cli(tmp_path, monkeypatch):
    def fake_run(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    meta = _run_subprocess(tmp_path, monkeypatch, fake_run)
    assert "CLI not found" in meta["note"]


def test_run_subprocess_merges_env_overlay(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(run_task.subprocess, "run", fake_run)
    monkeypatch.setenv("EXISTING_VAR", "keep-me")
    run_task.run_subprocess("demo-task", "opus", "pi", None, tmp_path, ["pi"], 5, 40,
                            env_overlay={"PI_CODING_AGENT_DIR": "/x"})
    assert seen["PI_CODING_AGENT_DIR"] == "/x"
    assert seen["EXISTING_VAR"] == "keep-me"


# ------------------------------------------------------------- direct-fmapi ---
class _FakeUsage:
    prompt_tokens = 11
    completion_tokens = 22


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = type("M", (), {"content": content})()
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [_FakeChoice(content, finish_reason)]
        self.usage = _FakeUsage()


def _install_fake_openai(monkeypatch, response=None, exc=None):
    """Install a stub `openai` module; run_direct_fmapi imports it lazily."""
    import sys
    import types

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            if exc is not None:
                raise exc
            return response

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = type("C", (), {"completions": FakeCompletions()})()

    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return captured


@pytest.fixture
def fmapi_workdir(bench_root, monkeypatch):
    monkeypatch.setattr(run_task.fmapi_auth, "resolve_host_token",
                        lambda: ("https://host", "tok", "env"))
    return run_task.prepare_workdir("demo-task", "opus")


def test_run_direct_fmapi_writes_artifact_and_usage(fmapi_workdir, monkeypatch):
    captured = _install_fake_openai(monkeypatch, _FakeResponse(f"here:\n{DOC}"))
    meta = run_task.run_direct_fmapi("demo-task", "opus", "direct-fmapi",
                                     "databricks-x", fmapi_workdir, 120)

    assert (fmapi_workdir / task_spec.ARTIFACT).read_text(encoding="utf-8") == DOC
    assert meta["exit_code"] == 0
    assert meta["artifact_exists"] is True
    assert (meta["prompt_tokens"], meta["completion_tokens"]) == (11, 22)
    assert meta["effective_model"] == "databricks-x"
    # the single-shot message is COMMON_PROMPT + the verbatim instructions
    sent = captured["messages"][0]["content"]
    assert sent.startswith(task_spec.COMMON_PROMPT)
    assert sent.endswith(DESCRIPTION)
    assert captured["client_kwargs"]["base_url"] == "https://host/serving-endpoints"


def test_run_direct_fmapi_warns_on_truncated_completion(fmapi_workdir, monkeypatch):
    _install_fake_openai(monkeypatch, _FakeResponse(DOC, finish_reason="length"))
    meta = run_task.run_direct_fmapi("demo-task", "opus", "direct-fmapi",
                                     "databricks-x", fmapi_workdir, 120)
    assert "finish_reason=length" in meta["note"]


def test_run_direct_fmapi_records_call_failure_without_crashing(fmapi_workdir, monkeypatch):
    _install_fake_openai(monkeypatch, exc=RuntimeError("gateway down"))
    meta = run_task.run_direct_fmapi("demo-task", "opus", "direct-fmapi",
                                     "databricks-x", fmapi_workdir, 120)
    assert "FMAPI call failed (RuntimeError: gateway down)" in meta["note"]
    assert meta["exit_code"] is None
    assert meta["artifact_exists"] is False


@pytest.mark.parametrize("model", [None, "claude-opus-4-8", "gpt-5"])
def test_run_direct_fmapi_rejects_non_fmapi_model(fmapi_workdir, model):
    with pytest.raises(SystemExit) as e:
        run_task.run_direct_fmapi("demo-task", "opus", "direct-fmapi", model, fmapi_workdir, 120)
    assert "databricks-" in str(e.value)


def test_run_direct_fmapi_exits_without_credentials(fmapi_workdir, monkeypatch):
    monkeypatch.setattr(run_task.fmapi_auth, "resolve_host_token", lambda: (None, None, "none"))
    with pytest.raises(SystemExit) as e:
        run_task.run_direct_fmapi("demo-task", "opus", "direct-fmapi", "databricks-x",
                                  fmapi_workdir, 120)
    assert "credentials" in str(e.value)


# --------------------------------------------------------------------- main ---
def _main(monkeypatch, argv):
    monkeypatch.setattr(run_task.sys, "argv", ["run-task", *argv])
    run_task.main()


def test_main_defaults_model_from_candidate(bench_root, monkeypatch):
    seen = {}

    def fake_direct(task, candidate, harness, model, workdir, max_seconds):
        seen["model"] = model
        return run_task.base_meta(task, candidate, harness, model, workdir, None, "direct-fmapi")

    monkeypatch.setattr(run_task, "run_direct_fmapi", fake_direct)
    _main(monkeypatch, ["--task", "demo-task", "--candidate", "opus", "--harness", "direct-fmapi"])
    assert seen["model"] == task_spec.CANDIDATE_MODELS["opus"]


def test_main_defaults_pi_provider_and_model_from_candidate(bench_root, monkeypatch):
    seen = {}

    def fake_subprocess(task, candidate, harness, model, workdir, argv, max_seconds,
                        max_turns, env_overlay=None):
        seen["argv"] = argv
        return run_task.base_meta(task, candidate, harness, model, workdir, argv, "headless")

    monkeypatch.setattr(run_task, "run_subprocess", fake_subprocess)
    monkeypatch.setattr(run_task, "pi_env", lambda: {"PI_CODING_AGENT_DIR": "/x"})
    _main(monkeypatch, ["--task", "demo-task", "--candidate", "opus", "--harness", "pi"])

    provider, model = task_spec.PI_CANDIDATE_MODELS["opus"]
    assert provider in seen["argv"] and model in seen["argv"]
    meta = json.loads((run_task.candidate_dir("demo-task", "opus") / "run_meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["effective_model"] == f"{provider}/{model}"


def test_main_rejects_unknown_harness(bench_root, monkeypatch):
    with pytest.raises(SystemExit):
        _main(monkeypatch, ["--task", "demo-task", "--candidate", "opus", "--harness", "nope"])


def test_main_allows_unknown_harness_with_manual(bench_root, monkeypatch):
    monkeypatch.setattr(run_task, "run_manual",
                        lambda task, cand, harness, model, workdir:
                        run_task.base_meta(task, cand, harness, model, workdir, None, "manual"))
    _main(monkeypatch, ["--task", "demo-task", "--candidate", "ui", "--harness", "playground",
                        "--manual"])
    assert (run_task.candidate_dir("demo-task", "ui") / "run_meta.json").exists()


def test_main_rejects_direct_fmapi_without_databricks_model(bench_root, monkeypatch):
    with pytest.raises(SystemExit):
        _main(monkeypatch, ["--task", "demo-task", "--candidate", "unknown-candidate",
                            "--harness", "direct-fmapi"])


def test_main_rejects_pi_without_defaults(bench_root, monkeypatch):
    with pytest.raises(SystemExit):
        _main(monkeypatch, ["--task", "demo-task", "--candidate", "glm", "--harness", "pi"])
