"""benchmark.fmapi_auth — host/token resolution (env wins, else ucode)."""

import subprocess

import pytest

from benchmark import fmapi_auth

CONFIG = """
[model_providers.ucode-databricks]
base_url = "https://dbc-demo.cloud.databricks.com/ai-gateway/codex/v1"

[model_providers.ucode-databricks.auth]
command = "ucode"
args = [
  "auth-token", "--host", "https://dbc-demo.cloud.databricks.com/",
  "--profile", "ai_devtools",
]
"""

CONFIG_NO_HOST_ARG = """
[model_providers.ucode-databricks]
base_url = "https://dbc-demo.cloud.databricks.com/ai-gateway/codex/v1"
"""


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)


@pytest.fixture
def ucode_config(tmp_path, monkeypatch):
    """Write a ucode config and point the module at it. Returns a writer callable."""
    path = tmp_path / "ucode.config.toml"

    def write(body: str | None):
        if body is None:
            if path.exists():
                path.unlink()
        else:
            path.write_text(body, encoding="utf-8")
        return path

    monkeypatch.setattr(fmapi_auth, "UCODE_CONFIG", path)
    return write


@pytest.fixture
def fake_ucode(tmp_path, monkeypatch):
    """A ucode executable on PATH whose `auth-token` output is controllable."""
    exe = tmp_path / "ucode"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(fmapi_auth.shutil, "which", lambda _name: str(exe))
    calls: list[list[str]] = []

    def install(stdout="tok-123", exc=None):
        def fake_run(argv, **kwargs):
            calls.append(argv)
            if exc is not None:
                raise exc
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(fmapi_auth.subprocess, "run", fake_run)
        return calls

    return install


def test_provider_cfg_missing_file_is_empty(ucode_config):
    ucode_config(None)
    assert fmapi_auth._ucode_provider_cfg() == {}


def test_provider_cfg_malformed_toml_is_empty(ucode_config):
    ucode_config("this is not = valid = toml [[[")
    assert fmapi_auth._ucode_provider_cfg() == {}


def test_provider_cfg_missing_provider_block_is_empty(ucode_config):
    ucode_config('[model_providers.other]\nbase_url = "https://x"\n')
    assert fmapi_auth._ucode_provider_cfg() == {}


def test_host_prefers_auth_host_arg_and_strips_slash(ucode_config):
    ucode_config(CONFIG)
    assert fmapi_auth._host_from_ucode() == "https://dbc-demo.cloud.databricks.com"


def test_host_falls_back_to_trimmed_base_url(ucode_config):
    ucode_config(CONFIG_NO_HOST_ARG)
    assert fmapi_auth._host_from_ucode() == "https://dbc-demo.cloud.databricks.com"


def test_host_none_without_config(ucode_config):
    ucode_config(None)
    assert fmapi_auth._host_from_ucode() is None


def test_token_from_ucode_passes_host_and_profile(ucode_config, fake_ucode):
    ucode_config(CONFIG)
    calls = fake_ucode(stdout="  tok-123\n")
    assert fmapi_auth._token_from_ucode("https://h") == "tok-123"
    argv = calls[0]
    assert argv[1:] == ["auth-token", "--host", "https://h", "--profile", "ai_devtools"]


def test_token_from_ucode_omits_profile_when_absent(ucode_config, fake_ucode):
    ucode_config(CONFIG_NO_HOST_ARG)
    calls = fake_ucode()
    assert fmapi_auth._token_from_ucode("https://h") == "tok-123"
    assert "--profile" not in calls[0]


def test_token_none_when_ucode_missing(ucode_config, monkeypatch):
    ucode_config(CONFIG)
    monkeypatch.setattr(fmapi_auth.shutil, "which", lambda _name: None)
    monkeypatch.setattr(fmapi_auth.Path, "home",
                        staticmethod(lambda: fmapi_auth.Path("/nonexistent")))
    assert fmapi_auth._token_from_ucode("https://h") is None


@pytest.mark.parametrize("exc", [
    subprocess.CalledProcessError(1, "ucode"),
    subprocess.TimeoutExpired("ucode", 15),
    OSError("boom"),
])
def test_token_none_when_ucode_fails(ucode_config, fake_ucode, exc):
    ucode_config(CONFIG)
    fake_ucode(exc=exc)
    assert fmapi_auth._token_from_ucode("https://h") is None


def test_token_none_when_output_blank(ucode_config, fake_ucode):
    ucode_config(CONFIG)
    fake_ucode(stdout="  \n")
    assert fmapi_auth._token_from_ucode("https://h") is None


def test_resolve_env_wins_and_never_shells_out(monkeypatch, ucode_config):
    ucode_config(CONFIG)
    monkeypatch.setenv("DATABRICKS_HOST", "https://env-host/")
    monkeypatch.setenv("DATABRICKS_TOKEN", "env-token")
    monkeypatch.setattr(fmapi_auth, "_token_from_ucode",
                        lambda _h: pytest.fail("ucode must not be consulted"))
    assert fmapi_auth.resolve_host_token() == ("https://env-host", "env-token", "env")


def test_resolve_from_ucode(ucode_config, fake_ucode):
    ucode_config(CONFIG)
    fake_ucode()
    host, token, source = fmapi_auth.resolve_host_token()
    assert (host, token, source) == ("https://dbc-demo.cloud.databricks.com", "tok-123", "ucode")


def test_resolve_mixed_env_host_with_ucode_token(monkeypatch, ucode_config, fake_ucode):
    ucode_config(CONFIG)
    fake_ucode()
    monkeypatch.setenv("DATABRICKS_HOST", "https://env-host")
    host, token, source = fmapi_auth.resolve_host_token()
    assert (host, token, source) == ("https://env-host", "tok-123", "mixed")


def test_resolve_none_when_nothing_available(ucode_config, monkeypatch):
    ucode_config(None)
    monkeypatch.setattr(fmapi_auth, "_token_from_ucode", lambda _h: None)
    assert fmapi_auth.resolve_host_token() == (None, None, "none")


def test_resolve_none_when_host_found_but_token_missing(ucode_config, monkeypatch):
    ucode_config(CONFIG)
    monkeypatch.setattr(fmapi_auth, "_token_from_ucode", lambda _h: None)
    host, token, source = fmapi_auth.resolve_host_token()
    assert token is None and source == "none"
    assert host == "https://dbc-demo.cloud.databricks.com"
