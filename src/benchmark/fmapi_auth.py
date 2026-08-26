#!/usr/bin/env python3
"""
fmapi_auth.py — Resolve Databricks FMAPI host + bearer token, preferring `ucode`.

`ucode` (the Databricks AI Gateway CLI) already holds a workspace host and mints
short-lived bearer tokens via `ucode auth-token`. Rather than make the user export
DATABRICKS_HOST / DATABRICKS_TOKEN by hand, this module reads ucode's config
(~/.codex/ucode.config.toml) for the host and shells out to `ucode auth-token` for a
fresh token on every run — so a 15-minute token expiry never bites (each run re-mints).

Resolution order (host and token resolved independently):
  1. Explicit env vars DATABRICKS_HOST / DATABRICKS_TOKEN (always win — an escape hatch).
  2. ucode: host from ucode.config.toml, token from `ucode auth-token --host ... --profile ...`.

The FMAPI base_url is  {host}/serving-endpoints  (OpenAI-compatible chat completions),
which is a DIFFERENT path from ucode's own /ai-gateway/codex/v1 endpoint — but the same
workspace bearer token authorizes both.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib

UCODE_CONFIG = Path.home() / ".codex" / "ucode.config.toml"
UCODE_PROVIDER = "ucode-databricks"  # [model_providers.<name>] block in the config


def _ucode_provider_cfg() -> dict:
    """Parse ~/.codex/ucode.config.toml and return the ucode-databricks provider block."""
    if not UCODE_CONFIG.exists():
        return {}
    try:
        cfg = tomllib.loads(UCODE_CONFIG.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(
            f"WARNING: ucode config {UCODE_CONFIG}를 읽을 수 없음 "
            f"({type(e).__name__}: {e})",
            file=sys.stderr,
        )
        return {}
    return cfg.get("model_providers", {}).get(UCODE_PROVIDER, {})


def _host_from_ucode() -> str | None:
    """Workspace host from ucode config: prefer the auth command's --host arg (the bare
    workspace URL), falling back to trimming the /ai-gateway/... suffix off base_url."""
    prov = _ucode_provider_cfg()
    args = prov.get("auth", {}).get("args", [])
    if "--host" in args:
        i = args.index("--host")
        if i + 1 < len(args):
            return args[i + 1].rstrip("/")
    base = prov.get("base_url")
    if base:
        # e.g. https://dbc-...cloud.databricks.com/ai-gateway/codex/v1 -> https://dbc-...
        return base.split("/ai-gateway/")[0].rstrip("/")
    return None


def _token_from_ucode(host: str) -> str | None:
    """Mint a fresh bearer token via `ucode auth-token`, honoring the profile in the
    config's auth args (falls back to no explicit profile)."""
    ucode = shutil.which("ucode") or str(Path.home() / ".local" / "bin" / "ucode")
    if not Path(ucode).exists():
        return None
    prov_args = _ucode_provider_cfg().get("auth", {}).get("args", [])
    profile = None
    if "--profile" in prov_args:
        j = prov_args.index("--profile")
        if j + 1 < len(prov_args):
            profile = prov_args[j + 1]
    argv = [ucode, "auth-token", "--host", host]
    if profile:
        argv += ["--profile", profile]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "")[:200]
        print(
            f"WARNING: ucode auth-token 실행 실패 "
            f"({type(e).__name__}: {e}; stderr: {stderr})",
            file=sys.stderr,
        )
        return None
    except subprocess.TimeoutExpired as e:
        print(
            f"WARNING: ucode auth-token 시간 초과 "
            f"({type(e).__name__}: {e})",
            file=sys.stderr,
        )
        return None
    except OSError as e:
        print(
            f"WARNING: ucode auth-token 실행 실패 "
            f"({type(e).__name__}: {e})",
            file=sys.stderr,
        )
        return None
    token = out.stdout.strip()
    return token or None


def resolve_host_token() -> tuple[str | None, str | None, str]:
    """Return (host, token, source). Env vars win; else fall back to ucode.

    `source` is a short human-readable string for logging ('env', 'ucode', 'mixed', 'none')."""
    env_host = os.getenv("DATABRICKS_HOST")
    env_token = os.getenv("DATABRICKS_TOKEN")

    host = env_host or _host_from_ucode()
    token = env_token
    if token is None and host:
        token = _token_from_ucode(host)

    if env_host and env_token:
        source = "env"
    elif host and token and not (env_host and env_token):
        source = "ucode" if not (env_host or env_token) else "mixed"
    else:
        source = "none"
    return (host.rstrip("/") if host else None), token, source
