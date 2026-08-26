"""Databricks FMAPI serving-endpoint 어댑터.

여러 모델 계열(claude/openai/glm)을 단일 인터페이스로 추상화한다. 아래 동작은
모두 ai_devtools 워크스페이스에서 실측 검증된 사실을 반영한다 (plan §11, 부록):

- 응답 `content`가 계열마다 다름: opus-5·gemini는 리스트(reasoning 블록 포함),
  sol·glm은 문자열. glm은 답을 `reasoning_content`에 먼저 쓴다. → `_normalize_content`.
- reasoning 제어 파라미터가 계열마다 다름 → config의 minimal/full 딕셔너리를 그대로 병합.
- request_id를 응답에서 뽑아 두면 나중에 system.ai_gateway.usage와 조인해 시간·비용 산출.
- glm은 3~5배 느리므로 넉넉한 timeout + 지수 backoff 재시도.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


class FMAPIError(RuntimeError):
    """복구 불가능한 FMAPI 호출 실패."""


@dataclass
class ChatResponse:
    """정규화된 채팅 응답. 채점기·비용 계산이 이 형태만 알면 된다."""

    text: str                       # 평문 응답 (reasoning 블록·구조 제거 후)
    request_id: str | None          # ai_gateway.usage 조인 키
    finish_reason: str | None
    usage: dict[str, Any]           # prompt/completion/total tokens 등 원본
    raw: dict[str, Any] = field(repr=False, default_factory=dict)  # 감사용 원본 보존


def _get_workspace_auth(profile: str) -> tuple[str, str]:
    """databricks CLI로 host와 bearer token을 얻는다.

    CLI는 토큰 갱신을 알아서 처리하므로 매 클라이언트 생성 시 호출한다.
    """
    import json

    def _cli(*args: str) -> str:
        try:
            proc = subprocess.run(
                ["databricks", *args, "--profile", profile],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as e:
            raise FMAPIError(f"databricks CLI 실행 실패: {type(e).__name__}: {e}") from e
        if proc.returncode != 0:
            raise FMAPIError(f"databricks CLI 실패: {' '.join(args)}\n{proc.stderr.strip()}")
        return proc.stdout

    try:
        host = json.loads(_cli("auth", "env"))["env"]["DATABRICKS_HOST"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise FMAPIError(
            f"databricks CLI host 응답 파싱 실패: {type(e).__name__}: {e}"
        ) from e
    try:
        token = json.loads(_cli("auth", "token"))["access_token"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise FMAPIError(
            f"databricks CLI token 응답 파싱 실패: {type(e).__name__}: {e}"
        ) from e
    return host.rstrip("/"), token


def _normalize_content(message: dict[str, Any]) -> str:
    """계열별로 다른 message 형태를 평문 텍스트로 정규화한다 (실측 기반).

    - content가 문자열(sol): 그대로.
    - content가 리스트(opus-5·gemini): type=="text" 파트만 이어붙이고 reasoning 파트는 버림.
    - content가 비었고 reasoning_content만 있음(glm이 사고 도중 잘림): reasoning_content로 폴백.
    """
    content = message.get("content")

    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        text = "".join(parts).strip()
        if text:
            return text
        # 리스트에 text 파트가 없으면(전부 reasoning) reasoning_content 폴백
        return (message.get("reasoning_content") or "").strip()

    if isinstance(content, str) and content.strip():
        return content.strip()

    # content가 비었으면 glm처럼 reasoning_content에 답이 남아있을 수 있음
    return (message.get("reasoning_content") or "").strip()


class FMAPIClient:
    """단일 워크스페이스에 대한 FMAPI 호출 클라이언트."""

    def __init__(
        self,
        profile: str,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        backoff_initial_seconds: float = 0.5,
    ) -> None:
        self.profile = profile
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_initial_seconds = backoff_initial_seconds
        self._host, self._token = _get_workspace_auth(profile)
        self._client = httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FMAPIClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat(
        self,
        endpoint: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
        extra_params: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """chat completion 호출 후 정규화된 응답을 돌려준다.

        extra_params에 reasoning 제어 딕셔너리(config의 minimal/full)를 그대로 넘긴다.
        """
        url = f"{self._host}/serving-endpoints/{endpoint}/invocations"
        payload: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens}
        if extra_params:
            payload.update(extra_params)
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(url, json=payload, headers=headers)
            except httpx.RequestError as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    self._sleep_backoff(attempt)
                continue

            # 5xx·429는 재시도, 그 외 4xx는 즉시 실패(요청 자체가 잘못)
            if resp.status_code >= 500 or resp.status_code == 429:
                last_err = FMAPIError(f"{endpoint} HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < self.max_retries - 1:
                    self._sleep_backoff(attempt)
                continue
            if resp.status_code != 200:
                raise FMAPIError(f"{endpoint} HTTP {resp.status_code}: {resp.text[:300]}")

            try:
                data = resp.json()
            except ValueError as e:
                raise FMAPIError(
                    f"{endpoint} 응답이 JSON이 아님: {resp.text[:300]}"
                ) from e
            return self._parse(data, resp)

        raise FMAPIError(
            f"{endpoint} 재시도 {self.max_retries}회 모두 실패: {last_err}"
        ) from last_err

    def _parse(self, data: dict[str, Any], resp: httpx.Response) -> ChatResponse:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError) as e:
            raise FMAPIError(f"예상치 못한 응답 형태: {str(data)[:300]}") from e
        message = choice.get("message", {})
        # request_id: 응답 헤더(x-request-id) 우선, 없으면 body의 id
        request_id = resp.headers.get("x-request-id") or data.get("id")
        return ChatResponse(
            text=_normalize_content(message),
            request_id=request_id,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage", {}) or {},
            raw=data,
        )

    def _sleep_backoff(self, attempt: int) -> None:
        time.sleep(self.backoff_initial_seconds * (2**attempt))


def build_text_message(prompt: str) -> list[dict[str, Any]]:
    """텍스트 전용 메시지."""
    return [{"role": "user", "content": prompt}]


def build_image_message(prompt: str, image_data_url: str) -> list[dict[str, Any]]:
    """이미지+텍스트 멀티모달 메시지 (vision 지원 모델용)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]
