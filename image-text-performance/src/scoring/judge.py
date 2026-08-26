"""
LLM-as-judge scoring orchestration.

This module handles LLM judge inference and score parsing. The judge (Databricks
Gemini-3.1-Pro) evaluates free-form generation tasks using structured rubrics
and produces 1–5 scores.

Key components:
- parse_judge_score: Robustly extract a 1–5 integer score from judge text.
  Handles the Gemini list-response issue where the adapter has already
  normalized content to a single text string.
- load_rubrics: Load task-specific scoring rubrics from YAML config.
- build_judge_prompt: Compose a scoring prompt with anchored rubric to
  mitigate position and verbosity bias.
- resolve_rubric: Load a task rubric with a fallback when config is missing.
- judge_batch: Call the judge for a list of JudgeItem and return 1–5 scores,
  substituting a fallback score on call/parse failure.
- score_with_judge: **Phase 1 stub** that will call the judge via FMAPIClient
  and parse the result.

See plan.md appendix "채점 방법론 상세" (P0) for judge parsing strategy and
bias mitigation (position/verbosity).
"""

import re
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Optional

import yaml

DEFAULT_JUDGE_ENDPOINT = "databricks-gemini-3-1-pro"


def parse_judge_score(text: str) -> Optional[int]:
    """
    Robustly extract a 1–5 integer score from judge text.

    Handles various score formats:
    - "Score: 4" or "score: 4.5" (matches variations in phrasing)
    - "Rating: 3" or "rating: 4"
    - Standalone digit (e.g., "4") if it appears to be a score context
    - Clamped to [1, 5] range

    Handles the Gemini list-response issue: the caller (adapter) has already
    normalized Gemini's content list (`[{type, text, ...}]`) to a single
    plain-text string, so this function just parses the string robustly.

    Args:
        text: Judge response text (plain string, not structured).

    Returns:
        Integer score in [1, 5], or None if parsing fails.
    """
    if not text:
        return None

    text = text.strip()

    # Pattern 1: "score: N" or "rating: N" or "score is N" (case-insensitive, handles floats and any number)
    # Clamp result to [1, 5] even if specified value is out of range.
    pattern_named = r"(?:score|rating)(?:\s+is)?[\s:]+(\d+(?:\.\d+)?)"
    match = re.search(pattern_named, text, re.IGNORECASE)
    if match:
        try:
            score = float(match.group(1))
            score = int(score)  # Truncate/round as appropriate
            # Clamp: 0 → 1, 6+ → 5, others stay in [1,5]
            clamped = max(1, min(5, score))
            return clamped
        except (ValueError, IndexError):
            pass

    # Pattern 2: Standalone 1–5 digit at word boundary (most confident fallback)
    # Look for digits 1–5 that are isolated words, not part of longer numbers
    pattern_digit = r"\b([1-5])\b"
    matches = re.findall(pattern_digit, text)
    if matches:
        # Use the last one found (often the judge's final decision)
        try:
            return int(matches[-1])
        except (ValueError, IndexError):
            pass

    # Pattern 3: Last resort — any number digit string that could be clamped to [1,5]
    all_numbers = re.findall(r"\d+", text)
    if all_numbers:
        try:
            score = int(all_numbers[-1])
            if score != 0:  # Only clamp if it's a non-zero number
                return max(1, min(5, score))
        except (ValueError, IndexError):
            pass

    return None


def load_rubrics(path: str = "config/judge_rubrics.yaml") -> dict:
    """
    Load judge rubrics from YAML configuration.

    Rubrics define scoring anchors (1–5) for each task, guiding consistent
    judge evaluation. File format:

        task_id:
          name: "Task Name"
          description: "What this task measures"
          anchors:
            1: "Poor description (1 point)"
            2: "Fair description (2 points)"
            ...
            5: "Excellent description (5 points)"

    Args:
        path: Path to rubrics YAML file (relative to project root).

    Returns:
        Dictionary mapping task_id -> rubric definition (with name,
        description, anchors).

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    with open(path, "r", encoding="utf-8") as f:
        rubrics = yaml.safe_load(f)
    return rubrics or {}


def build_judge_prompt(
    task_id: str,
    question: str,
    reference: str,
    candidate: str,
    rubric: dict,
) -> str:
    """
    Compose a judge scoring prompt with anchored rubric.

    Constructs a prompt that:
    1. States the task and goal clearly.
    2. Presents the reference (ground truth) and candidate output.
    3. Embeds the 1–5 rubric to anchor scoring.
    4. Asks for a score in the specified range.

    Position and verbosity bias mitigation:
    - Reference and candidate are shown in fixed order (not shuffled per-sample,
      since the judge model is single and consistent).
    - Rubric is embedded inline to avoid hidden biases from unstated assumptions.
    - Prompt avoids length or style cues that might bias scoring.

    Args:
        task_id: Task identifier (e.g., "IMG-1", "TXT-4").
        question: Question or context (e.g., "What is in this image?").
        reference: Ground truth / reference output.
        candidate: Model's candidate output to be scored.
        rubric: Dict with keys {name, description, anchors},
                where anchors is a dict {1: "...", 2: "...", ..., 5: "..."}.

    Returns:
        Formatted prompt string ready to send to judge model.
    """
    rubric_name = rubric.get("name", task_id)
    rubric_desc = rubric.get("description", "")
    anchors = rubric.get("anchors", {})

    # Build anchor text
    anchor_text = "\n".join(
        f"  {score}: {desc}" for score, desc in sorted(anchors.items())
    )

    prompt = f"""You are an expert evaluator for the following task:

**Task:** {rubric_name}
**Description:** {rubric_desc}

**Question/Context:**
{question}

**Reference (ground truth):**
{reference}

**Candidate output to evaluate:**
{candidate}

**Scoring rubric (1–5):**
{anchor_text}

Please evaluate the candidate output against the reference and provide a score from 1 to 5.
Include brief reasoning, then state your final score clearly (e.g., "Score: 4").
"""
    return prompt


def resolve_rubric(
    task_id: str,
    fallback: dict | None = None,
    path: str = "config/judge_rubrics.yaml",
) -> dict:
    """task_id의 루브릭을 로드. 없거나 로드 실패면 fallback을 사용.

    Args:
        task_id: 태스크 식별자 (예: "TXT-1").
        fallback: 루브릭이 없을 때 쓸 기본 루브릭.
        path: 루브릭 YAML 경로.

    Returns:
        {name, description, anchors} 형태의 루브릭 (없으면 fallback 또는 {}).
    """
    try:
        rubrics = load_rubrics(path)
    except Exception as e:  # noqa: BLE001 - 루브릭 부재/파싱 오류 모두 fallback 처리
        print(f"Warning: 루브릭 로드 실패 ({path}): {e}")
        rubrics = {}

    return rubrics.get(task_id) or dict(fallback or {})


@dataclass
class JudgeItem:
    """judge 호출 1건에 필요한 입력."""

    question: str
    reference: str
    candidate: str
    sample_id: Any = None


def judge_batch(
    judge_client: Any,
    items: Sequence[JudgeItem],
    *,
    task_id: str,
    rubric: dict,
    judge_endpoint: str = DEFAULT_JUDGE_ENDPOINT,
    max_tokens: int = 256,
    fallback_score: int | None = 3,
    skip_empty: bool = False,
    extra_params: dict | None = None,
) -> list[int | None]:
    """항목별로 judge를 호출해 1–5 점수 리스트를 반환.

    호출 실패나 점수 파싱 실패 시 fallback_score를 사용한다(None이면 미평가로 남김).

    Args:
        judge_client: FMAPIClient (chat 메서드를 가진 클라이언트).
        items: judge 입력 목록. 반환 리스트는 이 순서와 1:1 대응.
        task_id: 프롬프트에 표시할 태스크 식별자.
        rubric: resolve_rubric 결과.
        judge_endpoint: judge 모델 엔드포인트.
        max_tokens: judge 응답 토큰 상한.
        fallback_score: 실패 시 사용할 점수 (일반적으로 중간값 3).
        skip_empty: True면 빈 candidate는 호출 없이 None 처리.
        extra_params: chat 호출에 그대로 전달할 추가 파라미터.

    Returns:
        items와 같은 길이의 점수 리스트 (실패 시 fallback_score 또는 None).
    """
    scores: list[int | None] = []

    for item in items:
        if skip_empty and not item.candidate:
            scores.append(None)
            continue

        prompt = build_judge_prompt(
            task_id=task_id,
            question=item.question,
            reference=item.reference,
            candidate=item.candidate,
            rubric=rubric,
        )

        try:
            response = judge_client.chat(
                endpoint=judge_endpoint,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                extra_params=extra_params,
            )
        except Exception as e:  # noqa: BLE001 - 개별 샘플 실패는 전체 평가를 막지 않음
            print(f"Judge 호출 실패 (샘플 {item.sample_id}): {e}")
            scores.append(fallback_score)
            continue

        score = parse_judge_score(response.text)
        scores.append(score if score is not None else fallback_score)

    return scores


def score_with_judge(
    task_id: str,
    question: str,
    reference: str,
    candidate: str,
    rubric: dict,
    judge_model_endpoint: str = "databricks-gemini-3-1-pro",
) -> Optional[int]:
    """
    Call the judge model and parse the score.

    **Phase 1 stub**: This function will:
    1. Build the judge prompt via build_judge_prompt().
    2. Call the judge model (FMAPI client) with the prompt.
    3. Parse the response via parse_judge_score().
    4. Return the 1–5 score, or None on failure.

    Implementation will integrate with the FMAPI adapter and handle
    retries/timeouts (see plan.md §11 runtime policy).

    Args:
        task_id: Task identifier.
        question: Question/context.
        reference: Ground truth.
        candidate: Candidate output.
        rubric: Scoring rubric.
        judge_model_endpoint: Endpoint name for the judge model.

    Returns:
        Judge score (1–5), or None if judge call fails.

    Raises:
        NotImplementedError: Phase 1 implementation pending.
    """
    raise NotImplementedError(
        "Phase 1: judge 호출 연결 (FMAPI 어댑터 + 파싱) 구현 필요"
    )
