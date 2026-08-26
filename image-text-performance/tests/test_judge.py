"""LLM-as-judge 파싱·프롬프트 테스트 (src/scoring/judge.py)."""

import pytest
import yaml
from src.scoring.judge import (
    build_judge_prompt,
    load_rubrics,
    parse_judge_score,
    score_with_judge,
)

RUBRIC = {
    "name": "Image Captioning",
    "description": "이미지 설명 품질",
    "anchors": {1: "매우 나쁨", 2: "나쁨", 3: "보통", 4: "좋음", 5: "매우 좋음"},
}


# -------------------------------------------------------- parse_judge_score ---
@pytest.mark.parametrize("text,expected", [
    ("Score: 4", 4),
    ("score: 4", 4),
    ("SCORE:5", 5),
    ("Rating: 3", 3),
    ("The score is 2", 2),
    ("Score: 4.7", 4),
])
def test_named_score_patterns(text, expected):
    assert parse_judge_score(text) == expected


@pytest.mark.parametrize("text,expected", [("Score: 0", 1), ("Score: 9", 5)])
def test_named_score_is_clamped(text, expected):
    assert parse_judge_score(text) == expected


def test_named_pattern_wins_over_other_digits():
    assert parse_judge_score("Anchor 1 and anchor 5 apply. Score: 3") == 3


def test_reasoning_then_final_score():
    text = "The caption misses two objects but is fluent.\n\nScore: 3\n"
    assert parse_judge_score(text) == 3


def test_standalone_digit_uses_last_occurrence():
    assert parse_judge_score("Between 2 and 4, I choose 4") == 4


def test_bare_digit_response():
    assert parse_judge_score("5") == 5


def test_out_of_range_number_without_keyword_is_clamped():
    assert parse_judge_score("I would give this a 42") == 5


def test_zero_only_response_is_unparseable():
    assert parse_judge_score("0") is None


@pytest.mark.parametrize("text", ["", "   ", "unable to evaluate"])
def test_unparseable_returns_none(text):
    assert parse_judge_score(text) is None


def test_none_input_returns_none():
    assert parse_judge_score(None) is None


# ------------------------------------------------------------- load_rubrics ---
def test_load_rubrics_from_yaml(tmp_path):
    path = tmp_path / "judge_rubrics.yaml"
    path.write_text(yaml.safe_dump({"IMG-1": RUBRIC}, allow_unicode=True), encoding="utf-8")
    rubrics = load_rubrics(str(path))
    assert rubrics["IMG-1"]["name"] == "Image Captioning"
    assert rubrics["IMG-1"]["anchors"][5] == "매우 좋음"


def test_empty_rubrics_file_yields_empty_dict(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_rubrics(str(path)) == {}


def test_missing_rubrics_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rubrics(str(tmp_path / "nope.yaml"))


def test_repo_rubrics_config_is_loadable_if_present():
    from pathlib import Path

    path = Path("config/judge_rubrics.yaml")
    if not path.exists():
        pytest.skip("리포지토리에 judge_rubrics.yaml이 없음")
    assert isinstance(load_rubrics(str(path)), dict)


# -------------------------------------------------------- build_judge_prompt ---
def test_prompt_contains_task_inputs_and_anchors():
    prompt = build_judge_prompt("IMG-1", "이 이미지는?", "빨간 사과", "사과 사진", RUBRIC)
    assert "Image Captioning" in prompt
    assert "이미지 설명 품질" in prompt
    assert "이 이미지는?" in prompt
    assert "빨간 사과" in prompt
    assert "사과 사진" in prompt
    assert "  1: 매우 나쁨" in prompt
    assert "  5: 매우 좋음" in prompt


def test_prompt_orders_anchors_ascending():
    shuffled = {"anchors": {5: "다섯", 1: "하나", 3: "셋"}}
    lines = [line for line in build_judge_prompt("T", "q", "r", "c", shuffled).splitlines()
             if line.startswith("  ")]
    assert lines == ["  1: 하나", "  3: 셋", "  5: 다섯"]


def test_prompt_falls_back_to_task_id_without_rubric():
    prompt = build_judge_prompt("TXT-4", "q", "r", "c", {})
    assert "**Task:** TXT-4" in prompt
    assert "**Description:** " in prompt


def test_prompt_asks_for_explicit_final_score():
    prompt = build_judge_prompt("IMG-1", "q", "r", "c", RUBRIC)
    assert 'state your final score clearly (e.g., "Score: 4")' in prompt
    assert parse_judge_score(prompt.split("Scoring rubric")[1]) is not None


def test_reference_precedes_candidate_for_stable_position_bias():
    prompt = build_judge_prompt("IMG-1", "q", "REFTEXT", "CANDTEXT", RUBRIC)
    assert prompt.index("REFTEXT") < prompt.index("CANDTEXT")


# --------------------------------------------------------- score_with_judge ---
def test_score_with_judge_is_phase1_stub():
    with pytest.raises(NotImplementedError):
        score_with_judge("IMG-1", "q", "r", "c", RUBRIC)
