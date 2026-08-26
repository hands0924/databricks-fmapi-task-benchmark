"""src/tasks/common.py 공용 유틸리티 테스트 (네트워크/데이터셋 불필요)."""

from src.scoring.judge import JudgeItem, judge_batch
from src.tasks.base import Sample
from src.tasks.common import (
    best_reference_score,
    binary_score_summary,
    detect_column,
    language_batches,
    mean,
    parse_binary_label,
    per_language_stats,
)


def _sample(sample_id: int, reference: int | str, lang: str = "en") -> Sample:
    return Sample(sample_id=sample_id, inputs={}, reference=reference, lang=lang)


def test_mean_empty_uses_default():
    assert mean([]) == 0.0
    assert mean([], default=3.0) == 3.0
    assert mean([1.0, 2.0]) == 1.5


def test_detect_column_prefers_priority_then_exclusion():
    rows = [{"label": 0, "sentence": "hi"}]
    assert detect_column(rows, ["sentence", "text"]) == "sentence"
    assert detect_column(rows, ["missing"], exclude=["sentence"]) == "label"


def test_detect_column_raises_without_fallback():
    try:
        detect_column([{"a": 1}], ["b"], what="텍스트 컬럼")
    except ValueError as e:
        assert "텍스트 컬럼" in str(e)
    else:
        raise AssertionError("컬럼 감지 실패 시 ValueError를 기대")


def test_parse_binary_label_keywords_and_numeric_fallback():
    kwargs = {"positive": ("toxic", "유해"), "negative": ("clean", "정상")}
    assert parse_binary_label("Toxic", **kwargs) == 1
    assert parse_binary_label("정상입니다", **kwargs) == 0
    assert parse_binary_label("label: 1", **kwargs) == 1
    assert parse_binary_label("label: 0", **kwargs) == 0
    assert parse_binary_label("   ", **kwargs) is None
    assert parse_binary_label("no idea", **kwargs) is None


def test_parse_binary_label_negative_first_ordering():
    kwargs = {"positive": ("yes", "weapon"), "negative": ("no", "none")}
    # "no weapon"은 부정 우선 검사에서 0, 기본 순서에서는 1
    assert parse_binary_label("no weapon", negative_first=True, **kwargs) == 0
    assert parse_binary_label("no weapon", **kwargs) == 1


def test_binary_score_summary_counts_unparsed_and_class_balance():
    samples = [_sample(0, 1), _sample(1, 0), _sample(2, 1)]
    result = binary_score_summary(
        [1, 0, None], samples, class_balance_keys=("sfw_count", "nsfw_count")
    )

    assert result["n_evaluated"] == 2
    assert result["n_unparsed"] == 1
    assert result["accuracy"] == 1.0
    assert result["class_balance"] == {"sfw_count": 1, "nsfw_count": 1}
    assert set(result["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}


def test_binary_score_summary_all_unparsed():
    result = binary_score_summary([None, None], [_sample(0, 1), _sample(1, 0)], per_language=True)

    assert result == {
        "accuracy": 0.0,
        "f1": 0.0,
        "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        "n_evaluated": 0,
        "n_unparsed": 2,
        "per_language": {},
    }


def test_per_language_stats_handles_missing_language():
    samples = [_sample(0, 1, "en"), _sample(1, 0, "en"), _sample(2, 1, "ko")]
    stats = per_language_stats(
        [1, 0, None],
        samples,
        metric_fn=lambda preds, golds: {"accuracy": float(preds == golds)},
        metric_keys=("accuracy",),
    )

    assert stats["en"] == {"n_evaluated": 2, "n_unparsed": 0, "accuracy": 1.0}
    assert stats["ko"] == {"n_evaluated": 0, "n_unparsed": 1, "accuracy": None}


def test_best_reference_score_picks_max_and_accepts_scalar():
    scorer = {("p", "a"): 0.2, ("p", "b"): 0.7}
    assert best_reference_score("p", ["a", "b"], lambda p, g: scorer[(p, g)]) == 0.7
    assert best_reference_score("p", "a", lambda p, g: scorer[(p, g)]) == 0.2


def test_language_batches_splits_remainder_to_earlier_languages(monkeypatch):
    calls = []

    def fake_load_dataset_rows(dataset_key, n, seed, *, registry=None, default_split="train"):
        calls.append((dataset_key, n, default_split))
        return [{"text": dataset_key}] * n

    monkeypatch.setattr("src.tasks.common.load_dataset_rows", fake_load_dataset_rows)

    config = {"datasets": {"en": "toxicity_en", "ko": "toxicity_ko"}}
    batches = list(language_batches(config, 5, 42, registry={}))

    assert [(b.lang, b.dataset_key, len(b.rows)) for b in batches] == [
        ("en", "toxicity_en", 3),
        ("ko", "toxicity_ko", 2),
    ]
    assert calls == [("toxicity_en", 3, "train"), ("toxicity_ko", 2, "train")]


def test_language_batches_requires_datasets_map():
    try:
        list(language_batches({}, 4, 42))
    except ValueError as e:
        assert "datasets" in str(e)
    else:
        raise AssertionError("datasets 맵이 없으면 ValueError를 기대")


class _StubJudgeClient:
    """chat 호출을 흉내내는 스텁: 응답 텍스트를 순서대로 반환하거나 예외 발생."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, endpoint, messages, *, max_tokens=1024, extra_params=None):
        self.calls.append((endpoint, messages, max_tokens, extra_params))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return type("ChatResponse", (), {"text": response})()


def test_judge_batch_parses_scores_and_uses_fallback_on_failure():
    client = _StubJudgeClient(["Score: 4", "no score here", RuntimeError("boom")])
    items = [
        JudgeItem(question="q", reference="r", candidate=f"c{i}", sample_id=i) for i in range(3)
    ]

    scores = judge_batch(client, items, task_id="TXT-1", rubric={})

    assert scores == [4, 3, 3]
    assert len(client.calls) == 3


def test_judge_batch_skip_empty_and_none_fallback():
    client = _StubJudgeClient([RuntimeError("boom")])
    items = [
        JudgeItem(question="q", reference="r", candidate="", sample_id=0),
        JudgeItem(question="q", reference="r", candidate="c", sample_id=1),
    ]

    scores = judge_batch(
        client, items, task_id="IMG-1", rubric={}, fallback_score=None, skip_empty=True
    )

    assert scores == [None, None]
    assert len(client.calls) == 1  # 빈 candidate는 호출하지 않음
