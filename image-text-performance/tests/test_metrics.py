"""정량 메트릭 테스트 (src/scoring/metrics.py)."""

import pytest
from src.scoring.metrics import (
    bertscore,
    binary_metrics,
    cell_f1,
    classification_metrics,
    exact_match,
    multilabel_prf,
    rouge_ko_en,
    token_f1,
)


# ------------------------------------------------------------------ token_f1 ---
def test_token_f1_identical_is_one():
    assert token_f1("the quick brown fox", "the quick brown fox", "en") == 1.0


def test_token_f1_disjoint_is_zero():
    assert token_f1("alpha beta", "gamma delta", "en") == 0.0


def test_token_f1_partial_overlap():
    # pred=3 tokens, gold=2 tokens, overlap=2 → 2*2/5
    assert token_f1("the quick fox", "the quick", "en") == pytest.approx(0.8)


def test_token_f1_counts_tokens_as_multiset():
    # pred has "a" twice, gold once → overlap 1, total 3
    assert token_f1("a a", "a", "en") == pytest.approx(2 / 3)


def test_token_f1_both_empty_is_one():
    assert token_f1("", "   ", "en") == 1.0


@pytest.mark.parametrize("pred,gold", [("", "something"), ("something", "")])
def test_token_f1_one_empty_is_zero(pred, gold):
    assert token_f1(pred, gold, "en") == 0.0


def test_token_f1_korean_uses_subword_tokenization():
    """공백 분리라면 0이 되는 조사 차이가 형태소/음절 토큰화로는 부분 점수를 받는다."""
    score = token_f1("서울은 대한민국의 수도", "서울이 대한민국 수도", "ko")
    assert 0.0 < score < 1.0


# --------------------------------------------------------------- exact_match ---
@pytest.mark.parametrize("pred,gold,expected", [
    ("Seoul", "seoul", 1.0),
    ("  Seoul  ", "Seoul", 1.0),
    ("Seoul", "Busan", 0.0),
    ("", "", 1.0),
    ("Seoul city", "Seoul", 0.0),
])
def test_exact_match(pred, gold, expected):
    assert exact_match(pred, gold) == expected


# ------------------------------------------------------------ binary_metrics ---
def test_binary_metrics_perfect():
    out = binary_metrics([1, 0, 1, 0], [1, 0, 1, 0])
    assert out["accuracy"] == 1.0
    assert out["f1"] == 1.0
    assert out["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_binary_metrics_confusion_counts():
    out = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert out["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert out["accuracy"] == 0.5
    assert out["f1"] == pytest.approx(0.5)


def test_binary_metrics_all_negative_predictions_f1_zero_not_nan():
    out = binary_metrics([0, 0], [1, 1])
    assert out["f1"] == 0.0
    assert out["accuracy"] == 0.0


def test_binary_metrics_single_class_labels_still_2x2():
    out = binary_metrics([0, 0], [0, 0])
    assert out["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 0}


# ----------------------------------------------------------- multilabel_prf ---
def test_multilabel_prf_perfect():
    out = multilabel_prf([{"a", "b"}, {"c"}], [{"a", "b"}, {"c"}])
    assert out["micro_f1"] == 1.0
    assert out["macro_f1"] == 1.0


def test_multilabel_prf_micro_and_macro_differ():
    pred = [{"a", "b", "x"}, {"c"}]
    gold = [{"a", "b"}, {"c", "d", "e"}]
    out = multilabel_prf(pred, gold)
    # micro: tp=3, fp=1, fn=2
    assert out["micro_precision"] == pytest.approx(0.75)
    assert out["micro_recall"] == pytest.approx(0.6)
    assert out["micro_f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    # macro: sample1 p=2/3 r=1.0, sample2 p=1.0 r=1/3
    assert out["macro_precision"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert out["macro_recall"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert out["macro_f1"] != out["micro_f1"]


def test_multilabel_prf_empty_predictions_score_zero():
    out = multilabel_prf([set(), set()], [{"a"}, {"b"}])
    assert out["micro_precision"] == 0.0
    assert out["micro_recall"] == 0.0
    assert out["macro_f1"] == 0.0


def test_multilabel_prf_no_samples_returns_zeros():
    out = multilabel_prf([], [])
    assert set(out.values()) == {0.0}


def test_multilabel_prf_length_mismatch_raises():
    with pytest.raises(ValueError):
        multilabel_prf([{"a"}], [{"a"}, {"b"}])


# --------------------------------------------------- classification_metrics ---
def test_classification_metrics_multiclass():
    out = classification_metrics([0, 1, 2, 2], [0, 1, 2, 1])
    assert out["accuracy"] == 0.75
    assert 0.0 < out["macro_f1"] < 1.0


def test_classification_metrics_unseen_class_no_nan():
    out = classification_metrics([0, 0, 0], [0, 1, 2])
    assert out["accuracy"] == pytest.approx(1 / 3)
    assert out["macro_f1"] == pytest.approx(0.5 / 3)


# --------------------------------------------------------------- Phase 1 stubs ---
@pytest.mark.parametrize("call", [
    lambda: rouge_ko_en("a", "b", "ko"),
    lambda: bertscore(["a"], ["b"], "en"),
    lambda: cell_f1("<table></table>", "<table></table>"),
])
def test_phase1_stubs_raise_not_implemented(call):
    with pytest.raises(NotImplementedError):
        call()
