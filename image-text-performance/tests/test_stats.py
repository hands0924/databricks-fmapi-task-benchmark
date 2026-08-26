"""통계 유의성 검정 테스트 (src/scoring/stats.py)."""

import pytest
from src.scoring.stats import wilcoxon_test


def test_clear_difference_is_significant():
    a = [0.9, 0.88, 0.91, 0.93, 0.95, 0.9, 0.92, 0.94, 0.9, 0.91]
    b = [0.5, 0.48, 0.52, 0.51, 0.49, 0.5, 0.53, 0.47, 0.5, 0.52]
    out = wilcoxon_test(a, b)
    assert out["n"] == 10
    assert out["pval"] < 0.05
    assert out["significant"] is True


def test_noise_is_not_significant():
    a = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5, 0.52, 0.48]
    b = [0.51, 0.59, 0.42, 0.53, 0.47, 0.49, 0.5, 0.5]
    out = wilcoxon_test(a, b)
    assert out["significant"] is False


def test_identical_scores_cannot_be_tested():
    out = wilcoxon_test([0.5] * 6, [0.5] * 6)
    assert out == {"pval": None, "significant": False, "n": 6}


def test_single_nonzero_difference_is_too_few():
    out = wilcoxon_test([0.5, 0.5, 0.9], [0.5, 0.5, 0.5])
    assert out["pval"] is None
    assert out["significant"] is False
    assert out["n"] == 3


@pytest.mark.parametrize("a,b,n", [([], [], 0), ([0.1], [0.2], 1)])
def test_too_few_samples(a, b, n):
    assert wilcoxon_test(a, b) == {"pval": None, "significant": False, "n": n}


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        wilcoxon_test([0.1, 0.2], [0.1])


def test_pval_and_flags_are_plain_python_types():
    out = wilcoxon_test([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert isinstance(out["pval"], float)
    assert isinstance(out["significant"], bool)
    assert isinstance(out["n"], int)


def test_scipy_failure_degrades_to_none(monkeypatch):
    import src.scoring.stats as stats_mod

    class Boom:
        @staticmethod
        def wilcoxon(_diffs):
            raise RuntimeError("scipy said no")

    monkeypatch.setattr(stats_mod, "stats", Boom)
    assert wilcoxon_test([1.0, 2.0, 3.0], [0.0, 1.0, 0.5]) == {
        "pval": None, "significant": False, "n": 3,
    }
