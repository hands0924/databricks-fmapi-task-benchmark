"""
Statistical tests for benchmark result analysis.

This module provides statistical significance testing to avoid overclaiming
performance differences. See plan.md §7 and appendix for design rationale.

Key function:
- wilcoxon_test: Paired non-parametric test (Wilcoxon signed-rank) for
  comparing two sets of scores, typically across multiple samples.
"""

import sys

from scipy import stats


def wilcoxon_test(
    scores_a: list[float], scores_b: list[float]
) -> dict:
    """
    Paired Wilcoxon signed-rank test for statistical significance.

    Compares two sets of scores (typically from multiple samples) to determine
    if the difference is statistically significant. This test does NOT assume
    normality and is suitable for small samples or non-normal distributions
    (common in language task metrics).

    Used in the benchmark report (plan.md §7, appendix) to avoid overclaiming:
    only report performance differences as significant if p-value < 0.05.

    Handles edge cases gracefully:
    - If all differences are zero (or too few non-zero diffs), returns pval=None
      and significant=False (cannot compute meaningful test).
    - If sample sizes are too small (<2), returns pval=None.

    Args:
        scores_a: List of scores from model/approach A.
        scores_b: List of scores from model/approach B.

    Returns:
        Dictionary with keys:
        - pval: p-value from the test, or None if test is invalid (no variation,
                too few samples).
        - significant: boolean, True if pval < 0.05, False otherwise.
        - n: int, number of samples (pairs).
        - error: exception type and message if scipy fails, otherwise None.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")

    n = len(scores_a)

    if n < 2:
        return {
            "pval": None,
            "significant": False,
            "n": n,
            "error": None,
        }

    # Compute differences
    diffs = [a - b for a, b in zip(scores_a, scores_b)]

    # Remove pairs where difference is zero (Wilcoxon ignores them anyway,
    # but we check explicitly to catch the all-zero case)
    nonzero_diffs = [d for d in diffs if d != 0]

    if len(nonzero_diffs) < 2:
        # Not enough variation to compute a meaningful test
        return {
            "pval": None,
            "significant": False,
            "n": n,
            "error": None,
        }

    # Compute Wilcoxon signed-rank test
    try:
        _stat, pval = stats.wilcoxon(diffs)
    except ValueError as e:
        print(
            f"Wilcoxon test failed ({type(e).__name__}: {e})",
            file=sys.stderr,
        )
        return {
            "pval": None,
            "significant": False,
            "n": n,
            "error": f"{type(e).__name__}: {e}",
        }

    # Check significance at α=0.05
    significant = pval < 0.05

    return {
        "pval": float(pval),
        "significant": bool(significant),
        "n": int(n),
        "error": None,
    }
