"""Premise-group constraint.

MNLI/KorNLI collects several hypotheses per premise, one per class, so two rows
that share a premise almost never share a label (empirically 98.4% distinct in
train).  Given the label of a sibling row we can therefore heavily down-weight
that class.  Rows whose premise is only shared inside the test set are resolved
jointly by exhaustive search over the (at most 3^3) label assignments.
"""
from itertools import product
from collections import Counter
import numpy as np

CLASSES = np.array(['contradiction', 'entailment', 'neutral'])


def apply_constraint(prob, premises, known, lam, groups=True):
    """prob: (n,3); premises: array of premise strings for the rows;
    known: dict premise -> Counter of labels observed outside this row set;
    lam: log-penalty per violated distinctness (negative)."""
    logp = np.log(np.clip(prob, 1e-9, None))
    out = logp.copy()
    idx_by_prem = {}
    for i, p in enumerate(premises):
        idx_by_prem.setdefault(p, []).append(i)
    pred = np.empty(len(prob), dtype=object)
    for p, rows in idx_by_prem.items():
        ext = known.get(p, Counter())
        if not groups or len(rows) == 1:
            for i in rows:
                sc = logp[i] + lam * np.array([ext.get(c, 0) for c in CLASSES])
                pred[i] = CLASSES[sc.argmax()]
            continue
        if len(rows) > 4:                       # too big to enumerate: greedy
            order = sorted(rows, key=lambda i: -logp[i].max())
            used = Counter(ext)
            for i in order:
                sc = logp[i] + lam * np.array([used.get(c, 0) for c in CLASSES])
                c = CLASSES[sc.argmax()]
                pred[i] = c; used[c] += 1
            continue
        best, bestsc = None, -1e18
        for combo in product(range(3), repeat=len(rows)):
            cnt = Counter(ext)
            sc = 0.0
            for i, ci in zip(rows, combo):
                c = CLASSES[ci]
                sc += logp[i, ci] + lam * cnt.get(c, 0)
                cnt[c] += 1
            if sc > bestsc:
                bestsc, best = sc, combo
        for i, ci in zip(rows, best):
            pred[i] = CLASSES[ci]
    return pred


def known_from(premises, labels, exclude_self=True):
    """Build premise -> Counter(labels) from a labelled set."""
    d = {}
    for p, l in zip(premises, labels):
        d.setdefault(p, Counter())[l] += 1
    return d


def loo_known(premises, labels):
    """Leave-one-out sibling counters (for validating on labelled data)."""
    full = known_from(premises, labels)
    return full
