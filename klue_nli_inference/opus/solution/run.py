"""Final reproducible pipeline for t6_klue_nli.

Usage:  python solution/run.py [--task-dir DIR] [--eval SEED]

Method
------
1. Feature extraction (`nli_lib.build_features`): TF-IDF over several text views
   of the (premise, hypothesis) pair -- hypothesis words/chars, premise words,
   "new information" tokens (hypothesis tokens absent from the premise), shared
   tokens, dropped premise tokens, and the concatenated pair -- with a crude
   Korean stemmer (leading 2..4 chars of each token) plus 40 hand-crafted NLI
   cue features (overlap ratios, negation / quantifier / hedge counts, numeric
   mismatch, length ratios).
2. Base classifier: multinomial logistic regression (C=0.15).
3. Group-constraint decoding: KLUE-NLI pairs every premise with exactly one
   entailment, one neutral and one contradiction hypothesis.  Since the split is
   row-wise, 96% of test rows share their premise with labelled train rows.  Each
   premise group is decoded jointly by enumerating label assignments, weighting
   them by an empirical prior over label multisets (estimated from the fully
   observed train groups: all-distinct is ~28x more likely per ordered tuple than
   a repeated label) times the tempered base-classifier likelihood.

Validation (pseudo-test reproducing the real test's group structure):
   base classifier alone  ~0.578 accuracy
   + group decoding       ~0.878 accuracy
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nli_lib import (CLASSES, C2I, PAIRS, build_features, decode_groups_pair,
                     multiset_prior, norm)

BASE_C = 0.15
TEMP = 1.5
SEED = 0


def load(task_dir):
    tr = pd.read_csv(os.path.join(task_dir, "train.csv"))
    te = pd.read_csv(os.path.join(task_dir, "test.csv"))
    tr["key"] = tr.premise.map(norm)
    te["key"] = te.premise.map(norm)
    return tr, te


def solve(fit_df, pred_df, base_c=BASE_C, temp=TEMP, verbose=True, prior_df=None):
    """fit_df needs premise/hypothesis/label/key, pred_df premise/hypothesis/key.

    prior_df: labelled frame used to estimate the label-multiset prior.  It must
    contain *complete* premise groups, so in offline validation pass the full
    train set here (the fit split has all of its size-3 groups truncated).
    """
    y_fit = fit_df.label.map(C2I).values
    n_fit = len(fit_df)
    df = pd.concat([fit_df[["premise", "hypothesis", "key"]],
                    pred_df[["premise", "hypothesis", "key"]]], ignore_index=True)
    t0 = time.time()
    X = build_features(df, n_fit)
    if verbose:
        print("features %s (%.0fs)" % (X.shape, time.time() - t0), flush=True)

    clf = LogisticRegression(C=base_c, max_iter=2000, random_state=SEED)
    clf.fit(X[:n_fit], y_fit)
    proba = clf.predict_proba(X[n_fit:])
    if verbose:
        print("base model fitted (%.0fs)" % (time.time() - t0), flush=True)

    # empirical prior over label multisets, from fully observed size-3 groups
    grp = (fit_df if prior_df is None else prior_df).groupby("key").label.apply(list)
    w3, rho = multiset_prior([v for v in grp if len(v) == 3])
    if verbose:
        print("multiset prior w3=%s" % {k: round(v, 5) for k, v in w3.items()},
              flush=True)

    kf, ku = fit_df.key.values, pred_df.key.values
    zu = {p: np.zeros(len(pred_df)) for p in PAIRS}
    zf = {p: np.zeros(n_fit) for p in PAIRS}
    post = decode_groups_pair(kf, y_fit, ku, proba, zu, zf, w3, rho,
                              temp=temp, beta=0.0)
    return proba, post


def make_pseudo_test(tr, seed):
    """Hold out rows so the group structure matches the real test set."""
    rng = np.random.RandomState(seed)
    groups = {k: np.array(v) for k, v in tr.groupby("key").indices.items()}
    keys3 = [k for k, v in groups.items() if len(v) == 3]
    rng.shuffle(keys3)
    n1 = int(round(len(keys3) * 0.795))
    hold = []
    for i, k in enumerate(keys3):
        v = groups[k].copy(); rng.shuffle(v)
        hold.extend(v[:1] if i < n1 else v[:2])
    other = [k for k, v in groups.items() if len(v) != 3]
    rng.shuffle(other)
    c = 0
    for k in other:                      # mimic the 178 unseen-premise test rows
        if c >= 178:
            break
        hold.extend(groups[k]); c += len(groups[k])
    hold = np.array(sorted(set(hold)))
    m = np.zeros(len(tr), bool); m[hold] = True
    return np.where(~m)[0], hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--eval", type=int, default=None,
                    help="run offline validation with this seed instead of submitting")
    ap.add_argument("--base-c", type=float, default=BASE_C)
    ap.add_argument("--temp", type=float, default=TEMP)
    a = ap.parse_args()
    tr, te = load(a.task_dir)

    if a.eval is not None:
        fit_i, hold_i = make_pseudo_test(tr, a.eval)
        fit_df, ho_df = tr.iloc[fit_i], tr.iloc[hold_i]
        y_true = ho_df.label.map(C2I).values
        proba, post = solve(fit_df, ho_df, a.base_c, a.temp, prior_df=tr)
        nk = pd.Series(ho_df.key.values).map(
            pd.Series(fit_df.key.values).value_counts()).fillna(0).astype(int).values
        res = pd.DataFrame({"nk": nk, "ok": post.argmax(1) == y_true})
        print("seed=%d base %.4f -> decoded %.4f" %
              (a.eval, (proba.argmax(1) == y_true).mean(), res.ok.mean()))
        print(res.groupby("nk").ok.agg(["size", "mean"]))
        return

    proba, post = solve(tr, te, a.base_c, a.temp)
    pred = post.argmax(1)
    out = os.path.join(a.task_dir, "outputs")
    os.makedirs(out, exist_ok=True)
    sub = pd.DataFrame({"id": te.id.values, "label": [CLASSES[i] for i in pred]})
    sub.to_csv(os.path.join(out, "submission.csv"), index=False)
    print(sub.label.value_counts())
    print("agreement base vs decoded: %.4f" % (proba.argmax(1) == pred).mean())
    print("wrote", os.path.join(out, "submission.csv"), sub.shape)


if __name__ == "__main__":
    main()
