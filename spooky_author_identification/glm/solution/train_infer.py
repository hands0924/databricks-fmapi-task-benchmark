"""
Spooky Author Identification - final solution.

Approach:
  - Word TF-IDF (1-3 grams, min_df=2, sublinear) + Char-wb TF-IDF (2-6 grams,
    min_df=3, sublinear), stacked.
  - Logistic Regression (L2, C=25, liblinear).
  - 10-fold Stratified CV, averaged over 8 seeds (seed-bagging) for stability and
    lower log-loss via ensemble variance reduction.
  - Light probability smoothing (eps) to keep log-loss safe at extremes.

Reproducible (fixed seeds). Only uses sklearn / numpy / pandas / scipy.
Metric: multiclass log-loss (lower is better). OOF log-loss ~0.366.

Run: python solution/train_infer.py
Outputs: outputs/submission.csv (id,EAP,HPL,MWS).
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

N_FOLDS = 10
SEEDS = [42, 2024, 7, 99, 13, 555, 314, 271]
C = 25.0
CLASSES = ["EAP", "HPL", "MWS"]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.dirname(here)
    train = pd.read_csv(os.path.join(task_dir, "train.csv"))
    test = pd.read_csv(os.path.join(task_dir, "test.csv"))
    sub = pd.read_csv(os.path.join(task_dir, "sample_submission.csv"))

    train["text"] = train["text"].fillna("").astype(str)
    test["text"] = test["text"].fillna("").astype(str)

    y = train["author"].values
    le = LabelEncoder().fit(CLASSES)
    y_enc = le.transform(y)

    # ---- Feature extraction ----
    word_vec = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 3), min_df=2, max_df=0.9,
        sublinear_tf=True, strip_accents="unicode", analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )
    char_vec = TfidfVectorizer(
        lowercase=True, ngram_range=(2, 6), min_df=3, max_df=0.95,
        sublinear_tf=True, analyzer="char_wb",
    )
    Xw_tr = word_vec.fit_transform(train["text"])
    Xw_te = word_vec.transform(test["text"])
    Xc_tr = char_vec.fit_transform(train["text"])
    Xc_te = char_vec.transform(test["text"])
    X_tr = hstack([Xw_tr, Xc_tr]).tocsr()
    X_te = hstack([Xw_te, Xc_te]).tocsr()
    print("train feature shape:", X_tr.shape)
    print("test  feature shape:", X_te.shape)

    # ---- Seed-bagged 10-fold OOF + test ensemble ----
    oof_acc = np.zeros((len(train), 3))
    test_pred = np.zeros((len(test), 3))
    n_total = len(SEEDS) * N_FOLDS
    t0 = time.time()
    for s in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        oof = np.zeros((len(train), 3))
        for tr_idx, va_idx in skf.split(X_tr, y_enc):
            clf = LogisticRegression(
                C=C, penalty="l2", solver="liblinear",
                max_iter=3000, random_state=s,
            )
            clf.fit(X_tr[tr_idx], y_enc[tr_idx])
            p = clf.predict_proba(X_tr[va_idx])
            p = p / p.sum(1, keepdims=True)
            oof[va_idx] = p
            test_pred += (clf.predict_proba(X_te) / n_total)
        oof_acc += oof / len(SEEDS)
        ll = log_loss(y_enc, oof_acc, labels=[0, 1, 2])
        print(f"seed {s}: cumul oof ll={ll:.4f} t={time.time()-t0:.1f}s", flush=True)

    overall = log_loss(y_enc, oof_acc, labels=[0, 1, 2])
    print(f"FINAL seed-avg OOF ll={overall:.4f}")

    # ---- Light smoothing to avoid 0/1 extremes ----
    eps = 1e-4
    test_pred = test_pred + eps
    test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

    # ---- Build submission ----
    out = pd.DataFrame({"id": test["id"].values})
    for i, c in enumerate(CLASSES):
        out[c] = test_pred[:, i]
    out = out[["id"] + CLASSES]

    out_path = os.path.join(task_dir, "outputs", "submission.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print("saved:", out_path, "shape", out.shape)

    assert set(out["id"]) == set(sub["id"]), "id mismatch"
    assert len(out) == len(sub), "row count mismatch"
    print("submission sanity OK")


if __name__ == "__main__":
    main()
