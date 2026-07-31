#!/usr/bin/env python3
"""
K-MLE-Bench t6_klue_nli - Korean Natural Language Inference.

Approach (no external data / no internet / scikit-learn only):
  - Character n-gram TF-IDF features (robust for Korean without a tokenizer).
  - Combined features from premise, hypothesis, and their concatenation.
  - Logistic Regression + Linear SVM (calibrated) ensemble plus handcrafted
    lexical overlap features.

Usage:
    python3 solution.py
Outputs:
    ../outputs/submission.csv  (id,label)
"""
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN = ROOT / "train.csv"
TEST = ROOT / "test.csv"
OUT = ROOT / "outputs" / "submission.csv"

LABELS = ["entailment", "neutral", "contradiction"]
RNG = 42


def normalize(text):
    """Light normalization: collapse repeated whitespace, lowercase."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def char_tokens(text, n=3):
    """Character n-gram tokens (spaces removed) for Korean robustness."""
    text = re.sub(r"\s+", "", text)
    if len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def space_word_tokens(text):
    return text.split()


def overlap_ratio(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0, 0.0, 0.0
    inter = len(sa & sb)
    return inter / len(sa), inter / len(sb), inter / len(sa | sb)


def char_overlap(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0, 0.0
    inter = len(sa & sb)
    return inter / len(sa), inter / len(sb)


NEG_CUES = ["안", "못", "아니", "없", "지 않", "결코", "전혀", "절대", "아무것도",
            "아무도", "비", "반대", "다르", "틀리"]


def neg_cues(text):
    return sum(1 for c in NEG_CUES if c in text)


def handcraft(df):
    feats = []
    for p, h in zip(df["premise"].astype(str), df["hypothesis"].astype(str)):
        pn, hn = normalize(p), normalize(h)
        o1, o2, jac = overlap_ratio(pn, hn)
        co1, co2 = char_overlap(pn, hn)
        feats.append([
            len(pn), len(hn), len(pn) - len(hn), len(pn) / (len(hn) + 1.0),
            len(pn.split()), len(hn.split()),
            len(pn.split()) - len(hn.split()),
            o1, o2, jac, co1, co2,
            neg_cues(pn), neg_cues(hn), abs(neg_cues(pn) - neg_cues(hn)),
            float(pn in hn or hn in pn),
        ])
    return np.array(feats, dtype=np.float32)


def build_vectorizers():
    vchar = TfidfVectorizer(
        analyzer=lambda x: char_tokens(x, 3), ngram_range=(1, 1),
        sublinear_tf=True, min_df=2, max_df=0.9, max_features=80000)
    vchar2 = TfidfVectorizer(
        analyzer=lambda x: char_tokens(x, 2), ngram_range=(1, 1),
        sublinear_tf=True, min_df=2, max_df=0.9, max_features=80000)
    vchar4 = TfidfVectorizer(
        analyzer=lambda x: char_tokens(x, 4), ngram_range=(1, 1),
        sublinear_tf=True, min_df=3, max_df=0.9, max_features=80000)
    vword = TfidfVectorizer(
        analyzer=space_word_tokens, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, max_df=0.9, max_features=60000)
    return vchar, vchar2, vchar4, vword


def main():
    t0 = time.time()
    print("Loading data...", flush=True)
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    print(f"train={train.shape} test={test.shape}", flush=True)

    train["premise"] = train["premise"].map(normalize)
    train["hypothesis"] = train["hypothesis"].map(normalize)
    test["premise"] = test["premise"].map(normalize)
    test["hypothesis"] = test["hypothesis"].map(normalize)

    y = train["label"].map({l: i for i, l in enumerate(LABELS)}).values
    Xp_tr = train["premise"].values
    Xh_tr = train["hypothesis"].values
    Xc_tr = Xp_tr + " <SEP> " + Xh_tr
    Xp_te = test["premise"].values
    Xh_te = test["hypothesis"].values
    Xc_te = Xp_te + " <SEP> " + Xh_te

    print("Building handcrafted features...", flush=True)
    hc_tr = handcraft(train)
    hc_te = handcraft(test)
    scaler = StandardScaler()
    hc_tr_s = scaler.fit_transform(hc_tr)
    hc_te_s = scaler.transform(hc_te)

    print("Fitting TF-IDF vectorizers...", flush=True)
    vchar, vchar2, vchar4, vword = build_vectorizers()
    all_text = list(Xp_tr) + list(Xh_tr)
    vchar.fit(all_text)
    vchar2.fit(all_text)
    vchar4.fit(all_text)
    vword.fit(all_text)

    def featurize(Xp, Xh, Xc):
        fp = vchar.transform(Xp)
        fh = vchar.transform(Xh)
        fc = vchar.transform(Xc)
        fp2 = vchar2.transform(Xp)
        fh2 = vchar2.transform(Xh)
        fp4 = vchar4.transform(Xp)
        fh4 = vchar4.transform(Xh)
        fw = vword.transform(Xc)
        return hstack([fp, fh, fc, fp2, fh2, fp4, fh4, fw]).tocsr()

    X_tr = featurize(Xp_tr, Xh_tr, Xc_tr)
    X_te = featurize(Xp_te, Xh_te, Xc_te)
    X_tr = hstack([X_tr, csr_matrix(hc_tr_s)]).tocsr()
    X_te = hstack([X_te, csr_matrix(hc_te_s)]).tocsr()
    print(f"X_tr={X_tr.shape} X_te={X_te.shape}", flush=True)

    print("Cross-validating...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    oof = np.zeros((len(y), 3))
    pred_te = np.zeros((len(test), 3))

    models = [
        ("lr", LogisticRegression(
            C=4.0, max_iter=2000, solver="liblinear", random_state=RNG)),
        ("svc", CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=RNG, max_iter=5000), cv=3)),
    ]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        Xt, yt = X_tr[tr_idx], y[tr_idx]
        Xv, yv = X_tr[va_idx], y[va_idx]
        fold_oof = np.zeros((len(va_idx), 3))
        fold_te = np.zeros((len(test), 3))
        for name, mdl in models:
            mdl.fit(Xt, yt)
            if hasattr(mdl, "predict_proba"):
                p = mdl.predict_proba(Xv)
                pt = mdl.predict_proba(X_te)
            else:
                p = mdl.decision_function(Xv)
                pt = mdl.decision_function(X_te)
                p = (p - p.min(axis=1, keepdims=True))
                p = p / p.sum(axis=1, keepdims=True)
                pt = (pt - pt.min(axis=1, keepdims=True))
                pt = pt / pt.sum(axis=1, keepdims=True)
            fold_oof += p
            fold_te += pt
        fold_oof /= len(models)
        fold_te /= len(models)
        oof[va_idx] = fold_oof
        pred_te += fold_te / skf.n_splits
        acc = (fold_oof.argmax(1) == yv).mean()
        print(f"  fold {fold}: acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    cv_acc = (oof.argmax(1) == y).mean()
    print(f"CV accuracy: {cv_acc:.4f}", flush=True)

    print("Full-fit final model...", flush=True)
    final = np.zeros((len(test), 3))
    for name, mdl in models:
        mdl.fit(X_tr, y)
        if hasattr(mdl, "predict_proba"):
            final += mdl.predict_proba(X_te)
        else:
            d = mdl.decision_function(X_te)
            d = (d - d.min(axis=1, keepdims=True))
            d = d / d.sum(axis=1, keepdims=True)
            final += d
    final /= len(models)
    blend = 0.5 * pred_te + 0.5 * final

    pred_idx = blend.argmax(1)
    pred_labels = [LABELS[i] for i in pred_idx]

    sub = pd.DataFrame({"id": test["id"], "label": pred_labels})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(sub)} rows) in {time.time()-t0:.0f}s", flush=True)
    print("Label distribution:")
    print(sub["label"].value_counts())


if __name__ == "__main__":
    main()
