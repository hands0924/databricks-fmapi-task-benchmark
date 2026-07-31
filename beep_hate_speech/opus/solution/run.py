"""
t8_beep — Korean hate-speech classification (BEEP!), 3 classes: none / offensive / hate.
Metric: macro F1.

Approach (no internet, no external data, train.csv only):
  * text normalisation (lowercase, url strip, collapse 3+ char repeats, squeeze spaces)
  * Hangul jamo decomposition; the model input is  "<normalised text> ␟ <jamo text>"
    so that a single char_wb TF-IDF sees both syllable-level and jamo-level n-grams
    (robust to the slang / typo / variant spellings typical of this dataset)
  * NB-weighted one-vs-rest logistic regression (NBSVM-style) as the base learner
  * uniform probability average over a small, diverse set of such learners

Usage:
    python solution/run.py            # CV report + fit on all train + write submission
    python solution/run.py --no-cv    # skip the CV report (faster)
"""
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nblr import NBLR  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["hate", "none", "offensive"]      # sklearn's sorted class order
CV_SEEDS = [42, 7, 2024]
N_JOBS = 4

# ----------------------------------------------------------------------------- text prep
CHO = list("\u3131\u3132\u3134\u3137\u3138\u3139\u3141\u3142\u3143\u3145"
           "\u3146\u3147\u3148\u3149\u314a\u314b\u314c\u314d\u314e")
JUNG = list("\u314f\u3150\u3151\u3152\u3153\u3154\u3155\u3156\u3157\u3158\u3159"
            "\u315a\u315b\u315c\u315d\u315e\u315f\u3160\u3161\u3162\u3163")
JONG = [""] + list("\u3131\u3132\u3133\u3134\u3135\u3136\u3137\u3139\u313a\u313b\u313c"
                   "\u313d\u313e\u313f\u3140\u3141\u3142\u3144\u3145\u3146\u3147\u3148"
                   "\u314a\u314b\u314c\u314d\u314e")
_S_BASE, _N_JUNG, _N_JONG = 0xAC00, 21, 28

_URL = re.compile(r"https?://\S+|www\.\S+")
_REPEAT = re.compile(r"(.)\1{2,}")
_SPACE = re.compile(r"\s+")


def normalize(text):
    t = str(text).lower()
    t = _URL.sub(" url ", t)
    t = _REPEAT.sub(r"\1\1", t)       # ㅋㅋㅋㅋㅋ -> ㅋㅋ
    return _SPACE.sub(" ", t).strip()


def to_jamo(text):
    out = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            idx = code - _S_BASE
            out.append(CHO[idx // (_N_JUNG * _N_JONG)])
            out.append(JUNG[(idx % (_N_JUNG * _N_JONG)) // _N_JONG])
            out.append(JONG[idx % _N_JONG])
        else:
            out.append(ch)
    return "".join(out)


def prepare(series):
    """-> dict of the three text representations used by the ensemble members."""
    norm = series.map(normalize).values
    jamo = np.array([to_jamo(t) for t in norm])
    both = np.array([a + " \u241f " + b for a, b in zip(norm, jamo)])
    return {"norm": norm, "jamo": jamo, "both": both}


# ----------------------------------------------------------------------------- members
def member(rep, analyzer, ngram, C, alpha, min_df=2):
    return rep, lambda: Pipeline([
        ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                                  min_df=min_df, sublinear_tf=True)),
        ("nblr", NBLR(C=C, alpha=alpha)),
    ])


MEMBERS = {
    "both_cw13_a015": member("both", "char_wb", (1, 3), 2.0, 0.15),
    "both_cw13_a025": member("both", "char_wb", (1, 3), 2.0, 0.25),
    "both_cw14_C3":   member("both", "char_wb", (1, 4), 3.0, 0.25),
    "both_cw13_C15":  member("both", "char_wb", (1, 3), 1.5, 0.25),
    "norm_cw13":      member("norm", "char_wb", (1, 3), 2.0, 0.25),
    "jamo_cw25":      member("jamo", "char_wb", (2, 5), 2.0, 0.25),
}
BOTH_ONLY = ["both_cw13_a015", "both_cw13_a025", "both_cw14_C3", "both_cw13_C15"]
SUBSETS = {                       # reported by the CV block for transparency
    "ens_both4": BOTH_ONLY,
    "ens_both4+norm": BOTH_ONLY + ["norm_cw13"],
    "ens_all6": list(MEMBERS),
}
ENSEMBLE = BOTH_ONLY              # uniform probability average over these members


# ----------------------------------------------------------------------------- CV report
def cv_report(reps, y):
    folds = {s: list(StratifiedKFold(5, shuffle=True, random_state=s).split(y, y))
             for s in CV_SEEDS}
    yi = np.array([CLASSES.index(v) for v in y])

    def job(name, seed, fi):
        rep, factory = MEMBERS[name]
        X = reps[rep]
        trn, val = folds[seed][fi]
        pipe = factory()
        pipe.fit(X[trn], y[trn])
        assert list(pipe.classes_) == CLASSES
        return name, seed, val, pipe.predict_proba(X[val])

    tasks = [(n, s, i) for n in MEMBERS for s in CV_SEEDS for i in range(5)]
    out = Parallel(n_jobs=N_JOBS)(delayed(job)(*t) for t in tasks)

    oof = {(n, s): np.zeros((len(y), 3)) for n in MEMBERS for s in CV_SEEDS}
    for name, seed, val, pr in out:
        oof[(name, seed)][val] = pr

    def macro(prob):
        return f1_score(yi, prob.argmax(1), average="macro")

    print("\n--- 3x5-fold CV macro F1 ---")
    for n in MEMBERS:
        sc = [macro(oof[(n, s)]) for s in CV_SEEDS]
        print(f"  {n:16s} {np.mean(sc):.4f}  {[round(x, 4) for x in sc]}")
    for label, subset in SUBSETS.items():
        sc = [macro(np.mean([oof[(n, s)] for n in subset], axis=0)) for s in CV_SEEDS]
        mark = " <== used" if subset == ENSEMBLE else ""
        print(f"  {label:16s} {np.mean(sc):.4f}  {[round(x, 4) for x in sc]}{mark}")
    ens = [macro(np.mean([oof[(n, s)] for n in ENSEMBLE], axis=0)) for s in CV_SEEDS]
    return np.mean(ens)


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    do_cv = "--no-cv" not in sys.argv
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    y = tr.label.values
    reps_tr, reps_te = prepare(tr.comment), prepare(te.comment)
    print(f"train={tr.shape} test={te.shape}")

    if do_cv:
        cv_report(reps_tr, y)

    def fit_predict(name):
        rep, factory = MEMBERS[name]
        pipe = factory()
        pipe.fit(reps_tr[rep], y)
        assert list(pipe.classes_) == CLASSES
        return pipe.predict_proba(reps_te[rep])

    probs = Parallel(n_jobs=N_JOBS)(delayed(fit_predict)(n) for n in ENSEMBLE)
    pred = np.mean(probs, axis=0).argmax(1)
    labels = [CLASSES[i] for i in pred]

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    sub = pd.DataFrame({"id": te.id.values, "label": labels})
    # sanity checks against the required format
    ss = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    assert list(sub.columns) == list(ss.columns)
    assert len(sub) == len(ss) and sub.id.is_unique
    assert set(sub.id) == set(ss.id)
    assert set(sub.label) <= set(CLASSES)
    sub.to_csv(os.path.join(ROOT, "outputs/submission.csv"), index=False)
    print("\nwrote outputs/submission.csv")
    print(sub.label.value_counts().to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
