"""
t3_ynat — Korean news topic classification (KLUE-YNAT), macro-F1.

Approach (sklearn only, no external data / pretrained weights):
  Level 1: 13 diverse base models over TF-IDF views of the raw title
           (char_wb n-grams, char n-grams, word n-grams) — LinearSVC,
           LogisticRegression, RidgeClassifier, ComplementNB, SGD,
           NB-SVM, kNN-cosine, Rocchio centroid.
           5-fold OOF decision scores are collected as meta-features;
           each model is also refit on the full train set to score test.
  Level 2: per-model z-normalised score matrices are concatenated and fed to
           two meta-learners (LogisticRegression C=0.1, LinearSVC C=0.05);
           their z-normalised decision functions are averaged, argmax = label.

OOF macro-F1 (5-fold): best single model 0.8437 -> stacked ensemble 0.8501

Usage:  python solution/run.py            (from the task root directory)
Runtime: ~10 min on 4 CPU cores. Writes outputs/submission.csv
"""
import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import LinearSVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
N_FOLDS = 5
N_JOBS = 4

# ---------------------------------------------------------------- data
train = pd.read_csv(os.path.join(ROOT, "train.csv"))
test = pd.read_csv(os.path.join(ROOT, "test.csv"))
X, X_test = train["title"].values, test["title"].values
le = LabelEncoder()
y = le.fit_transform(train["label"].values)
NC = len(le.classes_)

# ------------------------------------------------------- feature views
VIEWS = {
    "cwb14": dict(analyzer="char_wb", ngram_range=(1, 4), min_df=2, sublinear_tf=True),
    "cwb26": dict(analyzer="char_wb", ngram_range=(2, 6), min_df=2, sublinear_tf=True),
    "cwb15m1": dict(analyzer="char_wb", ngram_range=(1, 5), min_df=1, sublinear_tf=True),
    "ch25": dict(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    "w12": dict(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b"),
}

# ------------------------------------------------- level-1 base models
SK_MODELS = [
    ("svc03_cwb14", "cwb14", lambda: LinearSVC(C=0.3)),
    ("svc05_cwb26", "cwb26", lambda: LinearSVC(C=0.5)),
    ("lr4_cwb14", "cwb14", lambda: LogisticRegression(C=4, max_iter=3000, n_jobs=N_JOBS)),
    ("ridge1_cwb15m1", "cwb15m1", lambda: RidgeClassifier(alpha=1.0)),
    ("cnb_cwb15m1", "cwb15m1", lambda: ComplementNB(alpha=0.3)),
    ("svc03_ch25", "ch25", lambda: LinearSVC(C=0.3)),
    ("svc05_w12", "w12", lambda: LinearSVC(C=0.5)),
    ("lr8_w12", "w12", lambda: LogisticRegression(C=8, max_iter=3000, n_jobs=N_JOBS)),
    ("cnb_w12", "w12", lambda: ComplementNB(alpha=0.3)),
    ("sgd_cwb14", "cwb14", lambda: SGDClassifier(loss="modified_huber", alpha=1e-5,
                                                 max_iter=50, tol=1e-4, random_state=0)),
]
CUSTOM = ["knn_cwb14", "rocchio_cwb14", "nbsvm_cwb14"]  # all on the cwb14 view
NAMES = [m[0] for m in SK_MODELS] + CUSTOM


def model_scores(clf, M):
    """Uniform 'higher is better' score matrix (n x NC)."""
    if hasattr(clf, "predict_proba"):
        return np.log(np.clip(clf.predict_proba(M), 1e-9, None))
    return clf.decision_function(M)


def rocchio_scores(A, ya, B):
    """Cosine similarity between each doc and the L2-normalised class centroids."""
    cent = normalize(np.vstack([np.asarray(A[ya == c].mean(0)).ravel() for c in range(NC)]))
    return np.asarray(B @ cent.T)


def nbsvm_scores(A, ya, B, alpha=1.0, C=0.3):
    """One-vs-rest NB-SVM: tf-idf features rescaled by NB log-count ratios."""
    Ab = (A > 0).astype(np.float64)
    out = []
    for c in range(NC):
        pos = np.asarray(Ab[ya == c].sum(0)).ravel() + alpha
        neg = np.asarray(Ab[ya != c].sum(0)).ravel() + alpha
        r = sp.diags(np.log((pos / pos.sum()) / (neg / neg.sum())))
        clf = LinearSVC(C=C).fit(A @ r, (ya == c).astype(int))
        out.append(clf.decision_function(B @ r))
    return np.vstack(out).T


def level1(Xa, ya, Xb):
    """Fit every base model on (Xa, ya); return dict of score matrices for Xb."""
    views = {}
    for vn, kw in VIEWS.items():
        v = TfidfVectorizer(**kw)
        views[vn] = (v.fit_transform(Xa), v.transform(Xb))
    out = {}
    for name, vn, make in SK_MODELS:
        A, B = views[vn]
        out[name] = model_scores(make().fit(A, ya), B)
    A, B = views["cwb14"]
    knn = KNeighborsClassifier(n_neighbors=40, metric="cosine", algorithm="brute",
                               weights="distance", n_jobs=N_JOBS).fit(A, ya)
    out["knn_cwb14"] = np.log(np.clip(knn.predict_proba(B), 1e-6, None))
    out["rocchio_cwb14"] = rocchio_scores(A, ya, B)
    out["nbsvm_cwb14"] = nbsvm_scores(A, ya, B)
    return out


# ------------------------------------------- OOF + full-train level-1
t0 = time.time()
oof = {n: np.zeros((len(X), NC)) for n in NAMES}
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
for k, (tri, vai) in enumerate(folds):
    res = level1(X[tri], y[tri], X[vai])
    for n in NAMES:
        oof[n][vai] = res[n]
    print(f"[level1] fold {k} done ({time.time() - t0:.0f}s)", flush=True)
test_scores = level1(X, y, X_test)
print(f"[level1] full-train fit done ({time.time() - t0:.0f}s)", flush=True)
for n in NAMES:
    print(f"    {n:16s} oof macro-F1 = {f1_score(y, oof[n].argmax(1), average='macro'):.5f}")

# ------------------------------------------------------- level-2 stack
Z, ZT = [], []
for n in NAMES:
    mu, sd = oof[n].mean(), oof[n].std()
    Z.append((oof[n] - mu) / sd)
    ZT.append((test_scores[n] - mu) / sd)
F, FT = np.hstack(Z), np.hstack(ZT)

METAS = [lambda: LogisticRegression(C=0.1, max_iter=3000, n_jobs=N_JOBS),
         lambda: LinearSVC(C=0.05)]


def znorm(d):
    return (d - d.mean()) / d.std()


# honest CV estimate of the stacked ensemble
S = np.zeros((len(y), NC))
for tri, vai in StratifiedKFold(N_FOLDS, shuffle=True, random_state=7).split(F, y):
    for make in METAS:
        S[vai] += znorm(make().fit(F[tri], y[tri]).decision_function(F[vai]))
print(f"[level2] stacked CV macro-F1 = {f1_score(y, S.argmax(1), average='macro'):.5f}"
      f" ({time.time() - t0:.0f}s)", flush=True)

# final: metas refit on all OOF meta-features, averaged on test
ST = np.zeros((len(X_test), NC))
for make in METAS:
    ST += znorm(make().fit(F, y).decision_function(FT))
pred = le.classes_[ST.argmax(1)]

os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
sub = pd.DataFrame({"id": test["id"].values, "label": pred})
assert len(sub) == len(test) and sub["id"].is_unique
sub.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
print(sub["label"].value_counts().to_string())
print(f"wrote outputs/submission.csv ({time.time() - t0:.0f}s)")
