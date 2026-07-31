import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["label"].values
titles = train["title"].astype(str).values
test_titles = test["title"].astype(str).values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def run_cv(X, model_factory):
    oof = np.zeros(len(y), dtype=object)
    for tr_idx, va_idx in skf.split(X, y):
        m = model_factory()
        m.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X[va_idx])
    return f1_score(y, oof, average="macro")

# char (no word boundary) vs char_wb
for analyzer in ["char", "char_wb"]:
    for ng in [(2,4),(2,5),(1,5)]:
        vec = TfidfVectorizer(sublinear_tf=True, analyzer=analyzer, ngram_range=ng, min_df=2, max_df=0.95)
        X = vec.fit_transform(titles)
        s = run_cv(X, lambda: LogisticRegression(max_iter=2000, C=3.0, solver="liblinear", random_state=SEED))
        print(f"{analyzer} {ng} LR C3: {s:.4f}")

# Combine char_wb(2,4) + char(2,5) (both good, different info)
vec_a = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(2,4), min_df=2, max_df=0.95)
Xa = vec_a.fit_transform(titles)
vec_b = TfidfVectorizer(sublinear_tf=True, analyzer="char", ngram_range=(2,5), min_df=2, max_df=0.95)
Xb = vec_b.fit_transform(titles)
Xab = hstack([Xa, Xb]).tocsr()
for C in [1.0, 3.0, 5.0]:
    s = run_cv(Xab, lambda: LogisticRegression(max_iter=2000, C=C, solver="liblinear", random_state=SEED))
    print(f"char_wb(2,4)+char(2,5) LR C{C}: {s:.4f}")

# Add char(3,6) too
vec_c = TfidfVectorizer(sublinear_tf=True, analyzer="char", ngram_range=(3,6), min_df=2, max_df=0.95)
Xc = vec_c.fit_transform(titles)
Xabc = hstack([Xa, Xb, Xc]).tocsr()
s = run_cv(Xabc, lambda: LogisticRegression(max_iter=2000, C=3.0, solver="liblinear", random_state=SEED))
print(f"char_wb(2,4)+char(2,5)+char(3,6) LR C3: {s:.4f}")

# Try LinearSVC with calibrated probabilities + threshold tuning is overkill; just SVC raw
s = run_cv(Xa, lambda: LinearSVC(C=1.0, random_state=SEED, max_iter=5000))
print(f"char_wb(2,4) SVC C1: {s:.4f}")
s = run_cv(Xa, lambda: LinearSVC(C=3.0, random_state=SEED, max_iter=5000))
print(f"char_wb(2,4) SVC C3: {s:.4f}")
s = run_cv(Xab, lambda: LinearSVC(C=1.0, random_state=SEED, max_iter=5000))
print(f"char_wb(2,4)+char(2,5) SVC C1: {s:.4f}")

# Ensemble: average probabilities of two LR models
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
yle = le.fit_transform(y)
classes = le.classes_

def run_cv_proba(X, model_factory):
    oof = np.zeros((len(y), len(classes)))
    for tr_idx, va_idx in skf.split(X, y):
        m = model_factory()
        m.fit(X[tr_idx], yle[tr_idx])
        oof[va_idx] = m.predict_proba(X[va_idx])
    return oof

pa = run_cv_proba(Xa, lambda: LogisticRegression(max_iter=2000, C=3.0, solver="liblinear", random_state=SEED))
pb = run_cv_proba(Xb, lambda: LogisticRegression(max_iter=2000, C=3.0, solver="liblinear", random_state=SEED))
ens = np.argmax(pa + pb, axis=1)
s = f1_score(y, le.inverse_transform(ens), average="macro")
print(f"ensemble proba char_wb(2,4)+char(2,5): {s:.4f}")
