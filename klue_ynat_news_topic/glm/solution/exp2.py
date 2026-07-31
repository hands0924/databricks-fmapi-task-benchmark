import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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

# Jamo decomposition for Korean - may help char n-grams
def to_jamo(text):
    try:
        import jamo
        return jamo.h2jamo(text)
    except Exception:
        return text

try:
    import jamo
    print("jamo available")
    titles_jamo = [jamo.h2jamo(t) for t in titles]
    test_jamo = [jamo.h2jamo(t) for t in test_titles]
    has_jamo = True
except Exception as e:
    print("no jamo:", e)
    has_jamo = False

def run_cv(X, model_factory):
    oof = np.zeros(len(y), dtype=object)
    for tr_idx, va_idx in skf.split(X, y):
        m = model_factory()
        m.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X[va_idx])
    return f1_score(y, oof, average="macro"), oof

# char_wb (2,4) best so far. Try combining char_wb + word char
vec1 = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(2,4), min_df=2, max_df=0.95)
X1 = vec1.fit_transform(titles)
s1, _ = run_cv(X1, lambda: LogisticRegression(max_iter=2000, C=3.0, solver="liblinear", random_state=SEED))
print(f"char_wb(2,4) LR C3: {s1:.4f}")

# word-level using whitespace analyzer on space-joined tokens
vec_w = TfidfVectorizer(sublinear_tf=True, analyzer="word", ngram_range=(1,2), min_df=2, max_df=0.95, token_pattern=r"\S+")
Xw = vec_w.fit_transform(titles)
sw, _ = run_cv(Xw, lambda: LogisticRegression(max_iter=2000, C=3.0, solver="liblinear", random_state=SEED))
print(f"word(1,2) LR C3: {sw:.4f}")

# combined char_wb(2,4) + word(1,2)
Xc = hstack([X1, Xw]).tocsr()
sc, _ = run_cv(Xc, lambda: LogisticRegression(max_iter=2000, C=3.0, solver="liblinear", random_state=SEED))
print(f"char_wb(2,4)+word(1,2) LR C3: {sc:.4f}")

# combined with C=5
sc2, _ = run_cv(Xc, lambda: LogisticRegression(max_iter=2000, C=5.0, solver="liblinear", random_state=SEED))
print(f"char_wb(2,4)+word(1,2) LR C5: {sc2:.4f}")

# jamo char_wb
if has_jamo:
    vec_j = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(2,5), min_df=2, max_df=0.95)
    Xj = vec_j.fit_transform(titles_jamo)
    sj, _ = run_cv(Xj, lambda: LogisticRegression(max_iter=2000, C=3.0, solver="liblinear", random_state=SEED))
    print(f"jamo char_wb(2,5) LR C3: {sj:.4f}")
    # combine char_wb original + jamo
    Xcj = hstack([X1, Xj]).tocsr()
    scj, _ = run_cv(Xcj, lambda: LogisticRegression(max_iter=2000, C=3.0, solver="liblinear", random_state=SEED))
    print(f"char_wb(2,4)+jamo(2,5) LR C3: {scj:.4f}")
    Xcwj = hstack([X1, Xw, Xj]).tocsr()
    scwj, _ = run_cv(Xcwj, lambda: LogisticRegression(max_iter=2000, C=3.0, solver="liblinear", random_state=SEED))
    print(f"char_wb(2,4)+word(1,2)+jamo(2,5) LR C3: {scwj:.4f}")
