import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
train = pd.read_csv("train.csv")
y = train["label"].values
titles = train["title"].astype(str).values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def run_cv(X, C=3.0):
    oof = np.zeros(len(y), dtype=object)
    for tr_idx, va_idx in skf.split(X, y):
        m = LogisticRegression(max_iter=2000, C=C, solver="liblinear", random_state=SEED)
        m.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X[va_idx])
    return f1_score(y, oof, average="macro")

vec_a = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(2,4), min_df=2, max_df=0.95)
Xa = vec_a.fit_transform(titles)
print("char_wb(2,4):", run_cv(Xa))

vec_b = TfidfVectorizer(sublinear_tf=True, analyzer="char", ngram_range=(2,5), min_df=2, max_df=0.95)
Xb = vec_b.fit_transform(titles)
print("char(2,5):", run_cv(Xb))

Xab = hstack([Xa, Xb]).tocsr()
print("char_wb(2,4)+char(2,5):", run_cv(Xab))
print("char_wb(2,4)+char(2,5) C5:", run_cv(Xab, C=5.0))
