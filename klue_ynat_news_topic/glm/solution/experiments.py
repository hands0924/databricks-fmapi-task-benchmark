import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["label"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def cv_score(vec, model, X_text, y, fit_test=None):
    Xtr = vec.fit_transform(X_text)
    Xte = None
    if fit_test is not None:
        Xte = vec.transform(fit_test)
    oof = np.zeros(len(y), dtype=object)
    for tr_idx, va_idx in skf.split(Xtr, y):
        m = model
        try:
            m.fit(Xtr[tr_idx], y[tr_idx])
        except Exception:
            m = type(model)(**model.get_params())
            m.fit(Xtr[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(Xtr[va_idx])
    return f1_score(y, oof, average="macro"), Xtr, Xte

# Exp1: char_wb ngram range tuning with LogReg
for ng in [(2,4),(2,5),(2,6),(3,5),(3,6)]:
    vec = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=ng, min_df=2, max_df=0.95)
    s,_,_ = cv_score(vec, LogisticRegression(max_iter=2000, C=3.0, solver="liblinear", random_state=SEED), train["title"].astype(str), y)
    print(f"char_wb {ng} LogReg C3: {s:.4f}")

# Exp2: C tuning
for C in [1.0, 3.0, 5.0, 10.0]:
    vec = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(2,5), min_df=2, max_df=0.95)
    s,_,_ = cv_score(vec, LogisticRegression(max_iter=2000, C=C, solver="liblinear", random_state=SEED), train["title"].astype(str), y)
    print(f"char_wb (2,5) LogReg C={C}: {s:.4f}")

# Exp3: LinearSVC
vec = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(2,5), min_df=2, max_df=0.95)
s,_,_ = cv_score(vec, LinearSVC(C=1.0, random_state=SEED, max_iter=3000), train["title"].astype(str), y)
print(f"char_wb (2,5) LinearSVC C1: {s:.4f}")

# Exp4: word-level (whitespace) char + char_wb hybrid - concatenate features
from scipy.sparse import hstack
def combined_cv(C=3.0):
    vec_c = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(2,5), min_df=2, max_df=0.95)
    Xc = vec_c.fit_transform(train["title"].astype(str))
    oof = np.zeros(len(y), dtype=object)
    for tr_idx, va_idx in skf.split(Xc, y):
        clf = LogisticRegression(max_iter=2000, C=C, solver="liblinear", random_state=SEED)
        clf.fit(Xc[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict(Xc[va_idx])
    return f1_score(y, oof, average="macro")
print(f"char_wb only: {combined_cv():.4f}")
