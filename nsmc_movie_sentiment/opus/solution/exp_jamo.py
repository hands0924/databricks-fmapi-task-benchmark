"""Holdout experiments: jamo-level char n-grams and variants."""
import time, sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from korean import to_jamo_list

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

DATA = "/tmp/kmle/M1_t4_nsmc_full_20260730_143411/task"
tr = pd.read_csv(f"{DATA}/train.csv")
tr["document"] = tr["document"].astype(str)

Xtr, Xva, ytr, yva = train_test_split(
    tr["document"].values, tr["label"].values, test_size=0.15, random_state=42,
    stratify=tr["label"].values
)
Jtr = to_jamo_list(Xtr)
Jva = to_jamo_list(Xva)
log("jamo done", Jtr[0][:60])


def run(name, texts_tr, texts_va, vec, clfs):
    A = vec.fit_transform(texts_tr)
    B = vec.transform(texts_va)
    log(f"{name}: nfeat={A.shape[1]}")
    best = None
    for cname, clf in clfs:
        t = time.time()
        clf.fit(A, ytr)
        acc = accuracy_score(yva, clf.predict(B))
        log(f"  {name} | {cname}: acc={acc:.5f} ({time.time()-t:.0f}s)")
    return best


if __name__ == "__main__":
    stage = sys.argv[1]
    LRS = lambda: [
        ("LR C=2", LogisticRegression(C=2, max_iter=3000, solver="liblinear")),
        ("LR C=4", LogisticRegression(C=4, max_iter=3000, solver="liblinear")),
        ("LR C=8", LogisticRegression(C=8, max_iter=3000, solver="liblinear")),
    ]

    if stage == "jamo":
        run("jamo(2,5)md3", Jtr, Jva,
            TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, sublinear_tf=True), LRS())
    elif stage == "jamo6":
        run("jamo(2,6)md3", Jtr, Jva,
            TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=3, sublinear_tf=True), LRS())
    elif stage == "jamo7":
        run("jamo(2,7)md3", Jtr, Jva,
            TfidfVectorizer(analyzer="char", ngram_range=(2, 7), min_df=3, sublinear_tf=True), LRS())
    elif stage == "char6":
        run("charwb(1,6)md2", Xtr, Xva,
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 6), min_df=2, sublinear_tf=True), LRS())
    elif stage == "char_plain":
        run("char(2,5)md3", Xtr, Xva,
            TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, sublinear_tf=True), LRS())
