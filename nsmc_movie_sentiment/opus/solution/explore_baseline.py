"""Quick holdout experiments for NSMC sentiment classification (sklearn only)."""
import time, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

DATA = "/tmp/kmle/M1_t4_nsmc_full_20260730_143411/task"
tr = pd.read_csv(f"{DATA}/train.csv")
tr["document"] = tr["document"].astype(str)

Xtr, Xva, ytr, yva = train_test_split(
    tr["document"].values, tr["label"].values, test_size=0.15, random_state=42, stratify=tr["label"].values
)
log("split", Xtr.shape, Xva.shape)


def eval_feats(name, vecs, clfs):
    mats_tr, mats_va = [], []
    for v in vecs:
        mats_tr.append(v.fit_transform(Xtr))
        mats_va.append(v.transform(Xva))
    A = hstack(mats_tr).tocsr()
    B = hstack(mats_va).tocsr()
    log(f"{name}: features={A.shape[1]}")
    for cname, clf in clfs:
        t = time.time()
        clf.fit(A, ytr)
        p = clf.predict(B)
        log(f"  {name} | {cname} : acc={accuracy_score(yva, p):.5f}  ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "word"):
        eval_feats(
            "word12",
            [TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)],
            [
                ("LR C=4", LogisticRegression(C=4, max_iter=2000, solver="liblinear")),
                ("LinearSVC C=0.5", LinearSVC(C=0.5)),
            ],
        )

    if which in ("all", "char"):
        eval_feats(
            "charwb25",
            [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)],
            [
                ("LR C=4", LogisticRegression(C=4, max_iter=2000, solver="liblinear")),
                ("LinearSVC C=0.5", LinearSVC(C=0.5)),
            ],
        )

    if which in ("all", "combo"):
        eval_feats(
            "word12+charwb25",
            [
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True),
            ],
            [
                ("LR C=4", LogisticRegression(C=4, max_iter=2000, solver="liblinear")),
                ("LR C=8", LogisticRegression(C=8, max_iter=2000, solver="liblinear")),
                ("LinearSVC C=0.5", LinearSVC(C=0.5)),
                ("LinearSVC C=1", LinearSVC(C=1)),
            ],
        )
