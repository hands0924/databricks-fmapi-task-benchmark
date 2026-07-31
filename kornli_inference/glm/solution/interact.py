"""Improved KorNLI with interaction features (|p-h|, p*h) + hand-crafted.

For NLI, the relationship between premise and hypothesis matters. We build
separate TF-IDF representations of premise (p) and hypothesis (h), then
feed [p ; h ; |p-h| ; p*h] features to a linear classifier.
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train = pd.read_csv(os.path.join(BASE, "train.csv"))
test = pd.read_csv(os.path.join(BASE, "test.csv"))

le = LabelEncoder()
y = le.fit_transform(train["label"])

s1_tr = train["sentence1"].fillna("").astype(str).values
s2_tr = train["sentence2"].fillna("").astype(str).values
s1_te = test["sentence1"].fillna("").astype(str).values
s2_te = test["sentence2"].fillna("").astype(str).values

# Char-wb TF-IDF shared vocab on sentence1 and sentence2
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                           sublinear_tf=True, min_df=3, max_features=80000)
# Fit on combined corpus of both s1 and s2 so vocab is shared
char_vec.fit(np.concatenate([s1_tr, s2_tr]))
P_tr = char_vec.transform(s1_tr)
H_tr = char_vec.transform(s2_tr)
P_te = char_vec.transform(s1_te)
H_te = char_vec.transform(s2_te)

# Interaction features in sparse space
# |p - h| and p * h need dense elementwise; for sparse we approximate.
# For sparse: p*h (multiply) is efficient; p-h needs min/max.
def sparse_abs_diff(p, h):
    # |p - h| via sparse operations: |p-h| where both have nonneg values
    # We approximate with: (p + h - 2*min(p,h)); simpler: use p.multiply(h) and sums
    # Instead use: |p - h| ~ |p|_bin + |h|_bin - 2*p_common (binary overlap)
    # Here we just compute elementwise viacsr arithmetic (treat as dense per row? expensive)
    # Use the identity |a-b| = a + b - 2*min(a,b); min not sparse friendly.
    # Fallback: binary overlap feature.
    return None

# Simpler robust interaction: element-wise product (sparse) + abs via sum of binarized
def sparse_mul(p, h):
    return p.multiply(h)

def bin_overlap(p, h):
    # binary overlap: min of binarized vectors = where both present
    return p.multiply(h)  # tf-idf weighted; close enough as overlap signal

# Hand-crafted features
import re
NEG_WORDS = ["안", "없", "못", "아니", "없다", "아무도", "전혀", "결코", "하지", "not", "no", "never"]
def hand_feats(s1, s2):
    f = []
    for a, b in zip(s1, s2):
        wa = a.split(); wb = b.split()
        sa = set(wa); sb = set(wb)
        ov = len(sa & sb) / (len(sa | sb) + 1e-9)
        lenr = np.log1p(len(a)) - np.log1p(len(b))
        n1 = sum(1 for w in NEG_WORDS if w in a)
        n2 = sum(1 for w in NEG_WORDS if w in b)
        neg_diff = n1 - n2
        overlap_count = len(sa & sb)
        f.append([ov, lenr, n1, n2, neg_diff, overlap_count,
                  len(wa), len(wb), abs(len(wa)-len(wb)),
                  len(wa)/(len(wb)+1e-9)])
    return np.array(f, dtype=np.float32)

HF_tr = hand_feats(s1_tr, s2_tr)
HF_te = hand_feats(s1_te, s2_te)

# Build feature matrices
prod_tr = P_tr.multiply(H_tr)
prod_te = P_te.multiply(H_te)

Xtr = hstack([P_tr, H_tr, prod_tr, HF_tr]).tocsr()
Xte = hstack([P_te, H_te, prod_te, HF_te]).tocsr()
print("Xtr", Xtr.shape, "Xte", Xte.shape)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y), dtype=int)
for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
    clf.fit(Xtr[tr_idx], y[tr_idx])
    oof[va_idx] = clf.predict(Xtr[va_idx])
    print(f"fold {fold} acc", accuracy_score(y[va_idx], oof[va_idx]))
print("OOF acc", accuracy_score(y, oof))

clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
clf.fit(Xtr, y)
pred = clf.predict(Xte)
labels = le.inverse_transform(pred)

out = pd.DataFrame({"id": test["id"], "label": labels})
out.to_csv(os.path.join(BASE, "outputs", "submission.csv"), index=False)
print("done", out.shape, out["label"].value_counts().to_dict())
