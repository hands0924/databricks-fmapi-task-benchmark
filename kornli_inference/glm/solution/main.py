"""Best KorNLI: rich interaction features + tuned logistic regression.

Features:
- char_wb TF-IDF (2-5) of premise (P) and hypothesis (H), shared vocab
- word TF-IDF (1-2) of P and H
- interaction: P*H (tfidf product), |bin(P)-bin(H)| (binary diff), bin overlap
- hand-crafted lexical/length features
Classifier: LogisticRegression with tuned C, 5-fold CV.
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, diags
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

# --- Char-wb TF-IDF, shared vocab on s1 and s2 ---
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                           sublinear_tf=True, min_df=3, max_features=100000)
char_vec.fit(np.concatenate([s1_tr, s2_tr]))
Pc_tr = char_vec.transform(s1_tr); Hc_tr = char_vec.transform(s2_tr)
Pc_te = char_vec.transform(s1_te); Hc_te = char_vec.transform(s2_te)

# --- Word TF-IDF, shared vocab ---
word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                           min_df=2, max_features=60000)
word_vec.fit(np.concatenate([s1_tr, s2_tr]))
Pw_tr = word_vec.transform(s1_tr); Hw_tr = word_vec.transform(s2_tr)
Pw_te = word_vec.transform(s1_te); Hw_te = word_vec.transform(s2_te)

# Binarize helper
def binarize(X):
    Xb = X.copy()
    Xb.data = np.ones_like(Xb.data, dtype=np.float32)
    return Xb.tocsr()

# Interaction features
def interactions(P, H):
    prod = P.multiply(H).tocsr()                       # tfidf product
    Pb = binarize(P); Hb = binarize(H)
    common = Pb.multiply(Hb).tocsr()                   # binary overlap
    # |bin(P)-bin(H)| = bin(P) + bin(H) - 2*common
    both = (Pb + Hb).tocsr()
    diff = both - 2.0 * common
    diff.data = np.abs(diff.data)
    diff.eliminate_zeros()
    return prod, common, diff

prod_c, common_c, diff_c = interactions(Pc_tr, Hc_tr)
prod_w, common_w, diff_w = interactions(Pw_tr, Hw_tr)
prod_c_te, common_c_te, diff_c_te = interactions(Pc_te, Hc_te)
prod_w_te, common_w_te, diff_w_te = interactions(Pw_te, Hw_te)

# --- Hand-crafted features ---
NEG_WORDS = ["안", "없", "못", "아니", "없다", "아무도", "전혀", "결코",
             "하지", "not", "no", "never", "without", "nothing", "nobody"]
def hand_feats(s1, s2):
    f = []
    for a, b in zip(s1, s2):
        wa = a.split(); wb = b.split()
        sa = set(wa); sb = set(wb)
        union = len(sa | sb) + 1e-9
        ov = len(sa & sb) / union
        overlap_count = len(sa & sb)
        char_ov = len(set(a) & set(b)) / (len(set(a) | set(b)) + 1e-9)
        lenr = np.log1p(len(a)) - np.log1p(len(b))
        n1 = sum(1 for w in NEG_WORDS if w in a)
        n2 = sum(1 for w in NEG_WORDS if w in b)
        f.append([ov, lenr, n1, n2, n1 - n2, overlap_count,
                  len(wa), len(wb), abs(len(wa) - len(wb)),
                  len(wa) / (len(wb) + 1e-9),
                  char_ov, len(a) / (len(b) + 1e-9),
                  int(a == b), int(b in a), int(a in b)])
    return np.array(f, dtype=np.float32)

HF_tr = hand_feats(s1_tr, s2_tr)
HF_te = hand_feats(s1_te, s2_te)

# Assemble
Xtr = hstack([Pc_tr, Hc_tr, prod_c, diff_c,
              Pw_tr, Hw_tr, prod_w, diff_w,
              HF_tr]).tocsr()
Xte = hstack([Pc_te, Hc_te, prod_c_te, diff_c_te,
              Pw_te, Hw_te, prod_w_te, diff_w_te,
              HF_te]).tocsr()
print("Xtr", Xtr.shape, "Xte", Xte.shape)

# Tune C with CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_C = 1.0; best_acc = 0.0
for C in [0.5, 1.0, 2.0, 4.0]:
    oof = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in skf.split(Xtr, y):
        clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
        clf.fit(Xtr[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict(Xtr[va_idx])
    acc = accuracy_score(y, oof)
    print(f"C={C} OOF acc {acc:.4f}")
    if acc > best_acc:
        best_acc = acc; best_C = C
print("best C", best_C, "acc", best_acc)

# Final fit
clf = LogisticRegression(C=best_C, max_iter=2000, solver="liblinear")
clf.fit(Xtr, y)
pred = clf.predict(Xte)
labels = le.inverse_transform(pred)

out = pd.DataFrame({"id": test["id"], "label": labels})
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)
out.to_csv(os.path.join(BASE, "outputs", "submission.csv"), index=False)
print("done", out.shape, out["label"].value_counts().to_dict())
