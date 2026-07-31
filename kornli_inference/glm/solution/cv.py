"""Improved KorNLI: word+char TF-IDF features + logistic regression ensemble.

Strategy:
- Build TF-IDF on (premise + [SEP] + hypothesis) using both word (whitespace)
  and char-wb n-grams; concatenate them.
- Also build separate TF-IDF on premise and on hypothesis and feed abs-diff
  of mean-pooled probabilities is overkill here; instead we keep it simple
  and robust: concatenate char and word vectors and train a LogisticRegression.
- Use 5-fold CV to calibrate C and to estimate accuracy.
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
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

def prep(df):
    s1 = df["sentence1"].fillna("").astype(str).values
    s2 = df["sentence2"].fillna("").astype(str).values
    return s1, s2

tr_s1, tr_s2 = prep(train)
te_s1, te_s2 = prep(test)

tr_pair = np.array([a + " [SEP] " + b for a, b in zip(tr_s1, tr_s2)])
te_pair = np.array([a + " [SEP] " + b for a, b in zip(te_s1, te_s2)])

# Char-wb n-grams on pair
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                           sublinear_tf=True, min_df=3, max_features=200000)
Xtr_char = char_vec.fit_transform(tr_pair)
Xte_char = char_vec.transform(te_pair)

# Word n-grams on pair
word_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                           sublinear_tf=True, min_df=3, max_features=150000)
# Note: analyzer='char' is a fallback; below we use whitespace word n-grams
word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                           min_df=2, max_features=100000)
Xtr_word = word_vec.fit_transform(tr_pair)
Xte_word = word_vec.transform(te_pair)

Xtr = hstack([Xtr_char, Xtr_word]).tocsr()
Xte = hstack([Xte_char, Xte_word]).tocsr()

print("Xtr", Xtr.shape, "Xte", Xte.shape)

# 5-fold CV to estimate accuracy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y), dtype=int)
for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
    clf = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear")
    clf.fit(Xtr[tr_idx], y[tr_idx])
    oof[va_idx] = clf.predict(Xtr[va_idx])
    print(f"fold {fold} acc", accuracy_score(y[va_idx], oof[va_idx]))
print("OOF acc", accuracy_score(y, oof))

# Final fit on all data
clf = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear")
clf.fit(Xtr, y)
pred = clf.predict(Xte)
labels = le.inverse_transform(pred)

out = pd.DataFrame({"id": test["id"], "label": labels})
out.to_csv(os.path.join(BASE, "outputs", "submission.csv"), index=False)
print("done", out.shape, out["label"].value_counts().to_dict())
