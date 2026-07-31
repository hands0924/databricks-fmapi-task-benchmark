"""Baseline: TF-IDF (char+word) on premise+hypothesis pairs -> LogisticRegression."""
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train = pd.read_csv(os.path.join(BASE, "train.csv"))
test = pd.read_csv(os.path.join(BASE, "test.csv"))
sample = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))

le = LabelEncoder()
y = le.fit_transform(train["label"])

def prep(df):
    s1 = df["sentence1"].fillna("").astype(str)
    s2 = df["sentence2"].fillna("").astype(str)
    return s1, s2

tr_s1, tr_s2 = prep(train)
te_s1, te_s2 = prep(test)

# Char n-gram TF-IDF on concatenation
def char_feats(s1, s2, tr1, tr2, te1, te2):
    train_txt = (tr1 + " [SEP] " + tr2).values
    test_txt = (te1 + " [SEP] " + te2).values
    vec = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=2, max_features=100000)
    Xtr = vec.fit_transform(train_txt)
    Xte = vec.transform(test_txt)
    return Xtr, Xte

Xtr, Xte = char_feats(tr_s1, tr_s2, tr_s1, tr_s2, te_s1, te_s2)

clf = LogisticRegression(C=4.0, max_iter=2000, n_jobs=-1, solver="liblinear")
clf.fit(Xtr, y)
pred = clf.predict(Xte)
labels = le.inverse_transform(pred)

out = pd.DataFrame({"id": test["id"], "label": labels})
out.to_csv(os.path.join(BASE, "outputs", "submission.csv"), index=False)
print("done", out.shape, out["label"].value_counts().to_dict())
