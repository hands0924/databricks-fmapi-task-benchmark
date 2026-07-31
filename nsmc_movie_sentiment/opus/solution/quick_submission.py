"""Fast safe baseline: char_wb TF-IDF + LogisticRegression on full train."""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA = "/tmp/kmle/M1_t4_nsmc_full_20260730_143411/task"
tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")
tr["document"] = tr["document"].astype(str)
te["document"] = te["document"].astype(str)

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
A = vec.fit_transform(tr["document"].values)
B = vec.transform(te["document"].values)

clf = LogisticRegression(C=4, max_iter=2000, solver="liblinear")
clf.fit(A, tr["label"].values)
pred = clf.predict(B)

os.makedirs(f"{DATA}/outputs", exist_ok=True)
pd.DataFrame({"id": te["id"].values, "label": pred.astype(int)}).to_csv(
    f"{DATA}/outputs/submission.csv", index=False
)
print("saved", pred.mean())
