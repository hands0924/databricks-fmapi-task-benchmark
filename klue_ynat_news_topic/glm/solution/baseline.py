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
sub = pd.read_csv("sample_submission.csv")

print("train:", train.shape, "test:", test.shape)

# char n-gram TF-IDF works well for Korean short text
vec = TfidfVectorizer(
    sublinear_tf=True,
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=2,
    max_df=0.95,
)
Xtr = vec.fit_transform(train["title"].astype(str))
Xte = vec.transform(test["title"].astype(str))
y = train["label"].values
print("Xtr:", Xtr.shape, "Xte:", Xte.shape)

# Quick CV check
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(train), dtype=object)
for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
    clf = LogisticRegression(
        max_iter=2000, C=3.0, solver="liblinear", random_state=SEED
    )
    clf.fit(Xtr[tr_idx], y[tr_idx])
    oof[va_idx] = clf.predict(Xtr[va_idx])

print("CV Macro F1:", f1_score(y, oof, average="macro"))

# Final model
clf = LogisticRegression(max_iter=2000, C=3.0, solver="liblinear", random_state=SEED)
clf.fit(Xtr, y)
pred = clf.predict(Xte)

out = pd.DataFrame({"id": test["id"].values, "label": pred})
out.to_csv("outputs/submission.csv", index=False)
print("Saved outputs/submission.csv:", out.shape)
print(out["label"].value_counts())
