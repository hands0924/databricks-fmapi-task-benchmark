"""Final solution for BEEP! Korean hate-speech classification (t8_beep).

Approach: TF-IDF char n-grams (char_wb, 2-6) + LogisticRegression with
balanced class weights. A light text normalization collapses 3+ repeated
characters to 2 and lowercases (helps noisy web comments with ㅋㅋㅋ, etc.).

Dependencies: scikit-learn only (no internet / no pretrained weights).

Run:
    python solution/final.py
"""
import os
import re
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RANDOM_STATE = 42
N_SPLITS = 5
BEST_NGRAM = (2, 6)
BEST_C = 0.7


def normalize(text):
    """Collapse 3+ repeated chars to 2 and lowercase (handles ㅋㅋㅋㅋ, AAAA)."""
    t = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return t.lower()


def load_data():
    train = pd.read_csv(os.path.join(ROOT, "train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "test.csv"))
    sample = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    return train, test, sample


def build_pipeline(ngram=BEST_NGRAM, C=BEST_C):
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        lowercase=True,
        norm="l2",
    )
    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("tfidf", vec), ("clf", clf)])


def main():
    train, test, sample = load_data()
    X = np.array([normalize(t) for t in train["comment"].astype(str).values])
    le = LabelEncoder()
    y = le.fit_transform(train["label"].values)

    pipe = build_pipeline()

    # Cross-validation report
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[va])
        s = f1_score(y[va], pred, average="macro")
        scores.append(s)
        print(f"fold {fold}: macro_f1={s:.4f}")
    print(f"CV macro_f1 mean={np.mean(scores):.4f} std={np.std(scores):.4f}")

    # Fit on all training data, predict test
    pipe.fit(X, y)
    Xt = np.array([normalize(t) for t in test["comment"].astype(str).values])
    pred = pipe.predict(Xt)
    pred_labels = le.inverse_transform(pred)

    out = pd.DataFrame({"id": test["id"].values, "label": pred_labels})
    out_path = os.path.join(ROOT, "outputs", "submission.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")

    # Sanity checks against sample submission
    assert list(out.columns) == list(sample.columns), "columns mismatch"
    assert out["id"].tolist() == sample["id"].tolist(), "id mismatch"
    assert not out["id"].duplicated().any(), "duplicate ids"
    assert set(out["label"].unique()) <= set(["none", "offensive", "hate"]), "bad labels"
    print("Format checks passed.")


if __name__ == "__main__":
    main()
