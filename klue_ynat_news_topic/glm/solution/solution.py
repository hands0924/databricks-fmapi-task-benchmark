"""
뉴스 토픽 분류 (KLUE-YNAT) — 최종 솔루션
========================================
특징: char_wb n-gram TF-IDF (비 sublinear) + Logistic Regression (liblinear)

CV(Macro F1) ~= 0.842 (5-fold StratifiedKFold, seed=42)

재현 방법:
    python solution/solution.py
-> ../outputs/submission.csv 생성 (이 스크립트는 프로젝트 루트에서 실행)
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(ROOT, "train.csv")
TEST_CSV = os.path.join(ROOT, "test.csv")
OUT_CSV = os.path.join(ROOT, "outputs", "submission.csv")


def main():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    y = train["label"].values
    titles_tr = train["title"].astype(str).values
    titles_te = test["title"].astype(str).values
    print("train:", train.shape, "test:", test.shape)

    # char_wb n-gram TF-IDF — 한국어 짧은 제목에 강력
    # sublinear_tf=False 가 본 데이터에서 미세하게 우수 (CV 0.8424 vs 0.8420)
    vec = TfidfVectorizer(
        sublinear_tf=False,
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.95,
    )
    Xtr = vec.fit_transform(titles_tr)
    Xte = vec.transform(titles_te)
    print("Xtr:", Xtr.shape, "Xte:", Xte.shape)

    # (선택) 교차검증 리포트 — 최종 제출에는 영향 없음
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=object)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
        clf = LogisticRegression(
            max_iter=2000, C=3.0, solver="liblinear", random_state=SEED
        )
        clf.fit(Xtr[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict(Xtr[va_idx])
    print("CV Macro F1:", round(f1_score(y, oof, average="macro"), 4))

    # 최종 모델: 전체 학습 데이터로 학습
    clf = LogisticRegression(
        max_iter=2000, C=3.0, solver="liblinear", random_state=SEED
    )
    clf.fit(Xtr, y)
    pred = clf.predict(Xte)

    out = pd.DataFrame({"id": test["id"].values, "label": pred})
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV, out.shape)
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
