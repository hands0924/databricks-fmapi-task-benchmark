#!/usr/bin/env python3
"""Train the selected KorSTS ensemble and create outputs/submission.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict

from experiment import make_dense_features, normalize


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260731


def make_sparse_features(train, test):
    train_a = train.sentence1.map(normalize).tolist()
    train_b = train.sentence2.map(normalize).tolist()
    test_a = test.sentence1.map(normalize).tolist()
    test_b = test.sentence2.map(normalize).tolist()
    fit_docs = train_a + train_b
    train_parts = []
    test_parts = []

    for analyzer, gram_range, max_features in [
        ("word", (1, 2), 50000),
        ("char", (2, 5), 100000),
    ]:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=gram_range,
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
        )
        vectorizer.fit(fit_docs)
        tr1, tr2 = vectorizer.transform(train_a), vectorizer.transform(train_b)
        te1, te2 = vectorizer.transform(test_a), vectorizer.transform(test_b)
        train_parts.extend([abs(tr1 - tr2), tr1.multiply(tr2)])
        test_parts.extend([abs(te1 - te2), te1.multiply(te2)])

    return hstack(train_parts, format="csr"), hstack(test_parts, format="csr")


def dense_models():
    return [
        HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.045,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            random_state=SEED,
        ),
        ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=4,
            max_features=0.9,
            n_jobs=-1,
            random_state=SEED,
        ),
    ]


def pair_key(sentence1, sentence2):
    return tuple(sorted((normalize(sentence1), normalize(sentence2))))


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    y = train.score.to_numpy(dtype=float)

    # Fit all text transforms on train text only. Repeating the deterministic
    # transform keeps train and test in exactly the same feature space.
    train_dense = make_dense_features(train, fit_frame=train)
    test_dense = make_dense_features(test, fit_frame=train)
    train_sparse, test_sparse = make_sparse_features(train, test)

    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_columns = []
    fitted_dense = []
    for model in dense_models():
        oof_columns.append(cross_val_predict(model, train_dense, y, cv=folds, n_jobs=1))
        model.fit(train_dense, y)
        fitted_dense.append(model)

    ridge = Ridge(alpha=10.0, solver="lsqr")
    oof_columns.append(cross_val_predict(ridge, train_sparse, y, cv=folds, n_jobs=1))
    ridge.fit(train_sparse, y)

    oof = np.column_stack(oof_columns)
    blender = LinearRegression(positive=True).fit(oof, y)
    cv_score = pearsonr(y, blender.predict(oof)).statistic
    print(f"OOF Pearson: {cv_score:.6f}; blend weights: {blender.coef_}")

    test_base = np.column_stack(
        [model.predict(test_dense) for model in fitted_dense] + [ridge.predict(test_sparse)]
    )
    predictions = blender.predict(test_base)

    # Repeated pairs are annotation duplicates; their observed mean is much
    # more accurate than a lexical model (fold-safe CV Pearson about 0.97).
    exact_scores = {}
    for row in train.itertuples(index=False):
        exact_scores.setdefault(pair_key(row.sentence1, row.sentence2), []).append(row.score)
    replaced = 0
    for i, row in enumerate(test.itertuples(index=False)):
        scores = exact_scores.get(pair_key(row.sentence1, row.sentence2))
        if scores:
            predictions[i] = np.mean(scores)
            replaced += 1

    prediction_by_id = pd.Series(np.clip(predictions, 0.0, 5.0), index=test.id)
    submission = sample[["id"]].copy()
    submission["score"] = submission.id.map(prediction_by_id)
    if submission.score.isna().any() or not submission.id.is_unique:
        raise ValueError("Submission IDs do not map one-to-one to test IDs")
    if set(submission.id) != set(test.id) or len(submission) != len(test):
        raise ValueError("Submission IDs differ from test IDs")

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    print(f"Wrote {len(submission)} predictions; exact-pair replacements: {replaced}")


if __name__ == "__main__":
    main()
