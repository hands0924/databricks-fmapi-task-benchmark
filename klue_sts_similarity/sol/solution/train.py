#!/usr/bin/env python3
"""Train an offline KLUE-STS model and create outputs/submission.csv."""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


SEED = 20260731


def row_cosine(matrix, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[left].multiply(matrix[right]).sum(axis=1)).ravel()


def ngrams(text: str, n: int) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {compact[i : i + n] for i in range(max(0, len(compact) - n + 1))}


def overlap_features(a: str, b: str) -> list[float]:
    a = str(a).lower()
    b = str(b).lower()
    compact_a = re.sub(r"[\W_]", "", a)
    compact_b = re.sub(r"[\W_]", "", b)
    values: list[float] = []
    for n in (1, 2, 3, 4):
        sa, sb = ngrams(a, n), ngrams(b, n)
        inter = len(sa & sb)
        values.extend(
            [
                inter / max(1, len(sa | sb)),
                inter / max(1, min(len(sa), len(sb))),
            ]
        )
    ta, tb = set(a.split()), set(b.split())
    inter = len(ta & tb)
    values.extend([inter / max(1, len(ta | tb)), inter / max(1, min(len(ta), len(tb)))])
    la, lb = len(compact_a), len(compact_b)
    values.extend(
        [
            SequenceMatcher(None, compact_a, compact_b, autojunk=False).ratio(),
            min(la, lb) / max(1, max(la, lb)),
            abs(la - lb) / max(1, max(la, lb)),
            np.log1p(la),
            np.log1p(lb),
            float(compact_a == compact_b),
        ]
    )
    nums_a, nums_b = set(re.findall(r"\d+(?:\.\d+)?", a)), set(re.findall(r"\d+(?:\.\d+)?", b))
    values.extend(
        [
            float(bool(nums_a or nums_b)),
            len(nums_a & nums_b) / max(1, len(nums_a | nums_b)),
            float(bool(nums_a) != bool(nums_b)),
        ]
    )
    negations = ("안", "않", "못", "없", "아니", "말고", "금지")
    neg_a = sum(token in a for token in negations)
    neg_b = sum(token in b for token in negations)
    values.extend([float(neg_a), float(neg_b), float(bool(neg_a) != bool(neg_b))])
    return values


def make_features(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    pairs = pd.concat([train[["sentence1", "sentence2"]], test[["sentence1", "sentence2"]]], ignore_index=True)
    corpus = pd.concat([pairs.sentence1, pairs.sentence2], ignore_index=True).astype(str)
    n_pairs = len(pairs)
    left = np.arange(n_pairs)
    right = left + n_pairs
    columns: list[np.ndarray] = []
    specs = [
        ("char", (1, 2), 1, 80000),
        ("char", (2, 4), 1, 120000),
        ("char_wb", (2, 5), 1, 120000),
        ("word", (1, 2), 1, 80000),
    ]
    for analyzer, ngram_range, min_df, max_features in specs:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b" if analyzer == "word" else None,
        )
        matrix = vectorizer.fit_transform(corpus)
        columns.append(row_cosine(matrix, left, right))

    surface = np.asarray(
        [overlap_features(a, b) for a, b in zip(pairs.sentence1, pairs.sentence2)], dtype=np.float32
    )
    return np.column_stack(columns + [surface]).astype(np.float32)


def pair_ridge_matrix(train: pd.DataFrame, test: pd.DataFrame):
    pairs = pd.concat([train[["sentence1", "sentence2"]], test[["sentence1", "sentence2"]]], ignore_index=True)
    # Separate views let Ridge learn lexical/domain priors while swap averaging preserves symmetry.
    corpus = pd.concat([pairs.sentence1, pairs.sentence2], ignore_index=True).astype(str)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=140000, sublinear_tf=True)
    word_matrix = word.fit_transform(corpus)
    char_matrix = char.fit_transform(corpus)
    n = len(pairs)
    left, right = np.arange(n), np.arange(n) + n
    direct = hstack([word_matrix[left], word_matrix[right], char_matrix[left], char_matrix[right]], format="csr")
    swapped = hstack([word_matrix[right], word_matrix[left], char_matrix[right], char_matrix[left]], format="csr")
    return direct, swapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation reporting")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    y = train.score.to_numpy(dtype=np.float64)
    n_train = len(train)
    features = make_features(train, test)
    ridge_direct, ridge_swapped = pair_ridge_matrix(train, test)
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_tree = np.zeros(n_train)
    oof_hist = np.zeros(n_train)
    oof_ridge = np.zeros(n_train)
    if not args.no_cv:
        for fold, (fit_idx, val_idx) in enumerate(folds.split(train), 1):
            tree = ExtraTreesRegressor(
                n_estimators=400,
                min_samples_leaf=3,
                max_features=0.9,
                n_jobs=-1,
                random_state=SEED + fold,
            )
            hist = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.045, max_leaf_nodes=20, l2_regularization=2.0, random_state=SEED + fold
            )
            ridge = Ridge(alpha=18.0)
            tree.fit(features[fit_idx], y[fit_idx])
            hist.fit(features[fit_idx], y[fit_idx])
            ridge.fit(ridge_direct[fit_idx], y[fit_idx])
            oof_tree[val_idx] = tree.predict(features[val_idx])
            oof_hist[val_idx] = hist.predict(features[val_idx])
            oof_ridge[val_idx] = (ridge.predict(ridge_direct[val_idx]) + ridge.predict(ridge_swapped[val_idx])) / 2
        for name, pred in (
            ("extra", oof_tree),
            ("hist", oof_hist),
            ("ridge", oof_ridge),
        ):
            print(f"CV {name:>5}: {pearsonr(y, pred).statistic:.6f}")
        blend = 0.35 * oof_tree + 0.50 * oof_hist + 0.15 * oof_ridge
        print(f"CV blend: {pearsonr(y, blend).statistic:.6f}")

    final_tree = ExtraTreesRegressor(
        n_estimators=700, min_samples_leaf=3, max_features=0.9, n_jobs=-1, random_state=SEED
    )
    final_hist = HistGradientBoostingRegressor(
        max_iter=350, learning_rate=0.045, max_leaf_nodes=20, l2_regularization=2.0, random_state=SEED
    )
    final_ridge = Ridge(alpha=18.0)
    final_tree.fit(features[:n_train], y)
    final_hist.fit(features[:n_train], y)
    final_ridge.fit(ridge_direct[:n_train], y)
    test_ridge = (
        final_ridge.predict(ridge_direct[n_train:]) + final_ridge.predict(ridge_swapped[n_train:])
    ) / 2
    prediction = (
        0.35 * final_tree.predict(features[n_train:])
        + 0.50 * final_hist.predict(features[n_train:])
        + 0.15 * test_ridge
    )
    prediction = np.clip(prediction, 0.0, 5.0)

    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test.id, "score": prediction})
    submission.to_csv(output_dir / "submission.csv", index=False)
    print(f"Wrote {len(submission)} predictions to {output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
