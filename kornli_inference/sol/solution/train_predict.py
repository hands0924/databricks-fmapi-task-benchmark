#!/usr/bin/env python3
"""Train the offline KorNLI model and create outputs/submission.csv."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
NEGATIONS = ("않", "없", "못", "아니", "결코", "전혀")
BASE_BINS = [20] * 7 + [4, 4, 2, 2]
INTERACTION_SIZES = [400, 400, 400, 320, 320, 400]


def pair_features(frame: pd.DataFrame) -> csr_matrix:
    """Create binned pairwise and interaction features."""
    raw = []
    for premise, hypothesis in zip(frame["sentence1"], frame["sentence2"]):
        premise_words = set(TOKEN_RE.findall(premise.lower()))
        hypothesis_words = set(TOKEN_RE.findall(hypothesis.lower()))
        premise_bigrams = {
            premise[i : i + 2] for i in range(max(0, len(premise) - 1))
        }
        hypothesis_bigrams = {
            hypothesis[i : i + 2] for i in range(max(0, len(hypothesis) - 1))
        }
        raw.append(
            [
                len(premise_words & hypothesis_words) / max(len(hypothesis_words), 1),
                len(premise_words & hypothesis_words)
                / max(len(premise_words | hypothesis_words), 1),
                len(set(premise) & set(hypothesis)) / max(len(set(hypothesis)), 1),
                len(premise_bigrams & hypothesis_bigrams)
                / max(len(hypothesis_bigrams), 1),
                len(hypothesis) / max(len(premise), 1),
                min(len(premise), 200) / 200,
                min(len(hypothesis), 150) / 150,
                min(sum(token in premise for token in NEGATIONS), 3),
                min(sum(token in hypothesis for token in NEGATIONS), 3),
                int(premise == hypothesis),
                int(hypothesis in premise),
            ]
        )

    raw_array = np.asarray(raw)
    values = []
    for column, bin_count in enumerate(BASE_BINS):
        if column < 7:
            value = (raw_array[:, column] * bin_count).astype(int)
        else:
            value = raw_array[:, column].astype(int)
        values.append(np.minimum(bin_count - 1, value))

    negation_state = values[7] * 4 + values[8]
    values.extend(
        [
            values[0] * 20 + values[4],
            values[1] * 20 + values[4],
            values[3] * 20 + values[4],
            values[0] * 16 + negation_state,
            values[3] * 16 + negation_state,
            values[0] * 20 + values[3],
        ]
    )

    sizes = BASE_BINS + INTERACTION_SIZES
    offsets = np.cumsum([0] + sizes)
    rows = np.repeat(np.arange(len(frame)), len(values))
    columns = np.column_stack(
        [offsets[index] + value for index, value in enumerate(values)]
    ).ravel()
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, columns)), shape=(len(frame), offsets[-1]))


def apply_premise_constraint(
    scores: np.ndarray,
    classes: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    """Penalize labels already used by training examples of the same premise."""
    counts = (
        train.groupby(["sentence1", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=classes, fill_value=0)
    )
    test_counts = np.zeros_like(scores)
    repeated = test["sentence1"].isin(counts.index).to_numpy()
    if repeated.any():
        test_counts[repeated] = counts.loc[test.loc[repeated, "sentence1"]].to_numpy()
    return classes[(scores - 10.0 * test_counts).argmax(axis=1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/submission.csv")
    )
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv").fillna("")
    test = pd.read_csv(args.data_dir / "test.csv").fillna("")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 6),
        min_df=2,
        max_features=400_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_text = vectorizer.fit_transform(train["sentence2"])
    test_text = vectorizer.transform(test["sentence2"])
    train_pair = pair_features(train) * 0.25
    test_pair = pair_features(test) * 0.25
    train_matrix = hstack([train_text, train_pair], format="csr")
    test_matrix = hstack([test_text, test_pair], format="csr")

    model = LinearSVC(C=0.14, dual=True, random_state=2026)
    model.fit(train_matrix, train["label"])
    scores = model.decision_function(test_matrix)
    predictions = apply_premise_constraint(
        scores, model.classes_, train, test
    )

    submission = pd.DataFrame({"id": test["id"], "label": predictions})
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if submission["id"].duplicated().any() or set(submission["id"]) != set(test["id"]):
        raise ValueError("Submission IDs are not a one-to-one match with test.csv")
    if not set(submission["label"]).issubset(set(train["label"])):
        raise ValueError("Submission contains an unknown label")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
