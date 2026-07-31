#!/usr/bin/env python3
"""Train the final YNAT classifier and create outputs/submission.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


CLASSES = {"IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"}


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=workspace / "train.csv")
    parser.add_argument("--test", type=Path, default=workspace / "test.csv")
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=workspace / "sample_submission.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=workspace / "outputs" / "submission.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample_submission)

    if list(train.columns) != ["id", "title", "label"]:
        raise ValueError("train.csv must have columns: id,title,label")
    if list(test.columns) != ["id", "title"]:
        raise ValueError("test.csv must have columns: id,title")
    if list(sample.columns) != ["id", "label"]:
        raise ValueError("sample_submission.csv must have columns: id,label")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Input data contains missing values")
    if set(train["label"].unique()) != CLASSES:
        raise ValueError("Unexpected training labels")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 5),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=600_000,
        dtype=np.float32,
    )
    train_features = vectorizer.fit_transform(train["title"])
    test_features = vectorizer.transform(test["title"])

    # Partial inverse-frequency weighting targets macro F1 without overcorrecting.
    counts = train["label"].value_counts()
    class_weights = {
        label: (len(train) / (len(CLASSES) * counts[label])) ** 0.65
        for label in CLASSES
    }
    model = LinearSVC(C=0.35, class_weight=class_weights, dual=True)
    model.fit(train_features, train["label"])
    predictions = model.predict(test_features)

    submission = pd.DataFrame({"id": test["id"], "label": predictions})
    if len(submission) != len(test) or not submission["id"].is_unique:
        raise RuntimeError("Submission IDs are not one-to-one with test rows")
    if set(submission["id"]) != set(test["id"]):
        raise RuntimeError("Submission IDs do not match test.csv")
    if not set(submission["label"]).issubset(CLASSES):
        raise RuntimeError("Submission contains an invalid label")
    if submission.columns.tolist() != sample.columns.tolist():
        raise RuntimeError("Submission columns do not match the sample")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
