#!/usr/bin/env python3
"""Train the final classifier and create outputs/submission.csv."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
VALID_LABELS = {"none", "offensive", "hate"}


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    template = pd.read_csv(ROOT / "sample_submission.csv")

    if list(train.columns) != ["id", "comment", "label"]:
        raise ValueError("Unexpected train.csv columns")
    if list(test.columns) != ["id", "comment"]:
        raise ValueError("Unexpected test.csv columns")
    if list(template.columns) != ["id", "label"]:
        raise ValueError("Unexpected sample_submission.csv columns")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Missing values are not supported")
    if not set(train["label"]).issubset(VALID_LABELS):
        raise ValueError("Unknown training label")
    if test["id"].duplicated().any() or template["id"].duplicated().any():
        raise ValueError("Duplicate test or template IDs")
    if set(test["id"]) != set(template["id"]):
        raise ValueError("Test and template IDs differ")

    # Character-boundary features are robust to Korean slang and inconsistent spacing.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 5),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=300_000,
    )
    x_train = vectorizer.fit_transform(train["comment"].astype(str))
    x_test = vectorizer.transform(test["comment"].astype(str))

    counts = train["label"].value_counts()
    n_rows = len(train)
    # Start from balanced weights and apply small CV-selected adjustments.
    class_weight = {
        "hate": n_rows / (3 * counts["hate"]) * 1.2,
        "none": n_rows / (3 * counts["none"]),
        "offensive": n_rows / (3 * counts["offensive"]) * 1.1,
    }
    model = LinearSVC(C=0.3, class_weight=class_weight, dual=True, random_state=42)
    model.fit(x_train, train["label"])
    predictions = model.predict(x_test)

    prediction_by_id = dict(zip(test["id"], predictions, strict=True))
    submission = template[["id"]].copy()
    submission["label"] = submission["id"].map(prediction_by_id)

    if len(submission) != len(test) or submission["id"].nunique() != len(test):
        raise ValueError("Submission does not contain every test ID exactly once")
    if submission["label"].isna().any() or not set(submission["label"]).issubset(VALID_LABELS):
        raise ValueError("Submission contains an invalid label")

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path}")
    print(submission["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
