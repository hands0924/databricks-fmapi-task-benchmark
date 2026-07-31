#!/usr/bin/env python3
"""Train the final NSMC ensemble and create outputs/submission.csv."""

from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
SAMPLE_PATH = ROOT / "sample_submission.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission.csv"

MODEL_CONFIGS = (
    ({"analyzer": "char", "ngram_range": (2, 6)}, 0.25, 0.60, 500_000),
    ({"analyzer": "char_wb", "ngram_range": (2, 6)}, 0.50, 0.35, 500_000),
    ({"analyzer": "word", "ngram_range": (1, 2)}, 0.25, 0.05, 300_000),
)


def fit_component(train_text, labels, test_text, vectorizer_args, c, max_features):
    options = dict(vectorizer_args)
    if options["analyzer"] == "word":
        options["token_pattern"] = r"(?u)\b\w+\b"
    vectorizer = TfidfVectorizer(
        **options,
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    model = LinearSVC(C=c, dual="auto", max_iter=3000)
    model.fit(x_train, labels)
    return model.decision_function(x_test)


def validate_inputs(train, test, sample):
    if list(train.columns) != ["id", "document", "label"]:
        raise ValueError(f"Unexpected train columns: {list(train.columns)}")
    if list(test.columns) != ["id", "document"]:
        raise ValueError(f"Unexpected test columns: {list(test.columns)}")
    if list(sample.columns) != ["id", "label"]:
        raise ValueError(f"Unexpected sample submission columns: {list(sample.columns)}")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Training and test data must not contain missing values")
    if not train["label"].isin([0, 1]).all():
        raise ValueError("Training labels must be binary")
    if not test["id"].is_unique:
        raise ValueError("Test IDs must be unique")
    if len(sample) != len(test) or set(sample["id"]) != set(test["id"]):
        raise ValueError("Sample submission IDs do not match test IDs")


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    validate_inputs(train, test, sample)

    scores = np.zeros(len(test), dtype=np.float64)
    for args, c, weight, max_features in MODEL_CONFIGS:
        started = time.time()
        component = fit_component(
            train["document"],
            train["label"].to_numpy(),
            test["document"],
            args,
            c,
            max_features,
        )
        scores += weight * component
        print(
            f"Fitted {args['analyzer']} model in {time.time() - started:.1f}s "
            f"(C={c}, weight={weight})"
        )
        del component
        gc.collect()

    prediction_by_id = pd.Series((scores >= 0).astype(np.int8), index=test["id"])
    submission = sample[["id"]].copy()
    submission["label"] = submission["id"].map(prediction_by_id)
    if submission["label"].isna().any():
        raise ValueError("At least one submission ID has no prediction")
    submission["label"] = submission["label"].astype(np.int8)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(submission)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
