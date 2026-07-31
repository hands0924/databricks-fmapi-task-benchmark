#!/usr/bin/env python3
"""Run a reproducible holdout comparison for the NSMC task."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
SEED = 2026


def fit_scores(vectorizer, train_text, valid_text, y_train, y_valid, name):
    started = time.time()
    x_train = vectorizer.fit_transform(train_text)
    x_valid = vectorizer.transform(valid_text)
    print(f"{name}: shape={x_train.shape}, vectorize={time.time() - started:.1f}s")

    results = {}
    for c in (0.25, 0.5, 1.0, 1.5):
        model = LinearSVC(C=c, class_weight=None, dual="auto", max_iter=3000)
        model.fit(x_train, y_train)
        score = model.decision_function(x_valid)
        results[c] = score
        print(f"{name} C={c}: accuracy={accuracy_score(y_valid, score >= 0):.6f}")
    return results


def main():
    data = pd.read_csv(ROOT / "train.csv")
    train_idx, valid_idx = train_test_split(
        np.arange(len(data)), test_size=0.1, random_state=SEED, stratify=data["label"]
    )
    train_text = data.loc[train_idx, "document"]
    valid_text = data.loc[valid_idx, "document"]
    y_train = data.loc[train_idx, "label"].to_numpy()
    y_valid = data.loc[valid_idx, "label"].to_numpy()

    char_scores = fit_scores(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 6),
            min_df=2,
            max_features=500_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        train_text,
        valid_text,
        y_train,
        y_valid,
        "char",
    )
    char_wb_scores = fit_scores(
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 6),
            min_df=2,
            max_features=500_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        train_text,
        valid_text,
        y_train,
        y_valid,
        "char_wb",
    )
    word_scores = fit_scores(
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=300_000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        ),
        train_text,
        valid_text,
        y_train,
        y_valid,
        "word",
    )

    best = (0.0, None)
    for char_c, char_score in char_scores.items():
        for word_c, word_score in word_scores.items():
            for char_weight in np.arange(0.5, 0.91, 0.05):
                score = char_weight * char_score + (1 - char_weight) * word_score
                accuracy = accuracy_score(y_valid, score >= 0)
                if accuracy > best[0]:
                    best = (accuracy, (char_c, word_c, round(float(char_weight), 2)))
    print(f"best ensemble: accuracy={best[0]:.6f}, (char_C, word_C, char_weight)={best[1]}")

    best_three = (0.0, None)
    for char_c, char_score in char_scores.items():
        for wb_c, wb_score in char_wb_scores.items():
            for word_c, word_score in word_scores.items():
                for wb_weight in np.arange(0.3, 0.76, 0.05):
                    for word_weight in np.arange(0.0, 0.21, 0.05):
                        char_weight = 1 - wb_weight - word_weight
                        if char_weight < 0:
                            continue
                        score = (
                            char_weight * char_score
                            + wb_weight * wb_score
                            + word_weight * word_score
                        )
                        accuracy = accuracy_score(y_valid, score >= 0)
                        if accuracy > best_three[0]:
                            best_three = (
                                accuracy,
                                (char_c, wb_c, word_c, char_weight, wb_weight, word_weight),
                            )
    print(
        "best three-model ensemble: "
        f"accuracy={best_three[0]:.6f}, "
        f"(C values, weights)={best_three[1]}"
    )

    # Exact repeated reviews can be labeled directly without using validation labels.
    char_c, word_c, char_weight = best[1]
    ensemble = char_weight * char_scores[char_c] + (1 - char_weight) * word_scores[word_c]
    predictions = (ensemble >= 0).astype(np.int8)
    lookup = data.loc[train_idx].groupby("document")["label"].mean()
    known = valid_text.map(lookup)
    mask = known.notna().to_numpy()
    predictions[mask] = (known[mask].to_numpy() >= 0.5).astype(np.int8)
    print(
        f"ensemble + exact lookup: accuracy={accuracy_score(y_valid, predictions):.6f}, "
        f"matched={mask.sum()}"
    )


if __name__ == "__main__":
    main()
