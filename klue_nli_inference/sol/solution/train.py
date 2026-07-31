#!/usr/bin/env python3
"""Train a local KLUE-NLI model and create outputs/submission.csv."""

from __future__ import annotations

import argparse
import difflib
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


LABELS = np.array(["contradiction", "entailment", "neutral"])


def text_features(fit_df: pd.DataFrame, frames: list[pd.DataFrame]):
    corpus = pd.concat([fit_df["premise"], fit_df["hypothesis"]], ignore_index=True)
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=220_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=120_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char.fit(corpus)
    word.fit(corpus)

    output = []
    for df in frames:
        blocks = []
        for vectorizer in (char, word):
            p = vectorizer.transform(df["premise"])
            h = vectorizer.transform(df["hypothesis"])
            blocks.extend([p, h, abs(p - h), p.multiply(h)])
        blocks.append(pair_features(df))
        output.append(sparse.hstack(blocks, format="csr", dtype=np.float32))
    return output


def pair_features(df: pd.DataFrame):
    negative = ("안", "않", "없", "못", "아니", "반대", "실패", "금지")
    rows = []
    for premise, hypothesis in zip(df["premise"], df["hypothesis"]):
        p_chars, h_chars = set(premise.replace(" ", "")), set(hypothesis.replace(" ", ""))
        p_words, h_words = set(premise.split()), set(hypothesis.split())
        p_bigrams = set(zip(premise, premise[1:]))
        h_bigrams = set(zip(hypothesis, hypothesis[1:]))
        p_numbers, h_numbers = set(re.findall(r"\d+", premise)), set(re.findall(r"\d+", hypothesis))
        char_inter = len(p_chars & h_chars)
        word_inter = len(p_words & h_words)
        bigram_inter = len(p_bigrams & h_bigrams)
        p_neg = sum(token in premise for token in negative)
        h_neg = sum(token in hypothesis for token in negative)
        position = int(df.loc[df.index[len(rows)], "position"])
        rows.append(
            [
                len(premise) / 100,
                len(hypothesis) / 100,
                len(hypothesis) / max(len(premise), 1),
                char_inter / max(len(h_chars), 1),
                char_inter / max(len(p_chars | h_chars), 1),
                word_inter / max(len(h_words), 1),
                word_inter / max(len(p_words | h_words), 1),
                bigram_inter / max(len(h_bigrams), 1),
                bigram_inter / max(len(p_bigrams | h_bigrams), 1),
                difflib.SequenceMatcher(None, premise, hypothesis, autojunk=False).ratio(),
                p_neg,
                h_neg,
                abs(p_neg - h_neg),
                premise == hypothesis,
                hypothesis.rstrip(".") in premise,
                bool(h_numbers - p_numbers),
                p_numbers == h_numbers and bool(h_numbers),
                position == 0,
                position == 1,
                position >= 2,
            ]
        )
    return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))


def grouped_predictions(
    known: pd.DataFrame,
    target: pd.DataFrame,
    scores: np.ndarray,
    uniqueness_bonus: float,
) -> np.ndarray:
    """Jointly decode examples sharing a premise.

    KLUE-NLI usually contains one example of each class for a premise. The
    finite bonus captures that strong prior while allowing textual evidence
    to override annotation exceptions.
    """
    label_index = {label: i for i, label in enumerate(LABELS)}
    known_by_premise = known.groupby("premise")["label"].apply(list).to_dict()
    predictions = np.empty(len(target), dtype=object)

    positions_by_premise: dict[str, list[int]] = {}
    for position, premise in enumerate(target["premise"]):
        positions_by_premise.setdefault(premise, []).append(position)

    for premise, positions in positions_by_premise.items():
        known_indices = [label_index[x] for x in known_by_premise.get(premise, [])]
        best_value = -np.inf
        best_assignment = None
        for assignment in itertools.product(range(3), repeat=len(positions)):
            value = sum(scores[pos, label] for pos, label in zip(positions, assignment))
            value += uniqueness_bonus * len(set(known_indices + list(assignment)))
            if value > best_value:
                best_value = value
                best_assignment = assignment
        predictions[positions] = LABELS[list(best_assignment)]
    return predictions


def fit_model(train: pd.DataFrame, other: pd.DataFrame):
    # Fitting vocabulary on unlabeled target text is an allowed transductive step.
    fit_df = pd.concat([train.drop(columns="label"), other], ignore_index=True)
    x_train, x_other = text_features(fit_df, [train, other])
    model = LinearSVC(C=0.6, class_weight="balanced", dual=True, max_iter=10_000, tol=1e-3)
    model.fit(x_train, train["label"])
    raw_scores = model.decision_function(x_other)
    order = [list(model.classes_).index(label) for label in LABELS]
    return raw_scores[:, order]


def add_group_position(train: pd.DataFrame, test: pd.DataFrame):
    universe = pd.concat([train[["id", "premise"]], test[["id", "premise"]]], ignore_index=True)
    universe["number"] = universe["id"].str.extract(r"(\d+)$").astype(int)
    universe["position"] = universe.groupby("premise")["number"].rank(method="dense").astype(int) - 1
    positions = universe.set_index("id")["position"]
    train = train.copy()
    test = test.copy()
    train["position"] = train["id"].map(positions)
    test["position"] = test["id"].map(positions)
    return train, test


def validate(train: pd.DataFrame, test: pd.DataFrame) -> None:
    # Premises absent from the real test form complete groups, allowing a
    # holdout that faithfully reproduces final joint decoding.
    train, _ = add_group_position(train, test)
    complete = train[~train["premise"].isin(set(test["premise"]))].reset_index(drop=True)
    fit_idx, valid_idx = train_test_split(
        np.arange(len(complete)),
        test_size=0.2,
        random_state=2026,
        stratify=complete["label"],
    )
    fit = complete.iloc[fit_idx].reset_index(drop=True)
    valid = complete.iloc[valid_idx].reset_index(drop=True)
    scores = fit_model(fit, valid.drop(columns="label"))
    print(f"text-only accuracy: {accuracy_score(valid['label'], LABELS[scores.argmax(1)]):.6f}")
    for bonus in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0):
        pred = grouped_predictions(fit, valid, scores, bonus)
        print(f"group bonus {bonus:>4}: {accuracy_score(valid['label'], pred):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--bonus", type=float, default=10.0)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    train = pd.read_csv(args.root / "train.csv")
    test = pd.read_csv(args.root / "test.csv")
    if args.validate:
        validate(train, test)
        return

    train, test = add_group_position(train, test)
    scores = fit_model(train, test)
    predictions = grouped_predictions(train, test, scores, args.bonus)
    submission = pd.DataFrame({"id": test["id"], "label": predictions})
    output_dir = args.root / "outputs"
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    print(submission["label"].value_counts().to_string())
    print(f"wrote {output_dir / 'submission.csv'} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
