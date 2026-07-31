"""Train the final text ensemble and create outputs/submission.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["EAP", "HPL", "MWS"]


def fit_predict(
    train_text: pd.Series,
    train_y: pd.Series,
    test_text: pd.Series,
    vectorizer: TfidfVectorizer,
    c: float,
) -> np.ndarray:
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    model = LogisticRegression(
        C=c,
        solver="lbfgs",
        max_iter=500,
        random_state=2026,
    )
    model.fit(x_train, train_y)
    prediction = model.predict_proba(x_test)
    class_indices = [list(model.classes_).index(label) for label in CLASSES]
    return prediction[:, class_indices]


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")

    word_prediction = fit_predict(
        train["text"],
        train["author"],
        test["text"],
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.995,
            max_features=160_000,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        ),
        c=80.0,
    )
    char_prediction = fit_predict(
        train["text"],
        train["author"],
        test["text"],
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 6),
            min_df=2,
            max_features=220_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        c=50.0,
    )

    prediction = 0.4 * word_prediction + 0.6 * char_prediction
    prediction /= prediction.sum(axis=1, keepdims=True)
    predicted = pd.DataFrame(prediction, columns=CLASSES)
    predicted.insert(0, "id", test["id"].to_numpy())

    # Reorder against the template and fail early on any malformed IDs.
    if test["id"].duplicated().any() or set(test["id"]) != set(sample["id"]):
        raise ValueError("Test IDs must be unique and match sample_submission.csv")
    submission = sample[["id"]].merge(predicted, on="id", how="left", validate="one_to_one")
    if submission[CLASSES].isna().any().any():
        raise ValueError("Submission contains missing predictions")
    if not np.allclose(submission[CLASSES].sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("Submission probabilities do not sum to one")

    output_path = ROOT / "outputs" / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
