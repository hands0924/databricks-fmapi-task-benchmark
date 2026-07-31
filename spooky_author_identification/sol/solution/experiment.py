"""Local holdout experiments for model selection."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
CLASSES = np.array(["EAP", "HPL", "MWS"])


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    train_text, valid_text, train_y, valid_y = train_test_split(
        train["text"],
        train["author"],
        test_size=0.25,
        random_state=1337,
        stratify=train["author"],
    )

    configs = {
        "word": TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.995,
            max_features=160_000,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        ),
        "char": TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 6),
            min_df=2,
            max_features=220_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
    }

    predictions: dict[tuple[str, float], np.ndarray] = {}
    for name, vectorizer in configs.items():
        x_train = vectorizer.fit_transform(train_text)
        x_valid = vectorizer.transform(valid_text)
        print(f"{name}: {x_train.shape[1]:,} features")
        for c in (30.0, 50.0, 80.0, 120.0):
            model = LogisticRegression(
                C=c,
                solver="lbfgs",
                max_iter=500,
                random_state=2026,
            )
            model.fit(x_train, train_y)
            pred = model.predict_proba(x_valid)
            pred = pred[:, [list(model.classes_).index(label) for label in CLASSES]]
            predictions[(name, c)] = pred
            print(f"  C={c:<3}: {log_loss(valid_y, pred, labels=CLASSES):.6f}")

    print("blends (word weight):")
    for word_c in (30.0, 50.0, 80.0):
        for char_c in (30.0, 50.0, 80.0):
            scores = []
            for weight in np.arange(0.2, 0.81, 0.1):
                pred = weight * predictions[("word", word_c)] + (
                    1.0 - weight
                ) * predictions[("char", char_c)]
                scores.append((log_loss(valid_y, pred, labels=CLASSES), weight))
            score, weight = min(scores)
            print(
                f"  word C={word_c}, char C={char_c}: "
                f"{score:.6f} at word_weight={weight:.1f}"
            )


if __name__ == "__main__":
    main()
