#!/usr/bin/env python3
"""Train the final ensemble and create outputs/submission.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
TARGET = "rented_bike_count"


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(data["date"], dayfirst=True)
    features = data.drop(columns=["id", "date", TARGET], errors="ignore").copy()

    features["month"] = dates.dt.month
    features["dayofyear"] = dates.dt.dayofyear
    features["dayofweek"] = dates.dt.dayofweek
    features["day"] = dates.dt.day
    features["weekofyear"] = dates.dt.isocalendar().week.astype(int)
    features["weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    features["days_since_start"] = (dates - pd.Timestamp("2017-12-01")).dt.days

    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["dow_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
    features["doy_sin"] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)

    return pd.get_dummies(
        features,
        columns=["seasons", "holiday", "functioning_day"],
        dtype=float,
    )


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")

    train_features = make_features(train)
    test_features = make_features(test).reindex(
        columns=train_features.columns, fill_value=0.0
    )
    target = train[TARGET].to_numpy()

    # Non-functioning hours are deterministically zero, so excluding them keeps
    # the regressors focused on demand while the rule is applied after fitting.
    functioning_train = train["functioning_day"].eq("Yes").to_numpy()
    model_specs = [
        (0.50, "squared_error", 9, 3.0),
        (0.25, "squared_error", 15, 30.0),
        (0.25, "poisson", 15, 0.0),
    ]

    prediction = np.zeros(len(test), dtype=float)
    for weight, loss, max_leaf_nodes, l2_regularization in model_specs:
        model = HistGradientBoostingRegressor(
            loss=loss,
            max_iter=600,
            learning_rate=0.05,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=20,
            l2_regularization=l2_regularization,
            early_stopping=False,
            random_state=42,
        )
        model.fit(
            train_features.loc[functioning_train], target[functioning_train]
        )
        prediction += weight * model.predict(test_features)

    prediction = np.maximum(prediction, 0.0)
    prediction[test["functioning_day"].eq("No").to_numpy()] = 0.0

    if list(sample.columns) != ["id", TARGET]:
        raise ValueError("Unexpected sample_submission.csv columns")
    if test["id"].duplicated().any() or sample["id"].duplicated().any():
        raise ValueError("IDs must be unique")
    if set(test["id"]) != set(sample["id"]):
        raise ValueError("Test and sample submission IDs do not match")

    prediction_by_id = pd.Series(prediction, index=test["id"])
    submission = sample[["id"]].copy()
    submission[TARGET] = submission["id"].map(prediction_by_id)
    if submission[TARGET].isna().any() or not np.isfinite(submission[TARGET]).all():
        raise ValueError("Predictions contain missing or non-finite values")

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    print(f"Wrote {len(submission)} predictions to {output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
