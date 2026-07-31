#!/usr/bin/env python3
"""Train a match-aware PUBG placement model and create the submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor


SEED = 20260730
ID_COLS = ["Id", "groupId", "matchId"]
TARGET = "winPlacePerc"


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["totalDistance"] = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    df["healsAndBoosts"] = df["heals"] + df["boosts"]
    df["killsAndAssists"] = df["kills"] + df["assists"]
    df["headshotRate"] = df["headshotKills"] / df["kills"].clip(lower=1)
    df["killStreakRate"] = df["killStreaks"] / df["kills"].clip(lower=1)
    df["damagePerKill"] = df["damageDealt"] / df["kills"].clip(lower=1)
    df["distancePerSecond"] = df["totalDistance"] / df["matchDuration"].clip(lower=1)
    df["walkPerSecond"] = df["walkDistance"] / df["matchDuration"].clip(lower=1)
    df["items"] = df["weaponsAcquired"] + df["heals"] + df["boosts"]
    df["fpp"] = df["matchType"].str.contains("fpp").astype("int8")
    df["modePlayers"] = np.select(
        [df["matchType"].str.contains("solo"), df["matchType"].str.contains("duo")],
        [1, 2],
        default=4,
    ).astype("int8")
    df["specialMode"] = (~df["matchType"].str.match(r"^(solo|duo|squad)(-fpp)?$")) .astype("int8")
    return df.drop(columns="matchType")


def make_group_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one feature row and metadata row per (match, group)."""
    df = add_player_features(df)
    keys = ["matchId", "groupId"]
    excluded = set(ID_COLS + [TARGET])
    numeric = [c for c in df.columns if c not in excluded]

    grouped = df.groupby(keys, sort=False)[numeric].agg(["mean", "max", "min", "sum"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    grouped = grouped.reset_index()

    aggregate_cols = [c for c in grouped.columns if c not in keys]
    ranks = grouped.groupby("matchId", sort=False)[aggregate_cols].rank(pct=True)
    ranks.columns = [f"{c}_rank" for c in aggregate_cols]
    features = pd.concat([grouped[keys], grouped[aggregate_cols], ranks], axis=1)

    group_size = df.groupby(keys, sort=False).size().rename("groupSize").reset_index()
    match_size = df.groupby("matchId", sort=False).size().rename("matchSize")
    match_groups = grouped.groupby("matchId", sort=False).size().rename("observedGroups")
    features = features.merge(group_size, on=keys, how="left", validate="one_to_one")
    features = features.merge(match_size, on="matchId", how="left", validate="many_to_one")
    features = features.merge(match_groups, on="matchId", how="left", validate="many_to_one")
    features = features.assign(groupSizeRatio=features["groupSize"] / features["matchSize"])

    metadata_cols = keys + ["maxPlace", "numGroups"]
    if TARGET in df:
        metadata_cols.append(TARGET)
    metadata = df.groupby(keys, sort=False)[metadata_cols[2:]].first().reset_index()
    metadata = metadata.merge(group_size, on=keys, how="left", validate="one_to_one")

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features, metadata


def feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    return features.drop(columns=["matchId", "groupId"])


def make_model(name: str):
    if name == "hist_squared":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.06,
            max_iter=350,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=SEED,
        )
    if name == "hist_absolute":
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.06,
            max_iter=350,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=SEED,
        )
    if name == "hist_large":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=63,
            min_samples_leaf=15,
            l2_regularization=2.0,
            random_state=SEED,
        )
    if name == "extra":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_features=0.8,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        )
    raise ValueError(f"Unknown model: {name}")


def quantize(prediction: np.ndarray, max_place: np.ndarray) -> np.ndarray:
    prediction = np.clip(prediction, 0.0, 1.0)
    denominator = np.maximum(max_place - 1, 1)
    result = np.round(prediction * denominator) / denominator
    result[max_place == 1] = 1.0
    result[max_place == 0] = 0.0
    return result


def rank_prediction(raw: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    ranked = pd.DataFrame({"matchId": metadata["matchId"].values, "raw": raw})
    ranked["rank"] = ranked.groupby("matchId", sort=False)["raw"].rank(method="average") - 1
    counts = ranked.groupby("matchId", sort=False)["raw"].transform("size") - 1
    result = (ranked["rank"] / counts.replace(0, 1)).to_numpy()
    return result


def prediction_variants(raw: np.ndarray, metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    raw = np.clip(raw, 0.0, 1.0)
    ranked = rank_prediction(raw, metadata)
    variants = {"raw": raw, "raw_quantized": quantize(raw, metadata["maxPlace"].to_numpy())}
    variants["ranked"] = ranked
    variants["ranked_quantized"] = quantize(ranked, metadata["maxPlace"].to_numpy())
    for alpha in (0.25, 0.5, 0.75):
        blended = alpha * ranked + (1 - alpha) * raw
        variants[f"blend_{alpha:g}_quantized"] = quantize(
            blended, metadata["maxPlace"].to_numpy()
        )
    return variants


def weighted_mae(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average(np.abs(y - pred), weights=weight))


def validate(features: pd.DataFrame, metadata: pd.DataFrame, model_names: list[str]) -> None:
    matches = metadata["matchId"].drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(SEED)
    rng.shuffle(matches)
    valid_matches = set(matches[: max(1, len(matches) // 6)])
    valid_mask = metadata["matchId"].isin(valid_matches).to_numpy()
    x = feature_matrix(features)
    y = metadata[TARGET].to_numpy()
    weights = metadata["groupSize"].to_numpy()

    raw_predictions = {}
    for name in model_names:
        model = make_model(name)
        model.fit(x.loc[~valid_mask], y[~valid_mask], sample_weight=weights[~valid_mask])
        raw = model.predict(x.loc[valid_mask])
        raw_predictions[name] = raw
        valid_meta = metadata.loc[valid_mask].reset_index(drop=True)
        print(f"\n{name}")
        for variant, prediction in prediction_variants(raw, valid_meta).items():
            score = weighted_mae(y[valid_mask], prediction, weights[valid_mask])
            print(f"  {variant:24s} MAE={score:.6f}")

    if len(raw_predictions) > 1:
        best_single = min(
            raw_predictions,
            key=lambda name: weighted_mae(
                y[valid_mask],
                prediction_variants(raw_predictions[name], valid_meta)["ranked_quantized"],
                weights[valid_mask],
            ),
        )
        for other in raw_predictions:
            if other == best_single:
                continue
            print(f"\nensemble {best_single} + {other}")
            for alpha in (0.25, 0.5, 0.75):
                raw = alpha * raw_predictions[other] + (1 - alpha) * raw_predictions[best_single]
                prediction = prediction_variants(raw, valid_meta)["ranked_quantized"]
                score = weighted_mae(y[valid_mask], prediction, weights[valid_mask])
                print(f"  other_weight={alpha:g} MAE={score:.6f}")


def train_and_submit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_features: pd.DataFrame,
    train_meta: pd.DataFrame,
    test_features: pd.DataFrame,
    test_meta: pd.DataFrame,
    model_name: str,
    variant: str,
    output_path: Path,
) -> None:
    model = make_model(model_name)
    model.fit(
        feature_matrix(train_features),
        train_meta[TARGET],
        sample_weight=train_meta["groupSize"],
    )
    raw = model.predict(feature_matrix(test_features))
    variants = prediction_variants(raw, test_meta)
    if variant not in variants:
        raise ValueError(f"Unknown prediction variant {variant}; choose from {sorted(variants)}")
    group_predictions = test_meta[["matchId", "groupId"]].copy()
    group_predictions[TARGET] = variants[variant]
    submission = test[["Id", "matchId", "groupId"]].merge(
        group_predictions, on=["matchId", "groupId"], how="left", validate="many_to_one"
    )[["Id", TARGET]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission):,} predictions to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--model",
        choices=["hist_squared", "hist_absolute", "hist_large", "extra"],
        default="hist_large",
    )
    parser.add_argument("--variant", default="blend_0.75_quantized")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    train_features, train_meta = make_group_features(train)

    if args.validate or args.validate_all:
        names = ["hist_squared", "hist_absolute", "hist_large", "extra"] if args.validate_all else [args.model]
        validate(train_features, train_meta, names)
        if args.output is None:
            return

    test_features, test_meta = make_group_features(test)
    if list(feature_matrix(train_features).columns) != list(feature_matrix(test_features).columns):
        raise RuntimeError("Train and test feature columns differ")
    output = args.output or args.data_dir / "outputs" / "submission.csv"
    train_and_submit(
        train,
        test,
        train_features,
        train_meta,
        test_features,
        test_meta,
        args.model,
        args.variant,
        output,
    )


if __name__ == "__main__":
    main()
