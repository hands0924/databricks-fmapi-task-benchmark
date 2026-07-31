#!/usr/bin/env python3
"""
Seoul Public Bike Demand Prediction - t5_bike (RMSE).

Approach
---------
- Calendar + weather feature engineering.
- Train-derived, leakage-free historical aggregates (group-by means) which are
  strong baselines for the autumn test window.
- LightGBM on log1p target; chronological holdout (last 14 days of train).
- Hard rule: functioning_day == "No" -> target exactly 0 (verified in train).
- A light residual blend with the strongest aggregate (hour,seasons,is_weekend)
  mean keeps predictions calibrated to the autumn demand level, avoiding the
  shrinkage that pure GBM produces on the unseen test distribution.

Reproducible: fixed seed, no internet, no external data.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(ROOT, "train.csv")
TEST = os.path.join(ROOT, "test.csv")
OUT = os.path.join(ROOT, "outputs", "submission.csv")

SEED = 42
TARGET = "rented_bike_count"


def parse_id(id_s):
    date_part, hh = str(id_s).split("_")
    y, m, d = int(date_part[:4]), int(date_part[4:6]), int(date_part[6:8])
    return pd.Timestamp(year=y, month=m, day=d, hour=int(hh))


def add_base_features(df):
    df = df.copy()
    dt = df["id"].map(parse_id)
    df["datetime"] = dt
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["dayofyear"] = dt.dt.dayofyear
    df["week"] = dt.dt.isocalendar().week.astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_holiday"] = (df["holiday"] == "Holiday").astype(int)
    df["is_functioning"] = (df["functioning_day"] == "Yes").astype(int)
    df["is_winter"] = (df["seasons"] == "Winter").astype(int)
    df["is_summer"] = (df["seasons"] == "Summer").astype(int)
    df["is_spring"] = (df["seasons"] == "Spring").astype(int)
    df["is_autumn"] = (df["seasons"] == "Autumn").astype(int)

    df["temp_humid"] = df["temperature_c"] * df["humidity_pct"] / 100.0
    df["feels_like"] = df["temperature_c"] - (df["wind_speed_ms"] * 0.5)
    df["discomfort"] = df["temperature_c"] + df["humidity_pct"] / 20.0
    df["temp_sq"] = df["temperature_c"] ** 2
    df["is_rainy"] = (df["rainfall_mm"] > 0).astype(int)
    df["is_snowy"] = (df["snowfall_cm"] > 0).astype(int)
    df["rain_intensity"] = np.log1p(df["rainfall_mm"])
    df["snow_intensity"] = np.log1p(df["snowfall_cm"])
    df["has_solar"] = (df["solar_radiation_mj"] > 0).astype(int)
    df["visibility_low"] = (df["visibility_10m"] < 2000).astype(int)
    df["dew_point_spread"] = df["temperature_c"] - df["dew_point_c"]

    df["is_rush_am"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_rush_pm"] = df["hour"].isin([17, 18, 19]).astype(int)
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)
    df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
    df["is_lunchtime"] = df["hour"].isin([12, 13]).astype(int)

    df["temp_x_rusham"] = df["temperature_c"] * df["is_rush_am"]
    df["temp_x_rushpm"] = df["temperature_c"] * df["is_rush_pm"]
    df["rain_x_rusham"] = df["rain_intensity"] * df["is_rush_am"]
    df["rain_x_rushpm"] = df["rain_intensity"] * df["is_rush_pm"]
    return df


def add_hist_aggregate_features(df, train_df):
    """Leakage-free aggregates computed ONLY from train, joined to any frame."""
    t = train_df.copy()
    out = df.copy()

    groups = [
        (["hour", "seasons", "is_weekend"], "hsw"),
        (["hour", "is_holiday", "is_weekend"], "hhw"),
        (["hour", "dayofweek"], "hd"),
        (["hour", "seasons"], "hs"),
        (["hour", "is_functioning"], "hf"),
        (["seasons", "dayofweek"], "sd"),
        (["hour", "seasons", "is_holiday"], "hsh"),
    ]
    for keys, name in groups:
        g = t.groupby(keys, observed=True)[TARGET].agg(["mean", "median"]).reset_index()
        g = g.rename(columns={"mean": f"{name}_mean", "median": f"{name}_median"})
        out = out.merge(g, on=keys, how="left")

    gh = t.groupby("hour", observed=True)[TARGET].mean().reset_index().rename(
        columns={TARGET: "hour_global_mean"})
    out = out.merge(gh, on="hour", how="left")

    # Recent same-hour means from train tail (autumn-leaning since train ends Sep 18).
    for days, cname in [(14, "hour_recent14_mean"), (7, "hour_recent7_mean"),
                       (30, "hour_recent30_mean")]:
        cutoff = t["datetime"].max() - pd.Timedelta(days=days)
        recent = t[t["datetime"] >= cutoff]
        g = recent.groupby("hour", observed=True)[TARGET].mean().reset_index().rename(
            columns={TARGET: cname})
        out = out.merge(g, on="hour", how="left")

    # Recent same-(hour,seasons) mean over a longer window gives autumn-only trend.
    # Since test is entirely autumn, an autumn-specific aggregate is strongest.
    g_aut = t[t["seasons"] == "Autumn"].groupby(["hour", "is_weekend"], observed=True)[TARGET].mean().reset_index().rename(
        columns={TARGET: "autumn_hw_mean"})
    out = out.merge(g_aut, on=["hour", "is_weekend"], how="left")

    return out


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def main():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)

    train = add_base_features(train)
    test = add_base_features(test)
    train = add_hist_aggregate_features(train, train)
    test = add_hist_aggregate_features(test, train)

    feat_cols = [
        "hour", "temperature_c", "humidity_pct", "wind_speed_ms",
        "visibility_10m", "dew_point_c", "solar_radiation_mj",
        "rainfall_mm", "snowfall_cm",
        "day", "dayofweek", "is_weekend", "dayofyear", "week",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
        "is_holiday", "is_functioning", "is_winter", "is_summer",
        "is_spring", "is_autumn",
        "temp_humid", "feels_like", "discomfort", "temp_sq", "dew_point_spread",
        "is_rainy", "is_snowy", "rain_intensity", "snow_intensity",
        "has_solar", "visibility_low",
        "is_rush_am", "is_rush_pm", "is_night", "is_daytime", "is_lunchtime",
        "temp_x_rusham", "temp_x_rushpm", "rain_x_rusham", "rain_x_rushpm",
        "hsw_mean", "hsw_median", "hhw_mean", "hhw_median",
        "hd_mean", "hd_median", "hs_mean", "hs_median",
        "hf_mean", "hf_median", "sd_mean", "sd_median",
        "hsh_mean", "hsh_median",
        "hour_global_mean", "hour_recent14_mean", "hour_recent7_mean",
        "hour_recent30_mean", "autumn_hw_mean",
    ]

    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.03, num_leaves=48, min_data_in_leaf=40,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=1.0, seed=SEED, verbose=-1, n_jobs=-1,
    )

    # Chronological holdout: last 14 days (Sep 5 - Sep 18, 2018) -- autumn, matches test season.
    holdout_start = train["datetime"].max() - pd.Timedelta(days=14)
    val = train[train["datetime"] > holdout_start].copy()
    fit = train[train["datetime"] <= holdout_start].copy()

    dfit = lgb.Dataset(fit[feat_cols], np.log1p(fit[TARGET].values))
    dval = lgb.Dataset(val[feat_cols], np.log1p(val[TARGET].values), reference=dfit)
    model = lgb.train(params, dfit, num_boost_round=3000,
                      valid_sets=[dfit, dval], valid_names=["fit", "val"],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])

    val_pred = np.expm1(model.predict(val[feat_cols], num_iteration=model.best_iteration))
    val_pred = np.clip(val_pred, 0, None)
    val_pred = np.where(val["is_functioning"].values == 0, 0.0, val_pred)
    rmse_gbm = rmse(val[TARGET].values, val_pred)
    print(f"[GBM-VAL] RMSE (holdout 14d): {rmse_gbm:.4f}  best_iter={model.best_iteration}")

    # Second holdout (Summer: Aug 1 - Aug 14) for a distribution-robust RMSE estimate.
    sum_val = train[(train["datetime"] >= pd.Timestamp(2018, 8, 1)) &
                    (train["datetime"] < pd.Timestamp(2018, 8, 15))].copy()
    sum_fit = train[train["datetime"] < pd.Timestamp(2018, 8, 1)].copy()
    dfit2 = lgb.Dataset(sum_fit[feat_cols], np.log1p(sum_fit[TARGET].values))
    dval2 = lgb.Dataset(sum_val[feat_cols], np.log1p(sum_val[TARGET].values), reference=dfit2)
    m2 = lgb.train(params, dfit2, num_boost_round=3000,
                   valid_sets=[dfit2, dval2], valid_names=["fit", "val"],
                   callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    sv = np.expm1(m2.predict(sum_val[feat_cols], num_iteration=m2.best_iteration))
    sv = np.clip(sv, 0, None); sv = np.where(sum_val["is_functioning"].values == 0, 0.0, sv)
    rmse_sum = rmse(sum_val[TARGET].values, sv)
    print(f"[GBM-VAL] RMSE (summer holdout): {rmse_sum:.4f}  best_iter={m2.best_iteration}")

    # Baseline aggregates on the autumn holdout
    rmse_agg = rmse(val[TARGET].values, np.where(val["is_functioning"].values == 0, 0.0,
                                                 val["hsw_mean"].values))
    print(f"[AGG-VAL] RMSE (hsw_mean baseline): {rmse_agg:.4f}")
    rmse_aut = rmse(val[TARGET].values, np.where(val["is_functioning"].values == 0, 0.0,
                                                val["autumn_hw_mean"].values))
    print(f"[AGG-VAL] RMSE (autumn_hw_mean baseline): {rmse_aut:.4f}")

    # Blend grid search on the autumn holdout: gbm vs aggregate baseline
    best_blend = None
    agg_base = val["hsw_mean"].values.copy()
    for w in np.linspace(0.0, 1.0, 21):
        bp = w * val_pred + (1 - w) * agg_base
        bp = np.where(val["is_functioning"].values == 0, 0.0, bp)
        r = rmse(val[TARGET].values, bp)
        if best_blend is None or r < best_blend[0]:
            best_blend = (r, float(w))
    print(f"[BLEND-VAL] best blend w_gbm={best_blend[1]:.2f} -> RMSE {best_blend[0]:.4f}")

    # ---- Retrain on all train data with multi-seed ensemble for stability ----
    best_iter = model.best_iteration
    n_rounds = max(int(best_iter * 1.1), 100)
    seeds = [SEED, 7, 123, 2024]
    test_preds = []
    for sd in seeds:
        p = dict(params); p["seed"] = sd; p["bagging_seed"] = sd
        dfit_full = lgb.Dataset(train[feat_cols], np.log1p(train[TARGET].values))
        final = lgb.train(p, dfit_full, num_boost_round=n_rounds,
                          valid_sets=[dfit_full], valid_names=["fit"],
                          callbacks=[lgb.log_evaluation(0)])
        test_preds.append(np.expm1(final.predict(test[feat_cols])))

    test_pred = np.mean(test_preds, axis=0)
    test_pred = np.clip(test_pred, 0, None)

    # Blend with aggregate baseline using holdout-derived weight
    w = best_blend[1]
    agg_base_test = test["hsw_mean"].values.copy()
    test_pred = w * test_pred + (1 - w) * agg_base_test

    # Hard rule: functioning_day == "No" -> exactly 0
    test_pred = np.where(test["is_functioning"].values == 0, 0.0, test_pred)

    sub = pd.DataFrame({"id": test["id"].values, TARGET: test_pred})
    sub = sub.sort_values("id").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sub.to_csv(OUT, index=False)
    print(f"[SUB] wrote {len(sub)} rows -> {OUT}")
    print(sub.head())
    print(sub[TARGET].describe())

    meta = {
        "val_rmse_gbm_autumn": float(rmse_gbm),
        "val_rmse_gbm_summer": float(rmse_sum),
        "val_rmse_agg_hsw": float(rmse_agg),
        "val_rmse_agg_autumn": float(rmse_aut),
        "val_rmse_blend": float(best_blend[0]),
        "blend_w_gbm": float(best_blend[1]),
        "best_iteration": int(best_iter),
        "n_features": len(feat_cols),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "seeds": seeds,
    }
    with open(os.path.join(HERE, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("meta:", meta)


if __name__ == "__main__":
    main()
