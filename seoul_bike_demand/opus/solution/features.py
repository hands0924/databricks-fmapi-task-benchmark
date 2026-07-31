"""Feature engineering for Seoul bike hourly demand (t5_bike).

Only columns available in both train.csv and test.csv are used, so every
feature is computable for the future test period (no target lags, no leakage).
Rolling / daily weather aggregates are computed over the concatenated
train+test weather timeline, which is legitimate because test weather columns
are provided as inputs.
"""
import numpy as np
import pandas as pd

WEATHER = ["temperature_c", "humidity_pct", "wind_speed_ms", "visibility_10m",
           "dew_point_c", "solar_radiation_mj", "rainfall_mm", "snowfall_cm"]


def load(task_dir="."):
    tr = pd.read_csv(f"{task_dir}/train.csv")
    te = pd.read_csv(f"{task_dir}/test.csv")
    for df in (tr, te):
        df["d"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
        df["ts"] = df["d"] + pd.to_timedelta(df["hour"], unit="h")
    return tr, te


def build(tr, te, use_doy=False, use_season=False):
    """Return (Xtr, Xte) aligned feature frames."""
    tr = tr.copy()
    tr["_part"] = 0
    te = te.copy()
    te["_part"] = 1
    a = pd.concat([tr.drop(columns=[c for c in ["rented_bike_count"] if c in tr]), te],
                  ignore_index=True).sort_values("ts").reset_index(drop=True)

    f = pd.DataFrame(index=a.index)
    f["hour"] = a["hour"]
    f["hour_sin"] = np.sin(2 * np.pi * a["hour"] / 24)
    f["hour_cos"] = np.cos(2 * np.pi * a["hour"] / 24)
    dow = a["d"].dt.dayofweek
    f["dow"] = dow
    f["is_weekend"] = (dow >= 5).astype(int)
    f["holiday"] = (a["holiday"] == "Holiday").astype(int)
    # non-working day = weekend or holiday (strong driver of the hourly profile)
    f["nonwork"] = ((dow >= 5) | (a["holiday"] == "Holiday")).astype(int)
    f["hour_x_nonwork"] = f["hour"] + 24 * f["nonwork"]

    for c in WEATHER:
        f[c] = a[c].values

    # weather derivatives
    f["temp_dew_gap"] = a["temperature_c"] - a["dew_point_c"]
    f["rain_flag"] = (a["rainfall_mm"] > 0).astype(int)
    f["snow_flag"] = (a["snowfall_cm"] > 0).astype(int)
    f["log_rain"] = np.log1p(a["rainfall_mm"])
    # wind chill / apparent temperature style term
    f["feels"] = (13.12 + 0.6215 * a["temperature_c"]
                  - 11.37 * np.power(np.maximum(a["wind_speed_ms"] * 3.6, 0.1), 0.16)
                  + 0.3965 * a["temperature_c"]
                  * np.power(np.maximum(a["wind_speed_ms"] * 3.6, 0.1), 0.16))
    f["discomfort"] = (a["temperature_c"] - 0.55 * (1 - a["humidity_pct"] / 100)
                       * (a["temperature_c"] - 14.5))

    # rolling weather context (past window only, ordered in time)
    for w in (3, 6, 12):
        f[f"rain_roll{w}"] = a["rainfall_mm"].rolling(w, min_periods=1).sum().values
        f[f"temp_roll{w}"] = a["temperature_c"].rolling(w, min_periods=1).mean().values
    f["temp_diff1"] = a["temperature_c"].diff().fillna(0).values
    f["temp_diff24"] = a["temperature_c"].diff(24).fillna(0).values

    # daily aggregates (same calendar day; weather is a given input for test too)
    g = a.groupby("d")
    day = pd.DataFrame({
        "day_temp_mean": g["temperature_c"].mean(),
        "day_temp_max": g["temperature_c"].max(),
        "day_temp_min": g["temperature_c"].min(),
        "day_hum_mean": g["humidity_pct"].mean(),
        "day_rain_sum": g["rainfall_mm"].sum(),
        "day_snow_sum": g["snowfall_cm"].sum(),
        "day_sol_sum": g["solar_radiation_mj"].sum(),
        "day_wind_mean": g["wind_speed_ms"].mean(),
        "day_vis_mean": g["visibility_10m"].mean(),
    })
    day["day_rain_flag"] = (day["day_rain_sum"] > 0).astype(int)
    f = pd.concat([f, a[["d"]].join(day, on="d").drop(columns="d")], axis=1)

    if use_doy:
        doy = a["d"].dt.dayofyear
        f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if use_season:
        f["season"] = a["seasons"].map({"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3})

    f = f.astype(float)
    f["_part"] = a["_part"].values
    f["ts"] = a["ts"].values
    f["functioning"] = (a["functioning_day"] == "Yes").astype(int).values
    return f


def split(f):
    tr = f[f["_part"] == 0].copy()
    te = f[f["_part"] == 1].copy()
    return tr, te


FEAT_DROP = ["_part", "ts", "functioning"]


def cols(f):
    return [c for c in f.columns if c not in FEAT_DROP]
