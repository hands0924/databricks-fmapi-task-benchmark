"""PUBG winPlacePerc prediction - improved solution with feature engineering.

Pipeline:
  1. Per-player engineered features + match-normalized features + group-aggregate
     features (mean/sum/min/max + team size).
  2. 5-fold GroupKFold (by matchId) HistGradientBoostingRegressor, ensemble of 2 seeds.
  3. Average OOF / test predictions per groupId (target is constant per team).
  4. Clip to [0,1]; write submission matching sample_submission exactly.

OOF MAE (player) ~0.0395, (group) ~0.0394. Runtime ~7-8 min on 4 cores.

Run:
    python solution/main.py
Writes ../outputs/submission.csv (Id, winPlacePerc) matching sample_submission.csv.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

# Resolve paths relative to this script so it is reproducible from any cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
TARGET = "winPlacePerc"
SEEDS = [42, 2024]

NUM_FEATS = [
    "assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals",
    "killPlace", "killPoints", "kills", "killStreaks", "longestKill",
    "matchDuration", "maxPlace", "numGroups", "rankPoints", "revives",
    "rideDistance", "roadKills", "swimDistance", "teamKills",
    "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPoints",
]


def add_features(df, match_stats=None):
    df = df.copy()
    df["totalDistance"] = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    df["items"] = df["boosts"] + df["heals"]
    df["killsPerWalk"] = df["kills"] / (df["walkDistance"] + 1.0)
    df["dmgPerKill"] = df["damageDealt"] / (df["kills"] + 1.0)
    df["headshotRate"] = df["headshotKills"] / (df["kills"] + 1.0)
    if match_stats is None:
        match_stats = df.groupby("matchId")[NUM_FEATS].agg(["mean", "std"])
        match_stats.columns = ["_".join(c) for c in match_stats.columns]
    df = df.merge(match_stats, left_on="matchId", right_index=True, how="left")
    for f in NUM_FEATS:
        mu = df[f"{f}_mean"]
        sd = df[f"{f}_std"].fillna(0.0)
        df[f"{f}_mN"] = (df[f] - mu) / (sd + 1e-6)
    return df, match_stats


def add_group_aggs(df):
    grp = df.groupby("groupId")
    agg = grp[NUM_FEATS + ["totalDistance", "items"]].agg(["mean", "sum", "min", "max"])
    agg.columns = ["grp_" + "_".join(c) for c in agg.columns]
    agg["grp_size"] = grp.size()
    return df.merge(agg, left_on="groupId", right_index=True, how="left")


def main():
    t0 = time.time()
    train = pd.read_csv(f"{BASE}/train.csv")
    test = pd.read_csv(f"{BASE}/test.csv")
    sub = pd.read_csv(f"{BASE}/sample_submission.csv")
    y = train[TARGET].values
    print(f"load {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    train_fe, match_stats = add_features(train, match_stats=None)
    test_fe, _ = add_features(test, match_stats=match_stats)
    train_fe = add_group_aggs(train_fe)
    test_fe = add_group_aggs(test_fe)
    print(f"FE {time.time()-t1:.1f}s cols={train_fe.shape[1]}", flush=True)

    drop = ["Id", "groupId", "matchId", "matchType", TARGET]
    feat_cols = [c for c in train_fe.columns
                 if c not in drop and c in test_fe.columns]
    X = train_fe[feat_cols].values
    X_test = test_fe[feat_cols].values
    groups = train_fe["matchId"].values
    print(f"features={len(feat_cols)} train={X.shape[0]} test={X_test.shape[0]}", flush=True)

    n_splits = 5
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    kf = GroupKFold(n_splits=n_splits)
    folds = list(kf.split(X, y, groups))
    n_models = n_splits * len(SEEDS)
    for seed in SEEDS:
        for fold, (tr, va) in enumerate(folds):
            tf = time.time()
            m = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                l2_regularization=1.0, min_samples_leaf=40,
                early_stopping=False, random_state=seed,
            )
            m.fit(X[tr], y[tr])
            oof[va] += m.predict(X[va]) / len(SEEDS)
            test_pred += m.predict(X_test) / n_models
            print(f"seed {seed} fold {fold} tr={len(tr)} va={len(va)} {time.time()-tf:.1f}s", flush=True)

    oof = np.clip(oof, 0.0, 1.0)
    test_pred = np.clip(test_pred, 0.0, 1.0)
    print(f"OOF MAE (player): {mean_absolute_error(y, oof):.5f}", flush=True)

    train_fe["oof"] = oof
    grp_oof = train_fe.groupby("groupId")["oof"].mean()
    grp_y = train.groupby("groupId")[TARGET].mean()
    print(f"OOF MAE (group): {mean_absolute_error(grp_y, grp_oof):.5f}", flush=True)

    test_fe["pred"] = test_pred
    test_grp = test_fe.groupby("groupId")["pred"].mean()
    test_fe["pred"] = test_fe["groupId"].map(test_grp)
    preds = np.clip(test_fe["pred"].values, 0.0, 1.0)

    out = sub.copy()
    out["winPlacePerc"] = preds
    out.to_csv(f"{BASE}/outputs/submission.csv", index=False)
    print(f"saved {out.shape} mean={preds.mean():.4f} total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
