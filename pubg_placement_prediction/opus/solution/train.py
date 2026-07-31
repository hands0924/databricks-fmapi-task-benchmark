"""Train LightGBM at group level, post-process, and write submission.

Usage:
    python solution/train.py            # full run -> outputs/submission.csv
    python solution/train.py --cv-only  # CV diagnostics only
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_group_features, feature_columns  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
N_FOLDS = 5

PARAMS = dict(
    objective="mae",
    metric="mae",
    learning_rate=0.07,
    num_leaves=255,
    min_data_in_leaf=20,
    feature_fraction=0.6,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    num_threads=4,
    verbosity=-1,
    seed=SEED,
)
NUM_ROUNDS = 1600
EARLY_STOP = 100


# --------------------------------------------------------------------------- #
# post-processing
# --------------------------------------------------------------------------- #
BLEND_ALPHA = 0.72  # tuned on OOF (flat optimum over 0.65-0.78)


def postprocess(pred, meta, mode="blend_grid", alpha=BLEND_ALPHA):
    """pred: group-level raw predictions. meta: DataFrame with matchId, maxPlace,
    numGroupsActual. Returns adjusted group-level predictions."""
    df = meta[["matchId", "maxPlace", "numGroupsActual"]].copy()
    df["p"] = np.clip(pred, 0.0, 1.0)
    raw = df["p"].values

    if mode == "raw":
        return raw

    if mode == "grid":
        return _snap_grid(raw, df["maxPlace"].values)

    # rank-based: groups in a match take evenly spaced placement percentiles
    r = df.groupby("matchId", sort=False)["p"].rank(method="first") - 1.0
    n = df["numGroupsActual"].values
    adj = np.where(n > 1, r.values / np.maximum(n - 1.0, 1.0), raw)

    if mode == "rank":
        return np.clip(adj, 0, 1)
    if mode == "rank_grid":
        return _snap_grid(np.clip(adj, 0, 1), df["maxPlace"].values)
    # blend_grid: shrink the (order-preserving) rank mapping toward the
    # calibrated raw prediction, then snap to the achievable grid.
    bl = np.where(n > 1, alpha * adj + (1.0 - alpha) * raw, raw)
    return _snap_grid(np.clip(bl, 0, 1), df["maxPlace"].values)


def _snap_grid(p, maxPlace):
    """Snap to the nearest achievable 1/(maxPlace-1) grid point."""
    out = p.copy()
    mp = maxPlace.astype(np.float64)
    ok = mp > 1
    gap = np.where(ok, 1.0 / np.maximum(mp - 1.0, 1.0), 1.0)
    out = np.where(ok, np.round(p / gap) * gap, p)
    out = np.where(mp == 0, 0.0, out)
    out = np.where(mp == 1, 1.0, out)
    return np.clip(out, 0.0, 1.0)


def player_mae(group_pred, grp, y_col="winPlacePerc"):
    """MAE weighted by group size == player-level MAE."""
    w = grp["groupSize"].values.astype(np.float64)
    return np.sum(w * np.abs(group_pred - grp[y_col].values)) / w.sum()


# --------------------------------------------------------------------------- #
def load_features(cache=True):
    cdir = os.path.join(ROOT, "solution", ".cache")
    os.makedirs(cdir, exist_ok=True)
    ftr, fte = os.path.join(cdir, "gtr.pkl"), os.path.join(cdir, "gte.pkl")
    if cache and os.path.exists(ftr) and os.path.exists(fte):
        return pd.read_pickle(ftr), pd.read_pickle(fte)
    gtr = build_group_features(pd.read_csv(os.path.join(ROOT, "train.csv")), True)
    gte = build_group_features(pd.read_csv(os.path.join(ROOT, "test.csv")), False)
    if cache:
        gtr.to_pickle(ftr)
        gte.to_pickle(fte)
    return gtr, gte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-only", action="store_true")
    ap.add_argument("--from-cache", action="store_true",
                    help="reuse saved oof/test predictions, only redo post-processing")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    args = ap.parse_args()

    t0 = time.time()
    gtr, gte = load_features()

    if args.from_cache:
        cdir = os.path.join(ROOT, "solution", ".cache")
        oof = np.load(os.path.join(cdir, "oof.npy"))
        test_pred = np.load(os.path.join(cdir, "test_raw.npy"))
        best_mode = report_modes(oof, gtr)
        write_submission(test_pred, gte, best_mode)
        return

    fc = feature_columns(gtr)
    print(f"features={len(fc)} train_groups={len(gtr)} test_groups={len(gte)}")

    X = gtr[fc].values.astype(np.float32)
    y = gtr["winPlacePerc"].values.astype(np.float64)
    Xte = gte[fc].values.astype(np.float32)

    gkf = GroupKFold(n_splits=args.folds)
    oof = np.zeros(len(gtr))
    test_pred = np.zeros(len(gte))
    best_iters = []

    for k, (itr, iva) in enumerate(gkf.split(X, y, groups=gtr["matchId"])):
        dtr = lgb.Dataset(X[itr], y[itr], feature_name=fc)
        dva = lgb.Dataset(X[iva], y[iva], feature_name=fc, reference=dtr)
        m = lgb.train(
            PARAMS, dtr, num_boost_round=args.rounds, valid_sets=[dva],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(500)],
        )
        best_iters.append(m.best_iteration)
        oof[iva] = m.predict(X[iva], num_iteration=m.best_iteration)
        if not args.cv_only:
            test_pred += m.predict(Xte, num_iteration=m.best_iteration) / args.folds
        print(f"fold {k}: best_iter={m.best_iteration} "
              f"raw_mae={player_mae(np.clip(oof[iva], 0, 1), gtr.iloc[iva]):.5f} "
              f"[{time.time()-t0:.0f}s]")

    best_mode = report_modes(oof, gtr)
    print(f"mean best_iter: {np.mean(best_iters):.0f}")

    np.save(os.path.join(ROOT, "solution", ".cache", "oof.npy"), oof)

    if args.cv_only:
        return

    np.save(os.path.join(ROOT, "solution", ".cache", "test_raw.npy"), test_pred)
    write_submission(test_pred, gte, best_mode)
    print(f"done in {time.time()-t0:.0f}s")


def report_modes(oof, gtr):
    print("\n=== OOF player-level MAE ===")
    results = {}
    for mode in ["raw", "grid", "rank", "rank_grid", "blend_grid"]:
        results[mode] = player_mae(postprocess(oof, gtr, mode), gtr)
        print(f"{mode:>10}: {results[mode]:.6f}")
    best = min(results, key=results.get)
    print(f"best mode: {best}")
    return best


def write_submission(test_pred, gte, mode):
    p = postprocess(test_pred, gte, mode)
    gpred = gte[["matchId", "groupId"]].copy()
    gpred["winPlacePerc"] = p
    te_ids = pd.read_csv(os.path.join(ROOT, "test.csv"),
                         usecols=["Id", "matchId", "groupId"])
    out = te_ids.merge(gpred, on=["matchId", "groupId"], how="left")
    assert out["winPlacePerc"].notna().all(), "missing predictions"
    out = out[["Id", "winPlacePerc"]]
    sample = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    assert len(out) == len(sample) and set(out.Id) == set(sample.Id)
    assert out.Id.duplicated().sum() == 0
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
    print(f"wrote outputs/submission.csv rows={len(out)} mode={mode} "
          f"range=[{out.winPlacePerc.min():.4f},{out.winPlacePerc.max():.4f}]")


if __name__ == "__main__":
    main()
