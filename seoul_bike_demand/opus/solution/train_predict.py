"""Final model for t5_bike: train on train.csv, predict test.csv.

Usage:  python solution/train_predict.py [task_dir]

Design notes (decisions backed by the time-based CV in solution/cv.py):
  * Target transform: sqrt (RMSE-friendly variance stabilisation; beat log1p/raw).
  * Rows with functioning_day == "No" are removed from training and their
    prediction is hard-set to 0 (in train the target is 0 for all 72 such rows;
    the test block contains 223 of them).
  * Features: per-hour weather + calendar + a few weather derivatives.
    Daily aggregates / rolling windows / temp deltas were dropped: they
    overfit the small (7k row) training set (CV mean3 394 -> 369).
  * seasons and day-of-year are NOT used: the test block is entirely Autumn and
    covers Oct/Nov, which never appear in train, so those features cannot
    extrapolate (CV confirmed they hurt).
  * Final prediction = weighted blend of 4 diverse learners (2x HistGB, ExtraTrees,
    GradientBoosting). CV mean3 RMSE: best single 369.2 -> blend 367.0, and the
    blend is the more robust choice when extrapolating into an unseen season.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from models import make
from cv import fit_predict

DROP = ("day_", "_roll", "temp_diff")
# (model, transform, seeds, weight); seeds are averaged inside an entry
ENSEMBLE = [
    ("hgb_L10_i1200", "sqrt", (0,), 0.30),
    ("hgb_L15_i800", "sqrt", (0,), 0.25),
    ("et", "sqrt", (0, 1, 2), 0.25),
    ("gbr", "sqrt", (0,), 0.20),
]


def main(task_dir="."):
    tr_raw, te_raw = F.load(task_dir)
    f = F.build(tr_raw, te_raw)
    ftr, fte = F.split(f)
    ftr = ftr.reset_index(drop=True)
    fte = fte.reset_index(drop=True)
    y = tr_raw.sort_values("ts")["rented_bike_count"].reset_index(drop=True).values.astype(float)

    cs = [c for c in F.cols(ftr) if not any(d in c for d in DROP)]
    ok = ftr["functioning"].values == 1
    Xtr, ytr, Xte = ftr.loc[ok, cs], y[ok], fte[cs]
    print(f"train rows={len(Xtr)} (dropped {(~ok).sum()} non-functioning) "
          f"features={len(cs)} test rows={len(Xte)}")

    pred = np.zeros(len(Xte))
    wsum = 0.0
    for mn, tf, seeds, w in ENSEMBLE:
        ps = []
        for sd in seeds:
            ps.append(fit_predict(make(mn, sd), Xtr, ytr, Xte, tf))
        p = np.mean(ps, axis=0)
        print(f"  {mn} {tf} seeds={list(seeds)} w={w} mean={p.mean():.1f}", flush=True)
        pred += w * p
        wsum += w
    pred /= wsum

    # non-functioning hours have zero rentals by construction
    pred = np.clip(pred, 0, None) * fte["functioning"].values
    # cap at a sane level: never exceed the observed historical maximum
    pred = np.minimum(pred, y.max())

    ids = te_raw.sort_values("ts")["id"].values
    sub = pd.DataFrame({"id": ids, "rented_bike_count": np.round(pred, 3)})
    # restore the original test.csv row order
    sub = te_raw[["id"]].merge(sub, on="id", how="left")
    assert len(sub) == len(te_raw) and sub["rented_bike_count"].notna().all()
    assert sub["id"].is_unique and set(sub["id"]) == set(te_raw["id"])

    out_dir = os.path.join(task_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    sub.to_csv(os.path.join(out_dir, "submission.csv"), index=False)
    print(f"wrote {out_dir}/submission.csv  rows={len(sub)} "
          f"mean={sub.rented_bike_count.mean():.1f} max={sub.rented_bike_count.max():.1f} "
          f"zeros={(sub.rented_bike_count == 0).sum()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
