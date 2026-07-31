"""Time-based CV for t5_bike.

Folds mimic the real task: train on the past, predict a ~2.5-month future block.
An extra diagnostic fold ("cold") trains on Mar-Sep and scores Dec-Feb; it is a
proxy for the temperature extrapolation the hidden test block (Sep -> Nov,
cooling to ~0 C) requires.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from models import make

FOLDS = [
    ("f1", "2018-04-01", "2018-06-15"),
    ("f2", "2018-06-16", "2018-08-31"),
    ("f3", "2018-07-02", "2018-09-18"),
]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def fit_predict(model, Xtr, ytr, Xte, transform="sqrt", w=None):
    if transform == "log":
        z = np.log1p(ytr)
    elif transform == "sqrt":
        z = np.sqrt(ytr)
    else:
        z = np.asarray(ytr, float)
    try:
        model.fit(Xtr, z, sample_weight=w) if w is not None else model.fit(Xtr, z)
    except TypeError:
        model.fit(Xtr, z)
    p = model.predict(Xte)
    if transform == "log":
        p = np.expm1(p)
    elif transform == "sqrt":
        p = np.maximum(p, 0) ** 2
    return np.clip(p, 0, None)


def prepare(task_dir=".", use_doy=False, use_season=False):
    tr_raw, te_raw = F.load(task_dir)
    f = F.build(tr_raw, te_raw, use_doy=use_doy, use_season=use_season)
    ftr, fte = F.split(f)
    ftr = ftr.reset_index(drop=True)
    fte = fte.reset_index(drop=True)
    y = tr_raw.sort_values("ts")["rented_bike_count"].reset_index(drop=True).values.astype(float)
    return ftr, fte, y


def folds_of(ftr, include_cold=True):
    jobs = []
    for name, s, e in FOLDS:
        s, e = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(hours=23)
        jobs.append((name, (ftr["ts"] < s).values, ((ftr["ts"] >= s) & (ftr["ts"] <= e)).values))
    if include_cold:
        s, e = pd.Timestamp("2017-12-01"), pd.Timestamp("2018-02-28 23:00")
        jobs.append(("cold", (ftr["ts"] > e).values,
                     ((ftr["ts"] >= s) & (ftr["ts"] <= e)).values))
    return jobs


def oof(model_name, transform, ftr, y, half_life=0, include_cold=True, seed=0, drop=()):
    """Return dict fold -> (val_mask, preds)."""
    ok = ftr["functioning"].values == 1
    cs = [c for c in F.cols(ftr) if not any(d in c for d in drop)]
    out = {}
    for fold, trm, vam in folds_of(ftr, include_cold):
        m = trm & ok
        w = None
        if half_life:
            age = (ftr.loc[m, "ts"].max() - ftr.loc[m, "ts"]).dt.days.values
            w = 0.5 ** (age / half_life)
        p = fit_predict(make(model_name, seed), ftr.loc[m, cs], y[m],
                        ftr.loc[vam, cs], transform, w)
        p = p * ftr.loc[vam, "functioning"].values
        out[fold] = (vam, p)
    return out


def score(out, y):
    r = {k: rmse(y[v[0]], v[1]) for k, v in out.items()}
    r["mean3"] = float(np.mean([r[k] for k in ("f1", "f2", "f3")]))
    return r


def run(configs, task_dir=".", include_cold=True):
    cache = {}
    rows = []
    for cfg in configs:
        key = (cfg.get("doy", False), cfg.get("season", False))
        if key not in cache:
            cache[key] = prepare(task_dir, use_doy=key[0], use_season=key[1])
        ftr, fte, y = cache[key]
        out = oof(cfg["model"], cfg.get("tf", "sqrt"), ftr, y,
                  cfg.get("hl", 0), include_cold, cfg.get("seed", 0), cfg.get("drop", ()))
        r = score(out, y)
        row = {"model": cfg["model"], "tf": cfg.get("tf", "sqrt"), "hl": cfg.get("hl", 0),
               "doy": int(key[0]), "season": int(key[1]),
               "drop": ",".join(cfg.get("drop", ())) or "-",
               **{k: round(v, 1) for k, v in r.items()}}
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows).sort_values("mean3")
    print("\n=== sorted ===")
    print(df.to_string(index=False), flush=True)
    return df


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "b2"
    if which == "b2":
        cfgs = [{"model": m, "tf": "sqrt"} for m in
                ["hgb_L7_i1500", "hgb_L10_i1200", "hgb_L15_i800", "hgb_L20_i800",
                 "hgb_L15_i800_m60", "hgb_L7_i1500_m60", "et", "et_l5", "rf", "spline"]]
        cfgs += [{"model": "hgb_L7_i1500", "tf": "log"},
                 {"model": "hgb_L10_i1200", "tf": "log"},
                 {"model": "hgb_L10_i1200", "tf": "sqrt", "doy": True},
                 {"model": "hgb_L10_i1200", "tf": "sqrt", "season": True},
                 {"model": "hgb_L10_i1200", "tf": "sqrt", "hl": 120},
                 {"model": "hgb_L10_i1200", "tf": "sqrt", "hl": 240}]
        run(cfgs)
