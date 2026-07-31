"""Ablation + blend evaluation on the time-based folds."""
import os
import sys
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv
import features as F

ABLATIONS = [
    ("none", ()),
    ("no_roll", ("_roll", "temp_diff")),
    ("no_day", ("day_",)),
    ("no_deriv", ("feels", "discomfort", "temp_dew_gap", "log_rain")),
    ("no_cyc", ("hour_sin", "hour_cos")),
    ("no_hxn", ("hour_x_nonwork",)),
    ("no_vis", ("visibility", "day_vis")),
]

CANDIDATES = [
    ("hgb_L15_i800", "sqrt", 0),
    ("hgb_L10_i1200", "sqrt", 0),
    ("hgb_L10_i1200", "log", 0),
    ("hgb_L20_i800", "sqrt", 0),
    ("et", "sqrt", 0),
    ("rf", "sqrt", 0),
    ("gbr", "sqrt", 0),
]


def main(mode):
    ftr, fte, y = cv.prepare(".")
    if mode == "ablate":
        cfgs = []
        for nm, dr in ABLATIONS:
            cfgs.append({"model": "hgb_L15_i800", "tf": "sqrt", "drop": dr})
            cfgs.append({"model": "hgb_L10_i1200", "tf": "sqrt", "drop": dr})
        cv.run(cfgs)
        return

    if mode == "blend":
        preds, names = {}, []
        for mn, tf, hl in CANDIDATES:
            key = f"{mn}|{tf}"
            o = cv.oof(mn, tf, ftr, y, hl)
            preds[key] = o
            names.append(key)
            print(key, {k: round(v, 1) for k, v in cv.score(o, y).items()}, flush=True)
        np.save("/tmp/opencode/oof.npy", np.array([1]))  # marker
        rows = []
        for r in range(2, len(names) + 1):
            for combo in itertools.combinations(names, r):
                avg = {}
                for fold in preds[names[0]]:
                    vam = preds[names[0]][fold][0]
                    p = np.mean([preds[c][fold][1] for c in combo], axis=0)
                    avg[fold] = (vam, p)
                s = cv.score(avg, y)
                rows.append({"combo": "+".join(combo), **{k: round(v, 1) for k, v in s.items()}})
        df = pd.DataFrame(rows).sort_values("mean3")
        print(df.head(25).to_string(index=False), flush=True)
        df.to_csv("/tmp/opencode/blend.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
