"""Final blend: combine level-1 OOF/test predictions from both model sets
(plain features and swap-augmented features) via NNLS + ridge stacking.

Reads work/level1_v2.npz (plain) and work/level1_v3aug.npz (augmented),
selects the best combiner by 5-fold-consistent OOF Pearson, and writes
outputs/submission.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work")

te = pd.read_csv(os.path.join(ROOT, "test.csv"))
parts = []
for tag, fn in (("p", "level1_v2.npz"), ("a", "level1_v3aug.npz")):
    p = os.path.join(WORK, fn)
    if not os.path.exists(p):
        print("skip missing", fn)
        continue
    d = np.load(p, allow_pickle=True)
    keys = [f"{tag}_{k}" for k in d["keys"]]
    parts.append((keys, d["O"], d["T"], d["y"]))

keys = sum([p[0] for p in parts], [])
O = np.hstack([p[1] for p in parts])
T = np.hstack([p[2] for p in parts])
y = parts[0][3]
print("level-1 matrix", O.shape, keys)
for i, k in enumerate(keys):
    print(f"  {k:14s} {pearsonr(O[:, i], y)[0]:.5f}")

ntr, nte = len(y), len(te)
folds = list(KFold(5, shuffle=True, random_state=42).split(np.arange(ntr)))

cands = {}
cands["mean"] = (O.mean(1), T.mean(1))

# NNLS weights, cross-validated to get an honest OOF estimate
oof = np.zeros(ntr)
for trn, val in folds:
    w, _ = nnls(np.column_stack([O[trn], np.ones(len(trn))]), y[trn])
    oof[val] = np.column_stack([O[val], np.ones(len(val))]) @ w
w_full, _ = nnls(np.column_stack([O, np.ones(ntr)]), y)
cands["nnls"] = (oof, np.column_stack([T, np.ones(nte)]) @ w_full)
print("nnls weights", dict(zip(keys + ["bias"], w_full.round(3))))

# ridge stack
oof2 = np.zeros(ntr)
pte2 = np.zeros(nte)
for trn, val in folds:
    m = Ridge(alpha=1.0).fit(O[trn], y[trn])
    oof2[val] = m.predict(O[val])
    pte2 += m.predict(T) / len(folds)
cands["ridge_stack"] = (oof2, pte2)

# top-k simple average of the strongest, least-correlated models
strong = sorted(range(len(keys)), key=lambda i: -pearsonr(O[:, i], y)[0])[:6]
cands["top6_mean"] = (O[:, strong].mean(1), T[:, strong].mean(1))
cands["nnls+ridge"] = (0.5 * (cands["nnls"][0] + oof2), 0.5 * (cands["nnls"][1] + pte2))

# ridge stack that also sees the raw dense features
fp = os.path.join(WORK, "feat_v3.npz")
if os.path.exists(fp):
    from sklearn.preprocessing import StandardScaler
    F = np.load(fp, allow_pickle=True)["F"]
    X2 = np.hstack([F[:ntr], O])
    X2te = np.hstack([F[ntr:], T])
    oof3 = np.zeros(ntr)
    pte3 = np.zeros(nte)
    for trn, val in folds:
        sc = StandardScaler().fit(X2[trn])
        m = Ridge(alpha=3.0).fit(sc.transform(X2[trn]), y[trn])
        oof3[val] = m.predict(sc.transform(X2[val]))
        pte3 += m.predict(sc.transform(X2te)) / len(folds)
    cands["ridge_stack_dense"] = (oof3, pte3)
    cands["rsd+nnls"] = (0.5 * (oof3 + cands["nnls"][0]), 0.5 * (pte3 + cands["nnls"][1]))
    cands["rsd+rs+nnls"] = ((oof3 + oof2 + cands["nnls"][0]) / 3,
                            (pte3 + pte2 + cands["nnls"][1]) / 3)

scored = sorted(((pearsonr(v[0], y)[0], k) for k, v in cands.items()), reverse=True)
for r, k in scored:
    print(f"  cand {k:14s} {r:.5f}")
best_r, best = scored[0]
pred = np.clip(cands[best][1], 0, 5)
pd.DataFrame({"id": te.id.values, "score": pred}).to_csv(
    os.path.join(ROOT, "outputs", "submission.csv"), index=False)
print("selected", best, round(best_r, 5), "-> outputs/submission.csv",
      "mean", round(float(pred.mean()), 3))
