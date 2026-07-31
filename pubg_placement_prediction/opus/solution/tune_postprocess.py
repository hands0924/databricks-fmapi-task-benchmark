import sys
sys.path.insert(0, 'solution')
import numpy as np, pandas as pd
from train import postprocess, _snap_grid

g = pd.read_pickle('solution/.cache/gtr.pkl')
oof = np.load('solution/.cache/oof.npy')
y = g.winPlacePerc.values
w = g.groupSize.values.astype(float)
mp = g.maxPlace.values
n = g.numGroupsActual.values


def mae(p, m=None):
    if m is None:
        return np.sum(w * np.abs(p - y)) / w.sum()
    return np.sum(w[m] * np.abs(p[m] - y[m])) / w[m].sum()


raw = np.clip(oof, 0, 1)
rk = postprocess(oof, g, 'rank')

print('--- fine alpha (global, grid snap) ---')
res = {}
for a in np.arange(0.55, 0.96, 0.025):
    res[round(a, 3)] = mae(_snap_grid(np.clip(a * rk + (1 - a) * raw, 0, 1), mp))
for k, v in res.items():
    print(f'  a={k} {v:.6f}')
a_g = min(res, key=res.get)
print('global best', a_g, res[a_g])

print('--- no grid snap ---')
for a in [0.6, 0.7, 0.8]:
    print(f'  a={a} {mae(np.clip(a * rk + (1 - a) * raw, 0, 1)):.6f}')

# per-matchType-kind alpha
print('--- per matchType kind ---')
kind = g.mt_kind.values
alphas = {}
for kk in np.unique(kind):
    m = kind == kk
    best = (1e9, None)
    for a in np.arange(0.3, 1.001, 0.05):
        p = _snap_grid(np.clip(a * rk + (1 - a) * raw, 0, 1), mp)
        v = mae(p, m)
        if v < best[0]:
            best = (v, round(a, 3))
    alphas[int(kk)] = best[1]
    print(f'  kind={kk} n={m.sum()} best_a={best[1]} mae={best[0]:.6f}')

# per numGroups bucket alpha
print('--- per numGroups bucket ---')
buckets = [(0, 15), (15, 25), (25, 35), (35, 55), (55, 200)]
bidx = np.zeros(len(g), int)
for i, (lo, hi) in enumerate(buckets):
    bidx[(n >= lo) & (n < hi)] = i
balpha = {}
for i, (lo, hi) in enumerate(buckets):
    m = bidx == i
    if m.sum() == 0:
        continue
    best = (1e9, None)
    for a in np.arange(0.3, 1.001, 0.05):
        p = _snap_grid(np.clip(a * rk + (1 - a) * raw, 0, 1), mp)
        v = mae(p, m)
        if v < best[0]:
            best = (v, round(a, 3))
    balpha[i] = best[1]
    print(f'  ng[{lo},{hi}) n={m.sum()} best_a={best[1]} mae={best[0]:.6f}')

# combined: alpha per bucket
A = np.full(len(g), a_g, float)
for i, a in balpha.items():
    A[bidx == i] = a
print('bucket-alpha combined:', round(mae(_snap_grid(np.clip(A * rk + (1 - A) * raw, 0, 1), mp)), 6))

A2 = np.full(len(g), a_g, float)
for kk, a in alphas.items():
    A2[kind == kk] = a
print('kind-alpha combined  :', round(mae(_snap_grid(np.clip(A2 * rk + (1 - A2) * raw, 0, 1), mp)), 6))
print(f'GLOBAL a={a_g}       : {res[a_g]:.6f}')
