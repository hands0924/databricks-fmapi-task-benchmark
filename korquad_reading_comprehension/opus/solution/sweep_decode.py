"""Tune MBR decoding parameters on the validation split (no refit needed)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R
from qa_core import char_f1

K = int(os.environ.get("K", 16))
tr = pd.read_csv(os.path.join(R.ROOT, "train.csv"))
_, val = R.grouped_split(tr)
gold = val["answer"].astype(str).tolist()

paths = sorted(
    [
        os.path.join(R.CHUNKDIR, f)[: -len("_g.npy")]
        for f in os.listdir(R.CHUNKDIR)
        if f.startswith("pr_v2val_") and f.endswith("_g.npy")
    ],
    key=lambda p: int(p.split("_")[-1]),
)
g, texts = R._merge(paths, ["g", "t"])
s = np.load(os.path.join(R.WORK, "valscores.npy"))
assert len(s) == len(g), (len(s), len(g))

base = R.pick_best(s, g, texts, len(val))
print("argmax %.4f" % np.mean([char_f1(p, a) for p, a in zip(base, gold)]), flush=True)

sl = R.group_slices(g, len(val))
cache = []  # (cands, scores, pairwise F1, gold F1)
for gi in range(len(val)):
    if sl[gi] is None:
        cache.append(None)
        continue
    a, b = sl[gi]
    sv_all = s[a:b]
    k = min(K, b - a)
    idx = np.argpartition(-sv_all, k - 1)[:k] if b - a > k else np.arange(b - a)
    sv = sv_all[idx]
    cands = [texts[a + int(i)] for i in idx]
    M = np.ones((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(i + 1, k):
            M[i, j] = M[j, i] = char_f1(cands[i], cands[j])
    gf = np.array([char_f1(c, gold[gi]) for c in cands], dtype=np.float32)
    cache.append((sv, M, gf))
print("cached", flush=True)

best = (None, -1)
for temp in (0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0):
    for agg in (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        tot = 0.0
        for c in cache:
            if c is None:
                continue
            sv, M, gf = c
            w = np.exp((sv - sv.max()) / temp)
            w /= w.sum()
            exp_f1 = M @ w
            v = sv * (1 - agg) + agg * exp_f1
            tot += gf[int(np.argmax(v))]
        f = tot / len(val)
        if f > best[1]:
            best = ((K, temp, agg), f)
        print("K=%d temp=%.2f agg=%.2f -> %.4f" % (K, temp, agg, f), flush=True)
print("BEST", best)
