"""Greedy + optimized blending of cached OOF predictions in log space."""
import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from scipy.special import softmax

CLASSES = F.CLASSES
EPS = 1e-15


def load_all(oofdir='work/oof'):
    names, O, T = [], [], []
    for p in sorted(glob.glob(f'{oofdir}/*_oof.npy')):
        n = os.path.basename(p)[:-8]
        t = f'{oofdir}/{n}_test.npy'
        if not os.path.exists(t):
            continue
        o = np.load(p).astype(np.float32); tt = np.load(t).astype(np.float32)
        if not np.isfinite(o).all() or not np.isfinite(tt).all():
            continue
        names.append(n); O.append(o); T.append(tt)
    return names, np.array(O, dtype=np.float32), np.array(T, dtype=np.float32)


def norm(p):
    p = np.clip(p, 1e-9, 1)
    return p / p.sum(1, keepdims=True)


def geo_blend(Ps, w):
    """weighted geometric mean (== weighted average in log space) + renormalize"""
    L = np.zeros(Ps[0].shape, dtype=np.float64)
    for wi, P in zip(w, Ps):
        if wi != 0:
            L += wi * np.log(np.clip(P, 1e-9, None))
    return softmax(L, axis=1)


def ari_blend(Ps, w):
    return norm(np.tensordot(w, Ps, axes=1))


def _ll_from_logits(Z, y):
    """log-loss of softmax(Z) for integer labels y, vectorised."""
    Z = Z - Z.max(1, keepdims=True)
    lse = np.log(np.exp(Z).sum(1))
    return float(np.mean(lse - Z[np.arange(len(y)), y]))


def greedy(names, O, y, iters=120, verbose=True, patience=8):
    """Caruana greedy forward selection with replacement, geometric (log-space) mean.
    Vectorised: maintains cumulative sum of log-probs."""
    LG = np.log(np.clip(O, 1e-9, None)).astype(np.float32)   # (M, N, 3)
    M = len(O)
    acc = np.zeros(LG[0].shape, dtype=np.float32)
    counts = np.zeros(M)
    hist = []
    buf = np.empty(LG[0].shape, dtype=np.float32)
    for it in range(iters):
        tot = counts.sum() + 1.0
        scores = np.empty(M)
        for i in range(M):
            np.add(acc, LG[i], out=buf)
            buf /= tot
            scores[i] = _ll_from_logits(buf, y)
        bi = int(scores.argmin()); bs = scores[bi]
        counts[bi] += 1
        acc += LG[bi]
        hist.append(bs)
        if verbose and (it < 10 or it % 10 == 0):
            print(f'  it{it:3d} +{names[bi]:26s} -> {bs:.5f}', flush=True)
        if it >= patience and hist[-1] > min(hist[:-patience]) - 1e-5:
            break
    w = counts / counts.sum()
    return w, hist[-1]


def refine(O, y, w0, blend=geo_blend, l2=0.0):
    """Continuous refinement over softmax-parameterised weights (restricted to support)."""
    sup = np.where(w0 > 0)[0]
    Os = O[sup]
    z0 = np.log(np.clip(w0[sup], 1e-6, None))

    def f(z):
        w = softmax(z)
        return log_loss(y, blend(Os, w)) + l2 * np.sum(w ** 2)
    r = minimize(f, z0, method='Powell', options={'maxiter': 20000, 'xtol': 1e-4, 'ftol': 1e-6})
    w = np.zeros(len(O)); w[sup] = softmax(r.x)
    return w, log_loss(y, blend(O, w))
