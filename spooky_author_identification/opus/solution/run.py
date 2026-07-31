#!/usr/bin/env python
"""End-to-end reproducible pipeline for t2_spooky (multi-class log loss).

    python solution/run.py                 # full run: features -> zoo -> ensemble -> submission
    python solution/run.py --ensemble-only # reuse cached OOF predictions in work/oof

Everything is cached under work/ (features in work/cache, per-model OOF+test
predictions in work/oof), so the run is resumable and cheap to re-tune.
Must be executed from the task root (the directory containing train.csv).

Final OOF multi-class log loss: 0.26424 (see work/final_weights.txt after a run).
"""
import argparse, gc, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np, pandas as pd
import features as F
import zoo as Z
import blend as B
import stack as S
import models as M
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# level-2 configuration (values chosen by the OOF sweeps logged in work/)
STACK_CS = [0.05, 0.15, 0.5, 1.5]        # bagged LR stack on log-prob meta features
STACK_HAND_CS = [0.05, 0.3]              # same, plus the 26 dense stylometric features
BAG_SEEDS = [202, 707, 1313]
BAG_SEEDS_HAND = [202, 707]
GREEDY_ITERS = 150
PRUNE_MAX_LL = 1.2                       # drop level-1 models worse than this
PRUNE_MAX_CORR = 0.9995                  # drop near-duplicate level-1 models


def build_features():
    for fn in [F.word_tfidf, F.word_tfidf1, F.char_tfidf, F.char_full_tfidf,
               F.word_counts, F.char_counts, F.pos_tfidf, F.hand_feats,
               F.stopw_tfidf, F.charcase_tfidf, F.word3_tfidf, F.svd_feats]:
        fn()


def run_zoo():
    specs = M.specs()
    print(f'--- level 1: {len(specs)} base models ---', flush=True)
    for name, fname, fac in specs:
        try:
            Z.run(name, fac, fname)
        except Exception as e:                     # never let one model kill the run
            print(f'!! {name} failed: {e}', flush=True)


def prune(names, O, T, y):
    sc = np.array([log_loss(y, o) for o in O])
    keep = np.where(sc < PRUNE_MAX_LL)[0]
    sel = []
    for i in keep[np.argsort(sc[keep])]:           # best first, so duplicates drop the worse one
        if all(np.corrcoef(O[i].ravel(), O[j].ravel())[0, 1] < PRUNE_MAX_CORR for j in sel):
            sel.append(i)
    sel = np.array(sel)
    return [names[i] for i in sel], np.ascontiguousarray(O[sel]), np.ascontiguousarray(T[sel])


def write_sub(p, ids, path='outputs/submission.csv'):
    p = np.clip(p, 2e-6, 1)
    p = p / p.sum(1, keepdims=True)
    os.makedirs('outputs', exist_ok=True)
    sub = pd.DataFrame(p, columns=F.CLASSES)
    sub.insert(0, 'id', ids)
    sub.to_csv(path, index=False)
    return sub


def ensemble():
    t0 = time.time()
    tr, te, y = F.load()
    names, O, T = B.load_all()
    print(f'--- level 2: {len(names)} cached models ---', flush=True)
    names, O, T = prune(names, O, T, y)
    gc.collect()
    print('kept after pruning:', len(names), flush=True)

    # ---- 2a: Caruana greedy forward selection over a weighted geometric mean ----
    w, gs = B.greedy(names, O, y, iters=GREEDY_ITERS, verbose=False)
    w, gs = B.refine(O, y, w)
    go, gt = B.geo_blend(O, w), B.geo_blend(T, w)
    print(f'greedy geometric blend oof={gs:.5f} ({time.time()-t0:.0f}s)', flush=True)
    write_sub(gt, te.id.values)                    # keep a valid submission on disk at all times

    # ---- 2b: bagged logistic-regression stacks on centred log-probabilities ----
    A, Bm = S.meta_matrix(O, T)
    Ah, Bh = S.meta_matrix(O, T, use_hand=True)
    del O, T; gc.collect()
    A, Bm = A.astype(np.float32), Bm.astype(np.float32)
    Ah, Bh = Ah.astype(np.float32), Bh.astype(np.float32)
    print('meta matrix', A.shape, flush=True)

    cands = {}

    def bag(tag, X, Xt, C, seeds):
        aO = np.zeros((len(y), 3)); aT = np.zeros((Xt.shape[0], 3))
        for sd in seeds:
            o, t, _ = S.cv_stack(X, y, lambda C=C: LogisticRegression(C=C, max_iter=800),
                                 seed=sd, Bm=Xt)
            aO += o / len(seeds); aT += t / len(seeds)
        s = log_loss(y, aO)
        cands[tag] = (s, aO, aT)
        print(f'  {tag} oof={s:.5f} ({time.time()-t0:.0f}s)', flush=True)
        gc.collect()

    for C in STACK_CS:
        bag(f'lr{C}', A, Bm, C, BAG_SEEDS)
    for C in STACK_HAND_CS:
        bag(f'lrh{C}', Ah, Bh, C, BAG_SEEDS_HAND)

    # ---- level 3: greedy geometric mean over the level-2 candidates + the L1 blend ----
    keys = list(cands.keys())
    CO = np.array([cands[k][1] for k in keys] + [go])
    CT = np.array([cands[k][2] for k in keys] + [gt])
    cn = keys + ['l1greedy']
    w2, s2 = B.greedy(cn, CO, y, iters=80, verbose=False)
    w2, s2 = B.refine(CO, y, w2)
    bs = min((v[0], k) for k, v in cands.items())
    print(f'level-3 greedy oof={s2:.5f} | best single level-2 {bs[1]} oof={bs[0]:.5f}', flush=True)

    if s2 < bs[0] - 1e-4:
        p, tag, wsel = B.geo_blend(CT, w2), f'L3greedy({s2:.5f})', list(zip(cn, w2))
    else:
        p, tag, wsel = cands[bs[1]][2], f'{bs[1]}({bs[0]:.5f})', [(bs[1], 1.0)]

    sub = write_sub(p, te.id.values)
    with open('work/final_weights.txt', 'w') as f:
        f.write(f'# chosen: {tag}\n# L1 greedy oof={gs:.5f}\n')
        for k, v in sorted(cands.items(), key=lambda x: x[1][0]):
            f.write(f'# cand {k} oof={v[0]:.5f}\n')
        f.write('# level-3 weights:\n')
        for n, ww in sorted(wsel, key=lambda x: -x[1]):
            if ww > 1e-4:
                f.write(f'{ww:.4f}\t{n}\n')
        f.write('# level-1 greedy weights:\n')
        for n, ww in sorted(zip(names, w), key=lambda x: -x[1]):
            if ww > 1e-4:
                f.write(f'{ww:.4f}\t{n}\n')
    print(f'wrote outputs/submission.csv {sub.shape}  [{tag}]  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble-only', action='store_true')
    a = ap.parse_args()
    os.makedirs('work', exist_ok=True)
    if not a.ensemble_only:
        build_features()
        run_zoo()
    ensemble()
