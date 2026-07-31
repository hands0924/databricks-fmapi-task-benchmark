"""End-to-end runner: features -> GBM ranker -> submission.

Usage:
  python run.py feats_val     # features for grouped train/val split
  python run.py fit_val       # train on split, report val char-F1
  python run.py feats_full    # features on all train + test
  python run.py final         # fit on all train, predict test, write submission
"""
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import AnswerTypeStats, Idf, NFEAT, gen_features, get_ctx, heuristic_scores, question_info  # noqa
from qa_core import char_f1  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

TRAIN_NEG_TOP = 55
TRAIN_NEG_RAND = 25
SEED = 7

_G = {}


def _init_globals():
    if "idf" in _G:
        return
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    _G["tr"], _G["te"] = tr, te
    _G["idf"] = Idf(tr["context"].tolist() + te["context"].tolist())
    # answer-type prior is built ONLY from the non-validation article slice so
    # that validation numbers stay honest; the same prior is used everywhere.
    trn, _val = grouped_split(tr)
    _G["ats"] = AnswerTypeStats(trn["question"].tolist(), trn["answer"].astype(str).tolist(), _G["idf"])


CHUNKDIR = os.path.join(WORK, "chunks")
os.makedirs(CHUNKDIR, exist_ok=True)


def _work_train(args):
    i0, rows, tag = args
    outp = os.path.join(CHUNKDIR, f"tr_{tag}_{i0}")
    if os.path.exists(outp + "_X.npy"):
        return outp
    rng = np.random.RandomState(SEED + i0)
    idf = _G["idf"]
    Xs, ys, gs = [], [], []
    for gi, (ctx_s, q, ans) in rows:
        ctx = get_ctx(ctx_s)
        qi = question_info(q, idf)
        out = gen_features(ctx, qi, idf, gold=ans, max_cand=10 ** 6, ats=_G["ats"])
        if not out or len(out) != 3:
            continue
        X, metas, y = out
        n = len(y)
        h = heuristic_scores(X)
        order = np.argsort(h)[::-1]
        keep = set(int(x) for x in order[:TRAIN_NEG_TOP])
        keep |= set(int(x) for x in np.argsort(y)[::-1][:4])
        if n > len(keep):
            rest = [i for i in range(n) if i not in keep]
            keep |= set(int(x) for x in rng.choice(rest, size=min(TRAIN_NEG_RAND, len(rest)), replace=False))
        idxs = np.array(sorted(keep))
        Xs.append(X[idxs])
        ys.append(y[idxs])
        gs.append(np.full(len(idxs), gi, dtype=np.int32))
    if not Xs:
        return None
    np.save(outp + "_X.npy", np.vstack(Xs))
    np.save(outp + "_y.npy", np.concatenate(ys))
    np.save(outp + "_g.npy", np.concatenate(gs))
    return outp


def _work_pred(args):
    i0, rows, tag = args
    outp = os.path.join(CHUNKDIR, f"pr_{tag}_{i0}")
    if os.path.exists(outp + "_X.npy"):
        return outp
    idf = _G["idf"]
    Xs, gs, texts = [], [], []
    for gi, (ctx_s, q) in rows:
        ctx = get_ctx(ctx_s)
        qi = question_info(q, idf)
        out = gen_features(ctx, qi, idf, gold=None, max_cand=280, ats=_G["ats"])
        if not out or len(out) != 2 or len(out[1]) == 0:
            Xs.append(np.zeros((1, NFEAT), dtype=np.float32))
            gs.append(np.array([gi], dtype=np.int32))
            texts.append("")
            continue
        X, metas = out
        Xs.append(X)
        gs.append(np.full(len(metas), gi, dtype=np.int32))
        texts.extend(m[0] for m in metas)
    np.save(outp + "_X.npy", np.vstack(Xs))
    np.save(outp + "_g.npy", np.concatenate(gs))
    np.save(outp + "_t.npy", np.array(texts, dtype=object), allow_pickle=True)
    return outp


def chunkify(items, tag, size=200):
    out = []
    for i in range(0, len(items), size):
        out.append((i, items[i : i + size], tag))
    return out


def parallel(fn, tasks, nproc=4):
    _init_globals()
    if nproc <= 1:
        return [fn(t) for t in tasks]
    ctxmp = mp.get_context("fork")
    with ctxmp.Pool(nproc) as pool:
        res = []
        for i, r in enumerate(pool.imap(fn, tasks)):
            res.append(r)
            if i % 5 == 0:
                print(f"  chunk {i+1}/{len(tasks)} t={time.time()-T0:.0f}s", flush=True)
        return res


T0 = time.time()
VER = os.environ.get("VER", "v2")


def _merge(paths, keys):
    shapes = [np.load(p + f"_{keys[0]}.npy", mmap_mode="r").shape for p in paths]
    total = sum(s[0] for s in shapes)
    outs = {}
    for k in keys:
        first = np.load(paths[0] + f"_{k}.npy", allow_pickle=(k == "t"), mmap_mode=None if k == "t" else "r")
        shp = (total,) + first.shape[1:]
        outs[k] = np.empty(shp, dtype=first.dtype)
    pos = 0
    for p, s in zip(paths, shapes):
        n = s[0]
        for k in keys:
            arr = np.load(p + f"_{k}.npy", allow_pickle=(k == "t"))
            outs[k][pos : pos + n] = arr
        pos += n
    return [outs[k] for k in keys]


def train_features(df, tag):
    rows = [(i, (str(r.context), str(r.question), str(r.answer))) for i, r in enumerate(df.itertuples())]
    paths = [p for p in parallel(_work_train, chunkify(rows, tag)) if p is not None]
    X, y, g = _merge(paths, ["X", "y", "g"])
    print("train feats", X.shape, flush=True)
    return X, y, g


def pred_features(df, tag):
    rows = [(i, (str(r.context), str(r.question))) for i, r in enumerate(df.itertuples())]
    paths = parallel(_work_pred, chunkify(rows, tag))
    X, g, texts = _merge(paths, ["X", "g", "t"])
    print("pred feats", X.shape, flush=True)
    return X, g, texts


def group_slices(g, n_groups):
    """assumes g is non-decreasing by construction"""
    sl = [None] * n_groups
    start = 0
    for i in range(1, len(g) + 1):
        if i == len(g) or g[i] != g[start]:
            sl[g[start]] = (start, i)
            start = i
    return sl


def pick_mbr(scores, g, texts, n_groups, K=12, temp=0.15, agg=0.0):
    """MBR decoding: pick candidate maximizing expected char-F1 under the
    model's softmax distribution over the top-K candidates."""
    sl = group_slices(g, n_groups)
    out = [""] * n_groups
    for gi in range(n_groups):
        if sl[gi] is None:
            continue
        a, b = sl[gi]
        s = scores[a:b]
        k = min(K, b - a)
        idx = np.argpartition(-s, k - 1)[:k] if b - a > k else np.arange(b - a)
        sv = s[idx]
        w = np.exp((sv - sv.max()) / temp)
        w /= w.sum()
        cand = [texts[a + int(i)] for i in idx]
        if agg > 0:
            best, bi = -1e18, 0
            for i in range(k):
                exp_f1 = 0.0
                for j in range(k):
                    exp_f1 += w[j] * (1.0 if i == j else char_f1(cand[i], cand[j]))
                v = sv[i] * (1 - agg) + agg * exp_f1
                if v > best:
                    best, bi = v, i
            out[gi] = cand[bi]
        else:
            out[gi] = cand[int(np.argmax(sv))]
    return out


def pick_best(scores, g, texts, n_groups):
    best = np.full(n_groups, -1e18)
    out = [""] * n_groups
    for i in range(len(scores)):
        gi = g[i]
        if scores[i] > best[gi]:
            best[gi] = scores[i]
            out[gi] = texts[i]
    return out


def grouped_split(tr, frac=0.2, seed=3):
    ctxs = tr["context"].drop_duplicates().tolist()
    rng = np.random.RandomState(seed)
    # group by article prefix in id to mimic article-grouped split
    arts = tr["id"].astype(str).str.split("-").str[0]
    uniq = arts.unique()
    rng.shuffle(uniq)
    nval = int(len(uniq) * frac)
    val_arts = set(uniq[:nval])
    mask = arts.isin(val_arts).values
    return tr[~mask].reset_index(drop=True), tr[mask].reset_index(drop=True)


def fit_model(X, y, **kw):
    from sklearn.ensemble import HistGradientBoostingRegressor

    params = dict(
        max_iter=int(os.environ.get("ITER", 300)),
        learning_rate=float(os.environ.get("LR", 0.1)),
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=False,
        random_state=0,
    )
    params.update(kw)
    m = HistGradientBoostingRegressor(**params)
    m.fit(X, y)
    return m


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "final"
    _init_globals()
    tr, te = _G["tr"], _G["te"]

    if stage == "feats_val":
        _, val = grouped_split(tr)
        print("val size", len(val), flush=True)
        pred_features(val, VER + "val")

    elif stage == "fit_val":
        trn, val = grouped_split(tr)
        X, y, g = train_features(tr, VER + "full")
        arts = tr["id"].astype(str).str.split("-").str[0].values
        val_arts = set(val["id"].astype(str).str.split("-").str[0].unique())
        is_val_row = np.array([a in val_arts for a in arts])
        keep = ~is_val_row[g]
        X, y = X[keep], y[keep]
        Xv, gv, tv = pred_features(val, VER + "val")
        print("fitting", X.shape, flush=True)
        m = fit_model(X, y)
        s = m.predict(Xv)
        np.save(os.path.join(WORK, "valscores.npy"), s)
        gold = val["answer"].astype(str).tolist()
        preds = pick_best(s, gv, tv, len(val))
        f1 = np.mean([char_f1(p, a) for p, a in zip(preds, gold)])
        print("VAL char-F1 (model): %.4f" % f1, flush=True)
        for temp in (0.05, 0.1, 0.2):
            for agg in (0.3, 0.6, 1.0):
                pm = pick_mbr(s, gv, tv, len(val), K=12, temp=temp, agg=agg)
                print("  MBR temp=%.2f agg=%.1f -> %.4f" % (temp, agg, np.mean([char_f1(p, a) for p, a in zip(pm, gold)])), flush=True)
        hs = heuristic_scores(Xv)
        hp = pick_best(hs, gv, tv, len(val))
        print("VAL char-F1 (heuristic): %.4f" % np.mean([char_f1(p, a) for p, a in zip(hp, val["answer"].astype(str))]), flush=True)
        import joblib

        joblib.dump(m, os.path.join(WORK, "model_val.pkl"))
        np.save(os.path.join(WORK, "valpred.npy"), np.array(preds, dtype=object), allow_pickle=True)
        for i in range(20):
            print("Q:", val["question"][i], "| G:", val["answer"][i], "| P:", preds[i])

    elif stage == "feats_test":
        pred_features(te, VER + "test")

    elif stage == "feats_fulltrain":
        train_features(tr, VER + "full")

    elif stage == "heur_sub":
        Xt, gt, tt = pred_features(te, VER + "test")
        preds = pick_best(heuristic_scores(Xt), gt, tt, len(te))
        pd.DataFrame({"id": te["id"], "answer": preds}).to_csv(os.path.join(OUT, "submission.csv"), index=False)
        print("wrote heuristic submission", flush=True)

    elif stage == "final":
        import joblib

        mpath = os.path.join(WORK, "model_full.pkl")
        if os.path.exists(mpath):
            m = joblib.load(mpath)
        else:
            X, y, g = train_features(tr, VER + "full")
            print("fitting final", X.shape, flush=True)
            m = fit_model(X, y)
            joblib.dump(m, mpath)
            del X, y, g
        Xt, gt, tt = pred_features(te, VER + "test")
        st = np.empty(len(Xt), dtype=np.float64)
        step = 400000
        for i in range(0, len(Xt), step):
            st[i : i + step] = m.predict(Xt[i : i + step])
        agg = float(os.environ.get("AGG", 0.0))
        if agg > 0:
            preds = pick_mbr(st, gt, tt, len(te), K=int(os.environ.get("MBRK", 12)),
                             temp=float(os.environ.get("TEMP", 0.1)), agg=agg)
        else:
            preds = pick_best(st, gt, tt, len(te))
        preds = [p if str(p).strip() else "1" for p in preds]
        sub = pd.DataFrame({"id": te["id"], "answer": preds})
        sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
        print("wrote", len(sub), "rows", flush=True)
    print("done %.0fs" % (time.time() - T0))


if __name__ == "__main__":
    main()
