"""KLUE-STS solution: engineered similarity features + model ensemble (CPU only).

Usage:
    python solution/run.py plain   # level-1 models on plain features
    python solution/run.py aug     # level-1 models with swap augmentation/TTA
    python solution/blend.py       # final blend -> outputs/submission.csv

Each run also writes a standalone (already valid) outputs/submission.csv, so a
single invocation is enough for a complete submission; blend.py combines both
model sets for the best CV score.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from features import FeatureBuilder, clean, stem_text  # noqa: E402

WORK = os.path.join(ROOT, "work")
os.makedirs(WORK, exist_ok=True)
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
T0 = time.time()
SEED = 42


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def write_sub(pred, te, tag=""):
    pred = np.clip(pred, 0, 5)
    out = pd.DataFrame({"id": te.id.values, "score": pred})
    out.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
    log("wrote submission", tag, out.shape, "mean", round(float(pred.mean()), 3))


def main(aug=True):
    log("mode:", "aug" if aug else "plain")
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    ntr, nte = len(tr), len(te)
    y = tr.score.values.astype(np.float64)

    # ---------------- dense engineered features -------------------------
    def build(tag, dfs):
        cache = os.path.join(WORK, f"feat_{tag}.npz")
        if os.path.exists(cache):
            d = np.load(cache, allow_pickle=True)
            return d["F"], d["Fp"], list(d["names"])
        fb = FeatureBuilder()
        F, Fp = fb.fit_transform(dfs)
        np.savez_compressed(cache, F=F, Fp=Fp, names=np.array(fb.feature_names))
        return F, Fp, fb.feature_names

    F, Fp, names = build("v3", [tr, te])
    if aug:
        def swap(d):
            return d.rename(columns={"sentence1": "sentence2", "sentence2": "sentence1"})
        Fw, Fpw, _ = build("v3sw", [swap(tr), swap(te)])
    else:
        Fw, Fpw = F, Fp
    log("dense features", F.shape, "pair-emb", Fp.shape)

    # ---------------- sparse pair representations -----------------------
    s1 = pd.concat([tr.sentence1, te.sentence1]).astype(str).tolist()
    s2 = pd.concat([tr.sentence2, te.sentence2]).astype(str).tolist()
    n = len(s1)

    def pair_sparse(view_fn, **kw):
        texts = [view_fn(t) for t in s1] + [view_fn(t) for t in s2]
        v = TfidfVectorizer(**kw)
        X = normalize(v.fit_transform(texts))
        A, B = X[:n].tocsr(), X[n:].tocsr()
        return sparse.hstack([A.minimum(B), abs(A - B)]).tocsr()

    Sp = sparse.hstack([
        pair_sparse(clean, analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=3),
        pair_sparse(clean, analyzer="word", ngram_range=(1, 1), sublinear_tf=True,
                    min_df=1, token_pattern=r"\S+"),
        pair_sparse(stem_text, analyzer="word", ngram_range=(1, 1), sublinear_tf=True,
                    min_df=1, token_pattern=r"\S+"),
    ]).tocsr()
    log("sparse pair rep", Sp.shape)

    Xs, Xs_te = F[:ntr], F[ntr:]
    Xw, Xw_te = Fw[:ntr], Fw[ntr:]
    Fa = np.hstack([F, Fp])
    Faw = np.hstack([Fw, Fpw])
    Xa, Xa_te = Fa[:ntr], Fa[ntr:]
    Xaw, Xaw_te = Faw[:ntr], Faw[ntr:]
    if not aug:  # disable augmentation / TTA
        Xw = Xw_te = Xaw = Xaw_te = None
    Sp_tr, Sp_te = Sp[:ntr], Sp[ntr:]

    kf = KFold(5, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.arange(ntr)))

    models = {}

    def run_model(name, make, X, Xte, Xsw=None, Xsw_te=None):
        """Fit per fold. If swapped views are given, train on both sentence
        orders (augmentation) and average predictions over both (TTA)."""
        oof = np.zeros(ntr)
        pte = np.zeros(nte)
        for trn, val in folds:
            m = make()
            if Xsw is None:
                m.fit(X[trn], y[trn])
                oof[val] = m.predict(X[val])
                pte += m.predict(Xte) / len(folds)
            else:
                m.fit(np.vstack([X[trn], Xsw[trn]]), np.concatenate([y[trn], y[trn]]))
                oof[val] = 0.5 * (m.predict(X[val]) + m.predict(Xsw[val]))
                pte += 0.5 * (m.predict(Xte) + m.predict(Xsw_te)) / len(folds)
        r = pearsonr(oof, y)[0]
        log(f"{name:20s} pearson={r:.5f}")
        models[name] = (oof, pte, r)
        return r

    # scalar-feature GBM (fast, strongest single model)
    run_model("hgb", lambda: HistGradientBoostingRegressor(
        max_iter=700, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, max_features=0.8, early_stopping=False,
        random_state=0), Xs, Xs_te, Xw, Xw_te)
    write_sub(models["hgb"][1], te, "hgb")

    run_model("hgb2", lambda: HistGradientBoostingRegressor(
        max_iter=1200, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=10,
        l2_regularization=0.5, max_features=0.6, early_stopping=False,
        random_state=7), Xs, Xs_te, Xw, Xw_te)

    run_model("svr", lambda: make_pipeline(
        StandardScaler(), SVR(C=8.0, epsilon=0.1, gamma="scale")), Xs, Xs_te, Xw, Xw_te)

    run_model("et", lambda: ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=2, max_features=0.5, n_jobs=4,
        random_state=0), Xs, Xs_te, Xw, Xw_te)

    run_model("ridge_d", lambda: make_pipeline(
        StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))), Xa, Xa_te, Xaw, Xaw_te)

    run_model("ridge_sp", lambda: Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000),
              Sp_tr, Sp_te)

    log("models fitted:", list(models))

    # ---------------- stacking -----------------------------------------
    keys = [k for k in models]
    O = np.column_stack([models[k][0] for k in keys])
    T = np.column_stack([models[k][1] for k in keys])
    log("mean blend", round(pearsonr(O.mean(1), y)[0], 5))

    # second-level: HGB over dense features + level-1 OOF preds (nested-free
    # approximation: level-1 oof used as features, refit per fold)
    st_oof = np.zeros(ntr)
    st_te = np.zeros(nte)
    X2 = np.hstack([Xs, O])
    X2te = np.hstack([Xs_te, T])
    for trn, val in folds:
        m = Ridge(alpha=1.0)
        sc = StandardScaler().fit(X2[trn])
        m.fit(sc.transform(X2[trn]), y[trn])
        st_oof[val] = m.predict(sc.transform(X2[val]))
        st_te += m.predict(sc.transform(X2te)) / len(folds)
    log("stack ridge(dense+oof)", round(pearsonr(st_oof, y)[0], 5))

    # non-negative weights over level-1 preds via ridge on OOF
    from scipy.optimize import nnls
    Oc = np.column_stack([O, np.ones(ntr)])
    w, _ = nnls(Oc, y)
    bl_oof = Oc @ w
    bl_te = np.column_stack([T, np.ones(nte)]) @ w
    log("nnls blend", round(pearsonr(bl_oof, y)[0], 5), dict(zip(keys + ["b"], w.round(3))))

    cands = {
        "mean": (O.mean(1), T.mean(1)),
        "nnls": (bl_oof, bl_te),
        "stack": (st_oof, st_te),
        "nnls+stack": (0.5 * bl_oof + 0.5 * st_oof, 0.5 * bl_te + 0.5 * st_te),
    }
    for k in keys:
        cands[k] = (models[k][0], models[k][1])
    scored = sorted(((pearsonr(v[0], y)[0], k) for k, v in cands.items()), reverse=True)
    for r, k in scored:
        log(f"  cand {k:14s} {r:.5f}")
    best_r, best = scored[0]
    log("selected", best, round(best_r, 5))
    oof_b, te_b = cands[best]

    # isotonic recalibration check (Pearson is not monotone-invariant)
    iso_oof = np.zeros(ntr)
    iso_te = np.zeros(nte)
    for trn, val in folds:
        ir = IsotonicRegression(out_of_bounds="clip").fit(oof_b[trn], y[trn])
        iso_oof[val] = ir.predict(oof_b[val])
    ir_full = IsotonicRegression(out_of_bounds="clip").fit(oof_b, y)
    iso_te = ir_full.predict(te_b)
    r_iso = pearsonr(iso_oof, y)[0]
    log("isotonic", round(r_iso, 5), "vs raw", round(best_r, 5))

    final_te = iso_te if r_iso > best_r + 0.001 else te_b
    write_sub(final_te, te, "final:" + best + ("+iso" if final_te is iso_te else ""))
    np.savez_compressed(os.path.join(WORK, "level1_v3aug.npz" if aug else "level1_v2.npz"),
                        O=O, T=T, y=y, keys=np.array(keys))
    log("done")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "aug"
    main(aug=(mode == "aug"))
