"""KorSTS sentence-similarity regression — full reproducible pipeline.

Environment has no GPU and no torch/transformers, so no pretrained encoder is
used (also forbidden by the task).  The approach is a two-level stack:

  level 1  (a) ~240 dense pair features: multi-space tf-idf cosines (character,
               character-inside-word, jamo-decomposed, word, pseudo-stem),
               idf-weighted coverage, LSA/SVD cosines, BM25, soft-cosine,
               token soft-alignment coverage at several thresholds, unmatched
               idf mass, edit-opcode statistics, word order, lexical overlaps,
               PPMI+SVD word embeddings learned from the task corpus itself.
           (b) sparse "learned similarity" ridge models on n-gram interaction
               blocks  [a*b, min(a,b), |a-b|]  of L2-normalised tf-idf vectors
               (learns which shared / mismatched n-grams matter).
           (c) dense models re-fitted with the sparse OOF predictions added.
  level 2  non-negative ridge on the out-of-fold predictions.

Finally, test pairs whose (normalised) sentence pair also occurs in train get
their prediction pulled to the known train score (labels of duplicate pairs are
highly consistent: median spread 0.0).

Usage:  python solution/run.py            (run from the task root directory)
"""
import os, sys, time, pickle
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import feats as FT
import emb as EMB
import feats2 as FT2

SEED = 42
SEEDS = [42, 7]          # repeated CV: bagged test predictions + averaged OOF
NFOLD = 5
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
T0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - T0), *a, flush=True)


# ----------------------------------------------------------------- data
tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
te = pd.read_csv(os.path.join(ROOT, "test.csv"))
y = tr.score.values.astype(float)
kf = KFold(NFOLD, shuffle=True, random_state=SEED)
KFS = [KFold(NFOLD, shuffle=True, random_state=s) for s in SEEDS]
NREP = len(KFS)
log("data", tr.shape, te.shape)


def cached(name, fn):
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    obj = fn()
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    return obj


# ----------------------------------------------------------------- features
Ftr, Fte, _ = cached("F", lambda: FT.make_features(tr, te))
log("tf-idf / lexical features", Ftr.shape)
Etr, Ete = cached("E", lambda: EMB.build_emb_feats(tr, te))
log("corpus word-embedding features", Etr.shape)
Xtr, Xte = cached("X", lambda: FT2.build(tr, te))
log("bm25 / alignment / opcode features", Xtr.shape)

D = np.nan_to_num(pd.concat([Ftr, Etr, Xtr], axis=1).values.astype(np.float64))
Dt = np.nan_to_num(pd.concat([Fte, Ete, Xte], axis=1).values.astype(np.float64))
allD = np.vstack([D, Dt])
lo = np.percentile(allD, 0.5, axis=0)
hi = np.percentile(allD, 99.5, axis=0)
span = np.maximum(hi - lo, 1e-9)
D = np.clip(D, lo - 3 * span, hi + 3 * span)
Dt = np.clip(Dt, lo - 3 * span, hi + 3 * span)
log("dense matrix", D.shape)

# ----------------------------------------------------------------- level 1
OOF, PRED = {}, {}


def add(name, oof, pred):
    OOF[name] = oof
    PRED[name] = pred
    log("%-12s CV pearson %.4f" % (name, pearsonr(oof, y)[0]))


def run_dense(name, mk, Xa, Xb, scale=True):
    if scale:
        sc = StandardScaler().fit(np.vstack([Xa, Xb]))
        Xa, Xb = sc.transform(Xa), sc.transform(Xb)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xb))
    for k in KFS:
        for trn, val in k.split(Xa):
            m = mk(); m.fit(Xa[trn], y[trn])
            oof[val] += m.predict(Xa[val]) / NREP
            pred += m.predict(Xb) / (NFOLD * NREP)
    add(name, oof, pred)


def run_mlp(name, hid, alpha, Xa, Xb, seeds=(0, 1, 2)):
    sc = StandardScaler().fit(np.vstack([Xa, Xb]))
    Xa, Xb = sc.transform(Xa), sc.transform(Xb)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xb))
    for k in KFS:
        for trn, val in k.split(Xa):
            for s in seeds:
                m = MLPRegressor(hidden_layer_sizes=hid, alpha=alpha, max_iter=400,
                                 early_stopping=True, n_iter_no_change=20,
                                 validation_fraction=0.12, learning_rate_init=3e-3,
                                 random_state=s)
                m.fit(Xa[trn], y[trn])
                oof[val] += m.predict(Xa[val]) / (len(seeds) * NREP)
                pred += m.predict(Xb) / (NFOLD * len(seeds) * NREP)
    add(name, oof, pred)


# --- sparse learned-similarity models
corpus = pd.concat([tr.sentence1, tr.sentence2, te.sentence1, te.sentence2]).astype(str).tolist()
P_norm = lambda s: FT.norm(s)
P_jamo = lambda s: FT.to_jamo(FT.norm(s))
P_st2 = lambda s: FT.stem_tokens(FT.norm(s), 2)
P_st3 = lambda s: FT.stem_tokens(FT.norm(s), 3)


def pair_blocks(vec, prep, s1, s2, mode):
    A = normalize(vec.transform([prep(x) for x in s1]))
    B = normalize(vec.transform([prep(x) for x in s2]))
    if mode == "prodabs":
        return sp.hstack([A.multiply(B), abs(A - B)], format="csr")
    return sp.hstack([A.multiply(B), A.minimum(B), abs(A - B)], format="csr")


SPARSE_SPECS = [
    ("sp_jmmm1", "char_wb", (3, 5), P_jamo, 2, 1.0, "prodminmax"),
    ("sp_jmmm05", "char_wb", (3, 5), P_jamo, 2, 0.5, "prodminmax"),
    ("sp_jm35", "char_wb", (3, 5), P_jamo, 2, 0.5, "prodabs"),
    ("sp_jm36", "char_wb", (3, 6), P_jamo, 3, 0.5, "prodabs"),
    ("sp_ch25", "char_wb", (2, 5), P_norm, 2, 0.5, "prodabs"),
    ("sp_st2", "word", (1, 2), P_st2, 1, 0.5, "prodabs"),
    ("sp_st3", "word", (1, 1), P_st3, 1, 0.3, "prodabs"),
]
for name, an, ng, pr, mdf, alpha, mode in SPARSE_SPECS:
    v = TfidfVectorizer(analyzer=an, ngram_range=ng, min_df=mdf, sublinear_tf=True,
                        dtype=np.float32).fit([pr(x) for x in corpus])
    Str = pair_blocks(v, pr, tr.sentence1.astype(str), tr.sentence2.astype(str), mode)
    Ste = pair_blocks(v, pr, te.sentence1.astype(str), te.sentence2.astype(str), mode)
    oof = np.zeros(len(y)); pred = np.zeros(Ste.shape[0])
    for k in KFS:
        for trn, val in k.split(Str):
            m = Ridge(alpha=alpha, solver="sparse_cg", tol=1e-4); m.fit(Str[trn], y[trn])
            oof[val] += m.predict(Str[val]) / NREP
            pred += m.predict(Ste) / (NFOLD * NREP)
    add(name, oof, pred)

# --- dense models
run_dense("ridge", lambda: Ridge(alpha=10.0), D, Dt)
run_dense("svr3", lambda: SVR(C=3.0, epsilon=0.1), D, Dt)
run_dense("svr8", lambda: SVR(C=8.0, epsilon=0.2), D, Dt)
run_dense("krr", lambda: KernelRidge(alpha=1.0, kernel="rbf", gamma=1.0 / D.shape[1]), D, Dt)
run_dense("hgb", lambda: HistGradientBoostingRegressor(max_iter=800, learning_rate=0.035,
          max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, max_features=0.6,
          early_stopping=False, random_state=0), D, Dt, scale=False)
run_dense("hgb2", lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06,
          max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=3.0, max_features=0.4,
          early_stopping=False, random_state=1), D, Dt, scale=False)
run_dense("et", lambda: ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2,
          max_features=0.35, n_jobs=-1, random_state=0), D, Dt, scale=False)
run_mlp("mlp2", (64,), 3e-2, D, Dt)

# --- dense models augmented with sparse OOF predictions (feature-level stacking)
aug = ["sp_jmmm05", "sp_jmmm1", "sp_st2", "sp_ch25"]
D2 = np.hstack([D, np.column_stack([OOF[k] for k in aug])])
D2t = np.hstack([Dt, np.column_stack([PRED[k] for k in aug])])
run_dense("ridge_plus", lambda: Ridge(alpha=10.0), D2, D2t)
run_dense("svr_plus", lambda: SVR(C=3.0, epsilon=0.1), D2, D2t)
run_dense("hgb_plus", lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.045,
          max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, max_features=0.5,
          early_stopping=False, random_state=0), D2, D2t, scale=False)

# ----------------------------------------------------------------- level 2
names = list(OOF)
P = np.column_stack([OOF[k] for k in names])
Pt = np.column_stack([PRED[k] for k in names])
L2 = lambda: Ridge(alpha=1.0, positive=True)

oof2 = np.zeros(len(y))
for k in KFS:
    for trn, val in k.split(P):
        m = L2().fit(P[trn], y[trn]); oof2[val] += m.predict(P[val]) / NREP
cv2 = pearsonr(oof2, y)[0]
log("LEVEL-2 stack CV pearson = %.4f  (equal blend %.4f)" % (cv2, pearsonr(P.mean(1), y)[0]))

meta = L2().fit(P, y)
log("weights:", {n: round(w, 3) for n, w in zip(names, meta.coef_) if w > 1e-3})
pred = meta.predict(Pt)

# ----------------------------------------------------------------- duplicate-pair override
key = lambda a, b: (min(FT.norm(a), FT.norm(b)), max(FT.norm(a), FT.norm(b)))
ref = {}
for a, b, s in zip(tr.sentence1, tr.sentence2, y):
    ref.setdefault(key(a, b), []).append(s)
W = 0.9
n_over = 0
for i, (a, b) in enumerate(zip(te.sentence1, te.sentence2)):
    k = key(a, b)
    if k in ref:
        pred[i] = W * float(np.mean(ref[k])) + (1 - W) * pred[i]
        n_over += 1
log("duplicate-pair overrides: %d / %d" % (n_over, len(te)))

pred = np.clip(pred, 0.0, 5.0)

# ----------------------------------------------------------------- submission
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
sub = pd.DataFrame({"id": te.id.values, "score": pred})
out = os.path.join(ROOT, "outputs", "submission.csv")
sub.to_csv(out, index=False)

samp = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
assert list(sub.columns) == list(samp.columns), sub.columns
assert len(sub) == len(te) and sub.id.is_unique
assert set(sub.id) == set(te.id)
assert sub.score.notna().all() and sub.score.std() > 1e-6
log("wrote %s  rows=%d  mean=%.3f std=%.3f min=%.3f max=%.3f"
    % (out, len(sub), sub.score.mean(), sub.score.std(), sub.score.min(), sub.score.max()))
with open(os.path.join(CACHE, "level1.pkl"), "wb") as f:
    pickle.dump({"names": names, "oof": P, "pred": Pt, "cv": cv2}, f)
