"""Final v3 STS solution.

Pipeline (no external data, no internet, train.csv only):
  1. Normalize Korean sentences; also produce jamo-decomposed form.
  2. Build multiple TF-IDF representations (surface char/word + jamo char).
     For each, derive pairwise similarity features (cosine, dot, jaccard, norms).
  3. Build SVD dense embeddings of TF-IDF word/char matrices and derive
     cosine/euclidean/manhattan/elementwise-mean pairwise features.
  4. Add hand-crafted lexical features (token/bigram/trigram Jaccard, length
     ratio, common prefix, equality, etc.).
  5. Standardize the combined feature matrix.
  6. Train diverse base learners with 5-fold CV producing OOF + test preds:
       - Ridge (alphas: 1, 10, 50, 200)
       - BayesianRidge
       - SVR (rbf, C in {1, 3, 10})
  7. Train a Ridge meta-learner on the OOF predictions (5-fold nested OOF)
     to produce the final test prediction.
  8. Blend meta prediction with the best single-model test prediction and clip
     to [0, 5].

Reproducibility: fixed random seed; pure CPU sklearn/numpy/pandas/scipy.
"""
import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import vstack

BASE = "/tmp/kmle/M3_t11_korsts_full_20260731_010802/task"
RNG = 42
np.random.seed(RNG)


# ---------- Hangul jamo decomposition ----------
CHO_LIST = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG_LIST = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG_LIST = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

def syllable_to_jamo(ch):
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        idx = code - 0xAC00
        cho = idx // (21 * 28)
        jung = (idx % (21 * 28)) // 28
        jong = idx % 28
        return CHO_LIST[cho] + JUNG_LIST[jung] + (JONG_LIST[jong] if jong else "")
    return ch

def jamo_string(s):
    return "".join(syllable_to_jamo(c) for c in str(s))


def normalize(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# ---------- Load ----------
train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")
s1_tr_raw = train["sentence1"].astype(str)
s2_tr_raw = train["sentence2"].astype(str)
s1_te_raw = test["sentence1"].astype(str)
s2_te_raw = test["sentence2"].astype(str)
s1_tr = s1_tr_raw.map(normalize)
s2_tr = s2_tr_raw.map(normalize)
s1_te = s1_te_raw.map(normalize)
s2_te = s2_te_raw.map(normalize)
j1_tr = s1_tr_raw.map(jamo_string)
j2_tr = s2_tr_raw.map(jamo_string)
j1_te = s1_te_raw.map(jamo_string)
j2_te = s2_te_raw.map(jamo_string)
y = train["score"].values.astype(float)


# ---------- Pair feature helpers ----------
def build_pair_mats(vec, s1, s2, s1t, s2t):
    vec.fit(pd.concat([s1, s2, s1t, s2t]))
    return vec.transform(s1), vec.transform(s2), vec.transform(s1t), vec.transform(s2t)

def cosine_row(a, b):
    out = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        out[i] = cosine_similarity(a[i], b[i])[0, 0]
    return out

def dot_row(a, b):
    return np.array(a.multiply(b).sum(axis=1)).ravel()

def norm_l2(a):
    return np.sqrt(np.array(a.multiply(a).sum(axis=1)).ravel())

def jaccard_count(a, b):
    ab = a.astype(bool).astype(float).multiply(b.astype(bool).astype(float))
    inter = np.array(ab.sum(axis=1)).ravel()
    union = np.array(a.astype(bool).astype(float).sum(axis=1)).ravel() + \
            np.array(b.astype(bool).astype(float).sum(axis=1)).ravel() - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    return out

def feat_from_mats(a1, a2, b1, b2):
    sim_tr = cosine_row(a1, a2); sim_te = cosine_row(b1, b2)
    dot_tr = dot_row(a1, a2); dot_te = dot_row(b1, b2)
    n1_tr = norm_l2(a1); n2_tr = norm_l2(a2)
    n1_te = norm_l2(b1); n2_te = norm_l2(b2)
    jac_tr = jaccard_count(a1, a2); jac_te = jaccard_count(b1, b2)
    f_tr = np.column_stack([sim_tr, dot_tr, jac_tr, n1_tr, n2_tr,
                            n1_tr * n2_tr, n1_tr + n2_tr, np.abs(n1_tr - n2_tr)])
    f_te = np.column_stack([sim_te, dot_te, jac_te, n1_te, n2_te,
                            n1_te * n2_te, n1_te + n2_te, np.abs(n1_te - n2_te)])
    return f_tr, f_te


# ---------- TF-IDF pair features ----------
specs = [
    ("surf_char24", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True), s1_tr, s2_tr, s1_te, s2_te),
    ("surf_char36", dict(analyzer="char_wb", ngram_range=(3, 6), min_df=3, sublinear_tf=True), s1_tr, s2_tr, s1_te, s2_te),
    ("surf_char235", dict(analyzer="char", ngram_range=(2, 5), min_df=3, sublinear_tf=True), s1_tr, s2_tr, s1_te, s2_te),
    ("surf_word12", dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"), s1_tr, s2_tr, s1_te, s2_te),
    ("surf_word1", dict(analyzer="word", ngram_range=(1, 1), min_df=1, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"), s1_tr, s2_tr, s1_te, s2_te),
    ("jamo_char24", dict(analyzer="char", ngram_range=(2, 4), min_df=2, sublinear_tf=True), j1_tr, j2_tr, j1_te, j2_te),
    ("jamo_char35", dict(analyzer="char", ngram_range=(3, 5), min_df=3, sublinear_tf=True), j1_tr, j2_tr, j1_te, j2_te),
]
all_feats_tr, all_feats_te = [], []
for name, sp, a1s, a2s, b1s, b2s in specs:
    vec = TfidfVectorizer(**sp)
    a1, a2, b1, b2 = build_pair_mats(vec, a1s, a2s, b1s, b2s)
    f_tr, f_te = feat_from_mats(a1, a2, b1, b2)
    all_feats_tr.append(f_tr); all_feats_te.append(f_te)
    print(f"{name}: {f_tr.shape[1]} feats")

X_vec_tr = np.hstack(all_feats_tr)
X_vec_te = np.hstack(all_feats_te)
print("vectorizer features:", X_vec_tr.shape)


# ---------- SVD dense embeddings ----------
def svd_embed_features(vec_params, s1, s2, s1t, s2t, n_comp=100, seed=RNG):
    vec = TfidfVectorizer(**vec_params)
    vec.fit(pd.concat([s1, s2, s1t, s2t]))
    Xa = vec.transform(s1); Xb = vec.transform(s2)
    Ya = vec.transform(s1t); Yb = vec.transform(s2t)
    all_rows = vstack([Xa, Xb, Ya, Yb])
    n_comp = min(n_comp, min(all_rows.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    svd.fit(all_rows)
    ea = svd.transform(Xa); eb = svd.transform(Xb)
    fa = svd.transform(Ya); fb = svd.transform(Yb)
    def cos(A, B):
        na = np.linalg.norm(A, axis=1) + 1e-12
        nb = np.linalg.norm(B, axis=1) + 1e-12
        return np.sum(A * B, axis=1) / (na * nb)
    sim_tr = cos(ea, eb); sim_te = cos(fa, fb)
    euc_tr = np.linalg.norm(ea - eb, axis=1); euc_te = np.linalg.norm(fa - fb, axis=1)
    man_tr = np.sum(np.abs(ea - eb), axis=1); man_te = np.sum(np.abs(fa - fb), axis=1)
    mean_tr = (ea + eb) / 2; mean_te = (fa + fb) / 2
    out_tr = np.column_stack([sim_tr, euc_tr, man_tr, mean_tr])
    out_te = np.column_stack([sim_te, euc_te, man_te, mean_te])
    return out_tr, out_te

svd_specs = [
    (dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"), 100),
    (dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True), 100),
    (dict(analyzer="char", ngram_range=(2, 4), min_df=2, sublinear_tf=True), 80),
]
for sp, nc in svd_specs:
    f_tr, f_te = svd_embed_features(sp, s1_tr, s2_tr, s1_te, s2_te, n_comp=nc)
    X_vec_tr = np.hstack([X_vec_tr, f_tr]); X_vec_te = np.hstack([X_vec_te, f_te])
print("after SVD:", X_vec_tr.shape)


# ---------- Lexical features ----------
def word_tokenize(s):
    return s.split(" ")

def lex_features(s1, s2):
    feats = []
    for a, b in zip(s1, s2):
        ta = word_tokenize(a); tb = word_tokenize(b)
        sa = set(ta); sb = set(tb)
        inter = len(sa & sb); uni = len(sa | sb)
        jacc = inter / uni if uni else 0.0
        overlap1 = inter / len(sa) if sa else 0.0
        overlap2 = inter / len(sb) if sb else 0.0
        lena = len(a); lenb = len(b)
        lens_ratio = lena / lenb if lenb else 0.0
        abs_len_diff = abs(lena - lenb)
        pref = 0
        for ch_a, ch_b in zip(a, b):
            if ch_a == ch_b: pref += 1
            else: break
        ba = set(a[i:i+2] for i in range(max(0, len(a)-1)))
        bb = set(b[i:i+2] for i in range(max(0, len(b)-1)))
        bg_inter = len(ba & bb); bg_uni = len(ba | bb)
        bg_jacc = bg_inter / bg_uni if bg_uni else 0.0
        tba = set(zip(ta, ta[1:])) if len(ta) > 1 else set()
        tbb = set(zip(tb, tb[1:])) if len(tb) > 1 else set()
        tbg_inter = len(tba & tbb); tbg_uni = len(tba | tbb)
        tbg_jacc = tbg_inter / tbg_uni if tbg_uni else 0.0
        ta3 = set(a[i:i+3] for i in range(max(0, len(a)-2)))
        tb3 = set(b[i:i+3] for i in range(max(0, len(b)-2)))
        t3_inter = len(ta3 & tb3); t3_uni = len(ta3 | tb3)
        t3_jacc = t3_inter / t3_uni if t3_uni else 0.0
        feats.append([
            jacc, overlap1, overlap2, inter, uni, len(ta), len(tb),
            lens_ratio, abs_len_diff, pref, bg_jacc, tbg_jacc, t3_jacc,
            float(lena == lenb), float(a == b), lena, lenb, len(ta), len(tb),
        ])
    return np.array(feats, dtype=float)

X_lex_tr = lex_features(s1_tr, s2_tr)
X_lex_te = lex_features(s1_te, s2_te)
print("lexical features:", X_lex_tr.shape)

X_tr = np.hstack([X_vec_tr, X_lex_tr])
X_te = np.hstack([X_vec_te, X_lex_te])
print("total features:", X_tr.shape)


# ---------- Stacked CV ----------
def pearson(a, b):
    return stats.pearsonr(a, b)[0]

kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
scaler = StandardScaler()
Xs_tr = scaler.fit_transform(X_tr)
Xs_te = scaler.transform(X_te)

models = {
    "ridge1": Ridge(alpha=1.0, random_state=RNG),
    "ridge10": Ridge(alpha=10.0, random_state=RNG),
    "ridge50": Ridge(alpha=50.0, random_state=RNG),
    "ridge200": Ridge(alpha=200.0, random_state=RNG),
    "bayes": BayesianRidge(),
    "svr_c1": SVR(C=1.0, kernel="rbf", gamma="scale"),
    "svr_c3": SVR(C=3.0, kernel="rbf", gamma="scale"),
    "svr_c10": SVR(C=10.0, kernel="rbf", gamma="scale"),
}

oof = {k: np.zeros(len(y)) for k in models}
test_pred = {k: np.zeros(X_te.shape[0]) for k in models}
for fold_i, (tr_idx, va_idx) in enumerate(kf.split(Xs_tr)):
    Xa, Xb = Xs_tr[tr_idx], Xs_tr[va_idx]
    ya = y[tr_idx]
    for k, m in models.items():
        mm = type(m)(**m.get_params())
        mm.fit(Xa, ya)
        oof[k][va_idx] = mm.predict(Xb)
        test_pred[k] += mm.predict(Xs_te) / kf.n_splits
    print(f"fold {fold_i+1} done")

for k in models:
    print(f"  OOF Pearson {k}: {pearson(oof[k], y):.4f}")

stack = np.column_stack([oof[k] for k in models])
stack_te = np.column_stack([test_pred[k] for k in models])

meta_oof = np.zeros(len(y))
meta_te = np.zeros(X_te.shape[0])
for tr_idx, va_idx in kf.split(stack):
    mm = Ridge(alpha=0.5, random_state=RNG)
    mm.fit(stack[tr_idx], y[tr_idx])
    meta_oof[va_idx] = mm.predict(stack[va_idx])
    meta_te += mm.predict(stack_te) / kf.n_splits
print(f"  META OOF Pearson: {pearson(meta_oof, y):.4f}")

single_best = max(models, key=lambda k: pearson(oof[k], y))
print("best single:", single_best, pearson(oof[single_best], y))

pred_blend = 0.85 * meta_te + 0.15 * test_pred[single_best]
pred_blend = np.clip(pred_blend, 0.0, 5.0)

sub = pd.DataFrame({"id": test["id"], "score": pred_blend})
sub.to_csv(f"{BASE}/outputs/submission.csv", index=False)
print("saved submission", sub.shape)
print(sub.head())
print(sub["score"].describe())
