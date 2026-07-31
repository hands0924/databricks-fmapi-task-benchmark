"""KLUE-STS solution: TF-IDF similarity + manual lexical features + GBR/ExtraTrees ensemble.

No external data / no pretrained weights. Pure sklearn + scipy.
Validated with 5-fold OOF CV (Pearson ~0.942).

Features (38 total):
  - Per TF-IDF vectorizer (5 configs: char_wb 2-5, char 2-5, word 1-2, char 1-3,
    char_wb 3-6): cosine(dot), L1 distance, sum, mean-diff  -> 4 each
  - Manual lexical features (18): token Jaccard/overlap, char bigram/trigram/4gram
    Jaccard, difflib char & token ratios, token cosine, lengths & logs.

Models:
  - GradientBoostingRegressor(500, depth=3, lr=0.03, subsample=0.8)
  - ExtraTreesRegressor(300, n_jobs=-1)
  - Final prediction = 0.5 * GBR + 0.5 * ET, averaged over 5 fold splits.
"""
import os
import difflib
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


def vec_feats(s1, s2, vec):
    a = normalize(vec.transform(s1))
    b = normalize(vec.transform(s2))
    dot = np.asarray((a.multiply(b)).sum(axis=1)).ravel()
    abs_d = np.asarray(np.abs(a - b).sum(axis=1)).ravel()
    plus = np.asarray((a + b).sum(axis=1)).ravel()
    diff_mean = np.asarray((a - b).mean(axis=1)).ravel()
    return np.column_stack([dot, abs_d, plus, diff_mean])


def manual_feats(s1_list, s2_list):
    out = []
    for s1, s2 in zip(s1_list, s2_list):
        s1 = str(s1)
        s2 = str(s2)
        t1 = set(s1.split())
        t2 = set(s2.split())
        u = t1 | t2
        jac = len(t1 & t2) / len(u) if u else 0
        ov1 = len(t1 & t2) / len(t1) if t1 else 0
        ov2 = len(t1 & t2) / len(t2) if t2 else 0
        l1 = len(s1)
        l2 = len(s2)
        ld = difflib.SequenceMatcher(None, s1, s2).ratio()
        tt1 = s1.split()
        tt2 = s2.split()
        tok = difflib.SequenceMatcher(None, tt1, tt2).ratio()
        b1 = set(s1[i:i + 2] for i in range(len(s1) - 1))
        b2 = set(s2[i:i + 2] for i in range(len(s2) - 1))
        bu = b1 | b2
        bigram_jac = len(b1 & b2) / len(bu) if bu else 0
        t1g = set(s1[i:i + 3] for i in range(len(s1) - 2))
        t2g = set(s2[i:i + 3] for i in range(len(s2) - 2))
        tu = t1g | t2g
        tri_jac = len(t1g & t2g) / len(tu) if tu else 0
        f1g = set(s1[i:i + 4] for i in range(len(s1) - 3))
        f2g = set(s2[i:i + 4] for i in range(len(s2) - 3))
        fu = f1g | f2g
        four_jac = len(f1g & f2g) / len(fu) if fu else 0
        c1 = Counter(tt1)
        c2 = Counter(tt2)
        all_toks = set(c1) | set(c2)
        dot_t = sum(c1[t] * c2[t] for t in all_toks)
        n1 = np.sqrt(sum(v * v for v in c1.values()))
        n2 = np.sqrt(sum(v * v for v in c2.values()))
        tok_cos = dot_t / (n1 * n2) if n1 > 0 and n2 > 0 else 0
        out.append([
            jac, ov1, ov2, l1, l2, abs(l1 - l2), l1 + l2, ld, tok,
            bigram_jac, tri_jac, four_jac, np.log1p(l1), np.log1p(l2), tok_cos,
            len(tt1), len(tt2), abs(len(tt1) - len(tt2)),
        ])
    return np.array(out)


def build_X(df, vec_configs):
    Xs = []
    for v in vec_configs.values():
        Xs.append(csr_matrix(vec_feats(df["sentence1"], df["sentence2"], v)))
    Xman = csr_matrix(manual_feats(df["sentence1"], df["sentence2"]))
    return hstack(Xs + [Xman]).toarray()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    tr = pd.read_csv(os.path.join(root, "train.csv")).fillna("")
    te = pd.read_csv(os.path.join(root, "test.csv")).fillna("")
    y = tr["score"].values

    corpus = pd.concat([tr["sentence1"], tr["sentence2"]])
    vec_configs = {
        "charwb_2_5": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        "char_2_5": TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        "word_1_2": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+"),
        "char_1_3": TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
        "charwb_3_6": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=3, sublinear_tf=True),
    }
    for v in vec_configs.values():
        v.fit(corpus)
    Xtr = build_X(tr, vec_configs)
    Xte = build_X(te, vec_configs)

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_g = np.zeros(len(tr))
    oof_e = np.zeros(len(tr))
    test_g = np.zeros(len(te))
    test_e = np.zeros(len(te))
    for tr_idx, va_idx in kf.split(Xtr_s):
        g = GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42)
        g.fit(Xtr_s[tr_idx], y[tr_idx])
        oof_g[va_idx] = g.predict(Xtr_s[va_idx])
        test_g += g.predict(Xte_s) / kf.n_splits
        e = ExtraTreesRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
        e.fit(Xtr_s[tr_idx], y[tr_idx])
        oof_e[va_idx] = e.predict(Xtr_s[va_idx])
        test_e += e.predict(Xte_s) / kf.n_splits

    rg, _ = pearsonr(oof_g, y)
    re, _ = pearsonr(oof_e, y)
    ens_oof = 0.5 * oof_g + 0.5 * oof_e
    rens, _ = pearsonr(ens_oof, y)
    print(f"CV Pearson -- GBR: {rg:.5f}  ET: {re:.5f}  Ensemble: {rens:.5f}")

    test_pred = 0.5 * test_g + 0.5 * test_e
    pred = np.clip(test_pred, 0, 5)
    out = os.path.join(root, "outputs", "submission.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"id": te["id"], "score": pred}).to_csv(out, index=False)
    print(f"Wrote {out} ({len(pred)} rows)")


if __name__ == "__main__":
    main()
