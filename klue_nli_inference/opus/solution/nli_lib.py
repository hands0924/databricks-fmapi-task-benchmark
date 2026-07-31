"""Shared library for KLUE-NLI task (t6_klue_nli).

Two ingredients:
  1) A sparse-feature linear classifier P(label | premise, hypothesis) built from
     TF-IDF over word / char / "hypothesis-only" / "shared" token views plus
     hand-crafted NLI cue features.
  2) A group-constraint decoder.  In KLUE-NLI every premise is paired with
     exactly three hypotheses -- one entailment, one neutral, one contradiction.
     Because the train/test split is row-wise (not premise-wise), most test rows
     share their premise with 1-2 *labelled* train rows.  Decoding each premise
     group jointly under a prior over label multisets recovers the missing label
     with very high accuracy.

No internet, no external data, no pretrained weights: train.csv only.
"""
import itertools
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CLASSES = ["entailment", "neutral", "contradiction"]
C2I = {c: i for i, c in enumerate(CLASSES)}

# --------------------------------------------------------------------------- #
# text utilities
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[^0-9A-Za-z\uac00-\ud7a3\u3131-\u318e]+")
_NUM = re.compile(r"\d+")

NEG_PAT = [
    "않", "없", "못", "아니", "말고", "반대", "금지", "불가", "거짓", "실패",
    "전혀", "결코", "아무", "무",
]
QUANT_PAT = [
    "모든", "모두", "전부", "항상", "언제나", "절대", "유일", "오직", "가장",
    "최고", "최초", "제일", "매우", "너무", "조금", "약간", "일부", "대부분",
    "많", "적", "몇",
]
HEDGE_PAT = [
    "것이다", "일 것", "생각", "싶", "좋아", "싫", "예정", "계획", "아마",
    "듯", "같다", "때문", "위해", "려고", "겠",
]


def norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    return s.strip()


def toks(s):
    """whitespace/punct tokens"""
    return [t for t in _PUNCT.split(norm(s).lower()) if t]


def stems(s):
    """crude Korean morphological normalisation: keep leading 2..4 chars of
    every token (Korean stems sit at the front, particles/endings at the back)."""
    out = []
    for t in toks(s):
        out.append(t)
        for k in (2, 3, 4):
            if len(t) > k:
                out.append(t[:k])
    return out


def stemset(s):
    return set(stems(s))


def char3(s):
    z = re.sub(r"\s+", "", norm(s))
    return set(z[i:i + 3] for i in range(max(len(z) - 2, 0))) or {z}


def _view_texts(df):
    """Build the several text 'views' fed to separate TF-IDF vectorisers."""
    prem, hyp = df.premise.values, df.hypothesis.values
    v_h, v_p, v_new, v_sh, v_drop, v_pair = [], [], [], [], [], []
    for p, h in zip(prem, hyp):
        tp, th = toks(p), toks(h)
        sp_ = stemset(p)
        sh_ = stemset(h)
        v_h.append(" ".join(stems(h)))
        v_p.append(" ".join(stems(p)))
        # tokens of H whose stem is unseen in P  -> new information
        new = [t for t in th if t not in sp_ and (t[:2] not in sp_)]
        v_new.append(" ".join(new) if new else "∅")
        shared = [t for t in th if t in sp_ or t[:2] in sp_]
        v_sh.append(" ".join(shared) if shared else "∅")
        drop = [t for t in tp if t not in sh_ and (t[:2] not in sh_)]
        v_drop.append(" ".join(drop) if drop else "∅")
        v_pair.append(norm(p) + " ‖ " + norm(h))
    return dict(h=v_h, p=v_p, new=v_new, sh=v_sh, drop=v_drop, pair=v_pair)


def dense_feats(df):
    rows = []
    for p, h in zip(df.premise.values, df.hypothesis.values):
        tp, th = toks(p), toks(h)
        sp_, sh_ = set(tp), set(th)
        stp, sth = stemset(p), stemset(h)
        cp, ch = char3(p), char3(h)
        lp, lh = len(norm(p)), len(norm(h))
        ntp, nth = max(len(tp), 1), max(len(th), 1)
        inter_t = len(sp_ & sh_)
        inter_s = len(stp & sth)
        inter_c = len(cp & ch)
        newtok = sum(1 for t in th if t not in stp and t[:2] not in stp)
        negh = sum(h.count(w) for w in NEG_PAT)
        negp = sum(p.count(w) for w in NEG_PAT)
        qh = sum(h.count(w) for w in QUANT_PAT)
        qp = sum(p.count(w) for w in QUANT_PAT)
        hh = sum(h.count(w) for w in HEDGE_PAT)
        hp = sum(p.count(w) for w in HEDGE_PAT)
        nump = set(_NUM.findall(p))
        numh = set(_NUM.findall(h))
        rows.append([
            lp, lh, lh / max(lp, 1), lp - lh,
            ntp, nth, nth / ntp, ntp - nth,
            inter_t, inter_t / nth, inter_t / ntp,
            inter_s, inter_s / max(len(sth), 1), inter_s / max(len(stp), 1),
            inter_c, inter_c / max(len(ch), 1), inter_c / max(len(cp), 1),
            len(cp & ch) / max(len(cp | ch), 1),
            len(sp_ & sh_) / max(len(sp_ | sh_), 1),
            newtok, newtok / nth,
            negh, negp, negh - negp, float(negh > 0), float(negp > 0),
            float((negh > 0) != (negp > 0)),
            qh, qp, qh - qp, hh, hp, hh - hp,
            len(numh), len(nump), len(numh - nump), float(len(numh - nump) > 0),
            float(len(numh) > 0 and len(nump) == 0),
            float(norm(h) in norm(p)),
            float(th[-1] == tp[-1] if th and tp else 0.0),
        ])
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------- #
# feature matrix
# --------------------------------------------------------------------------- #
VIEW_SPEC = [
    ("h",    dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("h",    dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                  max_features=300000)),
    ("p",    dict(analyzer="word", ngram_range=(1, 1), min_df=3, sublinear_tf=True)),
    ("new",  dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("new",  dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True,
                  max_features=200000)),
    ("sh",   dict(analyzer="word", ngram_range=(1, 1), min_df=2, sublinear_tf=True)),
    ("drop", dict(analyzer="word", ngram_range=(1, 1), min_df=3, sublinear_tf=True)),
    ("pair", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=4, sublinear_tf=True,
                  max_features=300000)),
]


def build_features(df_all, n_fit):
    """Fit vectorisers on rows [0:n_fit] (labelled part) and transform everything.

    Returns a CSR matrix aligned with df_all.
    """
    views = _view_texts(df_all)
    blocks = []
    for name, kw in VIEW_SPEC:
        v = TfidfVectorizer(**kw)
        txt = views[name]
        v.fit(txt[:n_fit])
        blocks.append(v.transform(txt))
    dn = dense_feats(df_all)
    sc = StandardScaler().fit(dn[:n_fit])
    blocks.append(sp.csr_matrix(np.clip(sc.transform(dn), -6, 6)))
    return sp.hstack(blocks).tocsr()


def fit_predict_proba(X, y, tr_idx, pr_idx, C=2.0, seed=0):
    clf = LogisticRegression(C=C, max_iter=3000, n_jobs=-1, random_state=seed)
    clf.fit(X[tr_idx], y[tr_idx])
    return clf.predict_proba(X[pr_idx]), clf


# --------------------------------------------------------------------------- #
# group-constraint decoding
# --------------------------------------------------------------------------- #
def multiset_prior(labels_of_group_lists):
    """Estimate log-weights for label multisets of fully observed size-3 groups.

    Returns (w3, rho):
      w3  : dict {n_distinct_labels: per-ordered-tuple weight} for size-3 groups
      rho : pairwise 'same label' penalty used for other group sizes
    """
    n_dist, n_pair, n_same = 0, 0, 0
    for ls in labels_of_group_lists:
        u = len(set(ls))
        if u == 3:
            n_dist += 1
        elif u == 2:
            n_pair += 1
        else:
            n_same += 1
    # per-ordered-tuple mass: 6 distinct tuples, 18 pair tuples, 3 same tuples
    a = max(n_dist, 1) / 6.0
    b = max(n_pair, 0.5) / 18.0
    c = max(n_same, 0.5) / 3.0
    w3 = {3: 1.0, 2: b / a, 1: c / a}
    rho = w3[2]
    return w3, rho


def group_weight(labels, w3, rho):
    k = len(labels)
    if k == 3:
        return w3[len(set(labels))]
    if k <= 1:
        return 1.0
    same_pairs = sum(1 for i in range(k) for j in range(i + 1, k)
                     if labels[i] == labels[j])
    return rho ** same_pairs


def decode_groups(keys_known, y_known, keys_unk, proba_unk, w3, rho, temp=1.0,
                  max_enum=3 ** 7):
    """Joint decoding.

    keys_known / keys_unk : group key (premise) arrays
    y_known               : int labels for known rows
    proba_unk             : (n_unk, 3) classifier probabilities
    temp                  : temperature on classifier log-probs (>1 = softer)
    Returns posterior marginals (n_unk, 3).
    """
    P = np.clip(np.asarray(proba_unk, dtype=np.float64), 1e-9, 1.0)
    logP = np.log(P) / max(temp, 1e-6)
    logP -= logP.max(axis=1, keepdims=True)
    W = np.exp(logP)

    known_by_key = {}
    for k, yy in zip(keys_known, y_known):
        known_by_key.setdefault(k, []).append(int(yy))
    unk_by_key = {}
    for i, k in enumerate(keys_unk):
        unk_by_key.setdefault(k, []).append(i)

    out = np.zeros_like(W)
    for k, idxs in unk_by_key.items():
        kn = known_by_key.get(k, [])
        m = len(idxs)
        if 3 ** m > max_enum:          # pathological group: fall back
            out[idxs] = W[idxs]
            continue
        acc = np.zeros((m, 3))
        for combo in itertools.product(range(3), repeat=m):
            wt = group_weight(kn + list(combo), w3, rho)
            if wt <= 0:
                continue
            s = wt
            for j, c in enumerate(combo):
                s *= W[idxs[j], c]
            if s == 0.0:
                continue
            for j, c in enumerate(combo):
                acc[j, c] += s
        rs = acc.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        out[idxs] = acc / rs
    return out


# --------------------------------------------------------------------------- #
# pairwise (within-group) rankers
# --------------------------------------------------------------------------- #
PAIRS = [(0, 1), (0, 2), (1, 2)]   # (entail,neutral) (entail,contra) (neutral,contra)


def make_pair_examples(keys, y, idx):
    """All same-premise row pairs with distinct labels, restricted to `idx`."""
    pos = {}
    for i in idx:
        pos.setdefault(keys[i], []).append(i)
    out = {p: ([], []) for p in PAIRS}
    for k, rows in pos.items():
        for a in range(len(rows)):
            for b in range(len(rows)):
                if a == b:
                    continue
                i, j = rows[a], rows[b]
                la, lb = int(y[i]), int(y[j])
                if la == lb:
                    continue
                key = (la, lb) if (la, lb) in out else (lb, la)
                if (la, lb) == key:
                    out[key][0].append(i); out[key][1].append(j)
                else:
                    out[key][0].append(j); out[key][1].append(i)
    return out


def fit_pair_rankers(X, keys, y, idx, C=1.0):
    """For each label pair (A,B) fit an antisymmetric logistic model on feature
    differences, then return z[(A,B)] = X @ w  (a scalar per row)."""
    ex = make_pair_examples(keys, y, idx)
    Z = {}
    for p, (ia, ib) in ex.items():
        if len(ia) < 30:
            Z[p] = np.zeros(X.shape[0]); continue
        ia = np.asarray(ia); ib = np.asarray(ib)
        Dm = X[ia] - X[ib]                       # label order = p  -> y=1
        Dall = sp.vstack([Dm, -Dm]).tocsr()
        yy = np.r_[np.ones(len(ia)), np.zeros(len(ia))]
        m = LogisticRegression(C=C, max_iter=3000, fit_intercept=False, n_jobs=-1)
        m.fit(Dall, yy)
        Z[p] = (X @ m.coef_.ravel())
    return Z


def _logsig(x):
    return -np.logaddexp(0.0, -x)


def decode_groups_pair(keys_known, y_known, keys_unk, proba_unk, Z_unk, Z_known,
                       w3, rho, temp=2.0, beta=1.0, max_enum=3 ** 7):
    """Group MRF decoding: multiset prior x base unary x pairwise ranker terms."""
    P = np.clip(np.asarray(proba_unk, dtype=np.float64), 1e-9, 1.0)
    lu = np.log(P) / max(temp, 1e-6)
    lu -= lu.max(axis=1, keepdims=True)

    known_by_key = {}
    for i, (k, yy) in enumerate(zip(keys_known, y_known)):
        known_by_key.setdefault(k, []).append((i, int(yy)))
    unk_by_key = {}
    for i, k in enumerate(keys_unk):
        unk_by_key.setdefault(k, []).append(i)

    def zval(is_unk, ri, pair):
        return (Z_unk[pair][ri] if is_unk else Z_known[pair][ri])

    out = np.zeros_like(lu)
    for k, idxs in unk_by_key.items():
        kn = known_by_key.get(k, [])
        m = len(idxs)
        if 3 ** m > max_enum:
            out[idxs] = np.exp(lu[idxs] - lu[idxs].max(1, keepdims=True))
            out[idxs] /= out[idxs].sum(1, keepdims=True)
            continue
        # slots: (is_unk, row_index_in_its_array, fixed_label or None)
        slots = [(True, i, None) for i in idxs] + [(False, i, l) for i, l in kn]
        logs, combos = [], []
        for combo in itertools.product(range(3), repeat=m):
            labels = list(combo) + [s[2] for s in slots[m:]]
            wt = group_weight(labels, w3, rho)
            if wt <= 0:
                continue
            s = np.log(wt)
            for j, c in enumerate(combo):
                s += lu[idxs[j], c]
            if beta != 0.0:
                for a in range(len(slots)):
                    for b in range(a + 1, len(slots)):
                        la, lb = labels[a], labels[b]
                        if la == lb:
                            continue
                        pr = (la, lb) if (la, lb) in PAIRS else (lb, la)
                        za = zval(slots[a][0], slots[a][1], pr)
                        zb = zval(slots[b][0], slots[b][1], pr)
                        d = (za - zb) if (la, lb) == pr else (zb - za)
                        s += beta * _logsig(d)
            logs.append(s); combos.append(combo)
        if not logs:
            out[idxs] = 1.0 / 3
            continue
        logs = np.asarray(logs); logs -= logs.max()
        wgt = np.exp(logs)
        acc = np.zeros((m, 3))
        for w_, combo in zip(wgt, combos):
            for j, c in enumerate(combo):
                acc[j, c] += w_
        acc /= np.maximum(acc.sum(1, keepdims=True), 1e-300)
        out[idxs] = acc
    return out
