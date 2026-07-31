"""Feature construction for KorNLI (t10). CPU-only, no external data.

Builds:
  * sparse blocks: char n-gram, word n-gram, token-prefix ("pseudo-stem") views of
    the hypothesis, the premise/hypothesis overlap (elementwise min) and the
    hypothesis novelty (B - min).  This is the classic "aligned / unaligned"
    lexical NLI representation.
  * dense engineered features: lengths, IDF weighted coverage, jaccard,
    negation / quantifier / number asymmetries, difflib similarity, LSA cosine.
"""
import re
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher

TOKEN_RE = re.compile(r"[0-9]+|[^\W_]+", re.UNICODE)

NEG = ["아니", "않", "없", "못", "안 ", "결코", "절대", "전혀", "아무", "무"]
QUANT = ["모든", "모두", "항상", "언제나", "매우", "가장", "최고", "유일", "오직", "만"]
HEDGE = ["아마", "것 같", "듯", "수도", "생각", "믿", "좋아", "싫", "선호", "가장 좋",
         "많은", "여러", "몇", "때문", "위해", "이유", "매년", "매일"]


def tok(s):
    return TOKEN_RE.findall(s)


def stem_tokens(s):
    """Crude Korean stemming: keep the first 2 chars of each token (drops most
    inflectional endings / particles)."""
    out = []
    for t in tok(s):
        out.append(t[:2] if len(t) > 2 else t)
    return out


def stem_join(s):
    return " ".join(stem_tokens(s))


def build_sparse(tr, te, verbose=True):
    """Return (Xtr, Xte) sparse feature matrices + list of block sizes."""
    s1 = pd.concat([tr.sentence1, te.sentence1]).tolist()
    s2 = pd.concat([tr.sentence2, te.sentence2]).tolist()
    ntr = len(tr)
    corpus = s1 + s2

    blocks_tr, blocks_te, names = [], [], []

    specs = [
        ("char", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                      sublinear_tf=True, max_features=400000)),
        ("word", dict(analyzer="word", tokenizer=tok, ngram_range=(1, 2), min_df=3,
                      sublinear_tf=True, max_features=300000, lowercase=False)),
        ("stem", dict(analyzer="word", tokenizer=stem_tokens, ngram_range=(1, 1),
                      min_df=2, sublinear_tf=True, lowercase=False)),
    ]
    for name, kw in specs:
        vec = TfidfVectorizer(**kw)
        vec.fit(corpus)
        A = vec.transform(s1)
        B = vec.transform(s2)
        M = A.minimum(B)
        for tag, blk in [("hyp", B), ("ovl", M), ("nov", (B - M).tocsr()),
                         ("prem_nov", (A - M).tocsr())]:
            blk.eliminate_zeros()
            blocks_tr.append(blk[:ntr])
            blocks_te.append(blk[ntr:])
            names.append(f"{name}_{tag}:{blk.shape[1]}")
        if verbose:
            print("  sparse", name, B.shape[1], flush=True)
    Xtr = sp.hstack(blocks_tr, format="csr")
    Xte = sp.hstack(blocks_te, format="csr")
    return Xtr, Xte, names


def _idf_map(corpus):
    v = TfidfVectorizer(analyzer="word", tokenizer=stem_tokens, min_df=1, lowercase=False)
    v.fit(corpus)
    return dict(zip(v.get_feature_names_out(), v.idf_))


def dense_features(df, idf):
    rows = []
    med = np.median(list(idf.values()))
    for s1, s2 in zip(df.sentence1.values, df.sentence2.values):
        w1, w2 = tok(s1), tok(s2)
        t1, t2 = stem_tokens(s1), stem_tokens(s2)
        S1, S2 = set(t1), set(t2)
        W1, W2 = set(w1), set(w2)
        c1 = set(s1.replace(" ", ""))
        c2 = set(s2.replace(" ", ""))
        inter = S1 & S2
        uni = S1 | S2
        idf_i = sum(idf.get(t, med) for t in inter)
        idf_2 = sum(idf.get(t, med) for t in S2) or 1e-6
        idf_1 = sum(idf.get(t, med) for t in S1) or 1e-6
        nov = S2 - S1
        idf_nov = sum(idf.get(t, med) for t in nov)
        maxnov = max([idf.get(t, med) for t in nov], default=0.0)
        # char 3-gram overlap
        def cng(s, n=3):
            s = s.replace(" ", "")
            return set(s[i:i + n] for i in range(max(len(s) - n + 1, 1)))
        g1, g2 = cng(s1), cng(s2)
        gi = len(g1 & g2)
        f = [
            len(s1), len(s2), len(s2) / max(len(s1), 1), len(s1) - len(s2),
            len(w1), len(w2), len(w2) / max(len(w1), 1), len(w1) - len(w2),
            len(inter), len(inter) / max(len(S2), 1), len(inter) / max(len(S1), 1),
            len(inter) / max(len(uni), 1),
            len(W1 & W2), len(W1 & W2) / max(len(W2), 1),
            len(c1 & c2) / max(len(c2), 1), len(c1 & c2) / max(len(c1 | c2), 1),
            idf_i, idf_i / idf_2, idf_i / idf_1,
            idf_nov, idf_nov / idf_2, maxnov, len(nov), len(nov) / max(len(S2), 1),
            gi, gi / max(len(g2), 1), gi / max(len(g1 | g2), 1),
            SequenceMatcher(None, s1[:200], s2[:200]).ratio(),
        ]
        # cue-based asymmetry features
        for cues in (NEG, QUANT, HEDGE):
            a = sum(c in s1 for c in cues)
            b = sum(c in s2 for c in cues)
            f += [a, b, b - a, float(b > 0 and a == 0)]
        n1 = set(re.findall(r"\d+", s1)); n2 = set(re.findall(r"\d+", s2))
        f += [len(n1), len(n2), len(n1 & n2), float(len(n2 - n1) > 0),
              float(len(n2) > 0 and len(n1) == 0)]
        # proper-noun-ish: latin tokens
        l1 = set(re.findall(r"[A-Za-z]+", s1)); l2 = set(re.findall(r"[A-Za-z]+", s2))
        f += [len(l1), len(l2), len(l1 & l2), float(len(l2 - l1) > 0)]
        # first / last token match
        f += [float(bool(t1) and bool(t2) and t1[0] == t2[0]),
              float(bool(t1) and bool(t2) and t1[-1] == t2[-1])]
        rows.append(f)
    return np.asarray(rows, dtype=np.float32)


def build_dense(tr, te):
    corpus = pd.concat([tr.sentence1, tr.sentence2, te.sentence1, te.sentence2]).tolist()
    idf = _idf_map(corpus)
    return dense_features(tr, idf), dense_features(te, idf)
