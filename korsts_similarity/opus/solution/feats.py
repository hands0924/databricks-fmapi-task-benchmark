"""Feature engineering for KorSTS (no external data / no pretrained weights)."""
import re
import unicodedata
import difflib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def to_jamo(s):
    out = []
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            i = o - 0xAC00
            out.append(CHO[i // 588])
            out.append(JUNG[(i % 588) // 28])
            j = JONG[i % 28]
            if j != " ":
                out.append(j)
        else:
            out.append(ch)
    return "".join(out)


PUNCT = re.compile(r"[^0-9A-Za-z가-힣\s]")
WS = re.compile(r"\s+")
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = PUNCT.sub(" ", s)
    return WS.sub(" ", s).strip()


def stem_tokens(s, k=2):
    """Korean is agglutinative: suffixes attach at the end, so a word prefix
    approximates the stem."""
    return " ".join(t[:k] if len(t) > k else t for t in s.split())


def tokens(s):
    return s.split()


def _safe_div(a, b):
    return a / b if b else 0.0


def set_feats(a, b):
    A, B = set(a), set(b)
    inter = len(A & B)
    union = len(A | B)
    out = [
        _safe_div(inter, union),
        _safe_div(2.0 * inter, len(A) + len(B)),
        _safe_div(inter, min(len(A), len(B)) or 1),
        _safe_div(inter, max(len(A), len(B)) or 1),
        float(inter),
        abs(len(A) - len(B)),
        _safe_div(min(len(A), len(B)), max(len(A), len(B)) or 1),
    ]
    return out


def ngrams(s, n):
    return [s[i:i + n] for i in range(max(0, len(s) - n + 1))]


def lcs_len(a, b):
    """Longest common subsequence length (chars), on capped-length strings."""
    a, b = a[:200], b[:200]
    if not a or not b:
        return 0
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    for ca in a:
        cur = np.zeros(len(b) + 1, dtype=np.int32)
        for j, cb in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1])
        prev = cur
    return int(prev[-1])


def lcsubstr_len(a, b):
    sm = difflib.SequenceMatcher(None, a[:300], b[:300], autojunk=False)
    m = sm.find_longest_match(0, len(a[:300]), 0, len(b[:300]))
    return m.size


class SimSpace:
    """A TF-IDF space (with optional SVD) giving pair similarity features."""

    def __init__(self, name, analyzer, ngram_range, prep, min_df=1, svd=0,
                 sublinear=True, binary=False):
        self.name = name
        self.prep = prep
        self.svd_dim = svd
        self.vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                                   min_df=min_df, sublinear_tf=sublinear,
                                   binary=binary, dtype=np.float32)
        self.svd = None

    def fit(self, corpus):
        c = [self.prep(x) for x in corpus]
        self.X = self.vec.fit_transform(c)
        if self.svd_dim:
            self.svd = TruncatedSVD(n_components=self.svd_dim, random_state=0)
            self.Z = normalize(self.svd.fit_transform(self.X))
        return self

    def transform(self, s1, s2):
        A = normalize(self.vec.transform([self.prep(x) for x in s1]))
        B = normalize(self.vec.transform([self.prep(x) for x in s2]))
        cos = np.asarray(A.multiply(B).sum(1)).ravel()
        feats = {f"{self.name}_cos": cos}
        # binary-ish overlap in tfidf space
        Ab = A.copy(); Ab.data[:] = 1.0
        Bb = B.copy(); Bb.data[:] = 1.0
        inter = np.asarray(Ab.multiply(Bb).sum(1)).ravel()
        na = np.asarray(Ab.sum(1)).ravel(); nb = np.asarray(Bb.sum(1)).ravel()
        feats[f"{self.name}_jac"] = inter / np.maximum(na + nb - inter, 1)
        feats[f"{self.name}_cont"] = inter / np.maximum(np.minimum(na, nb), 1)
        # idf-weighted coverage (asymmetric -> min & max)
        idf = self.vec.idf_
        wa = A.copy(); wb = B.copy()
        cov_a = np.zeros(len(cos)); cov_b = np.zeros(len(cos))
        Aidx = A.tocsr(); Bidx = B.tocsr()
        for i in range(Aidx.shape[0]):
            ia = Aidx.indices[Aidx.indptr[i]:Aidx.indptr[i + 1]]
            ib = Bidx.indices[Bidx.indptr[i]:Bidx.indptr[i + 1]]
            sa = set(ia.tolist()); sb = set(ib.tolist())
            common = sa & sb
            wsum_a = idf[list(sa)].sum() if sa else 1.0
            wsum_b = idf[list(sb)].sum() if sb else 1.0
            wc = idf[list(common)].sum() if common else 0.0
            cov_a[i] = wc / max(wsum_a, 1e-9)
            cov_b[i] = wc / max(wsum_b, 1e-9)
        feats[f"{self.name}_idfcov_min"] = np.minimum(cov_a, cov_b)
        feats[f"{self.name}_idfcov_max"] = np.maximum(cov_a, cov_b)
        feats[f"{self.name}_idfcov_hm"] = 2 * cov_a * cov_b / np.maximum(cov_a + cov_b, 1e-9)
        if self.svd is not None:
            U = normalize(self.svd.transform(A))
            V = normalize(self.svd.transform(B))
            feats[f"{self.name}_svdcos"] = (U * V).sum(1)
            feats[f"{self.name}_svdl2"] = np.linalg.norm(U - V, axis=1)
            feats[f"{self.name}_svdl1"] = np.abs(U - V).sum(1)
            self.last_U, self.last_V = U, V
        return feats


def build_spaces():
    j = lambda s: to_jamo(norm(s))
    n = lambda s: norm(s)
    st2 = lambda s: stem_tokens(norm(s), 2)
    st3 = lambda s: stem_tokens(norm(s), 3)
    return [
        SimSpace("ch23", "char_wb", (2, 3), n, min_df=1, svd=200),
        SimSpace("ch45", "char_wb", (4, 5), n, min_df=1),
        SimSpace("ch25", "char", (2, 5), n, min_df=2, svd=250),
        SimSpace("jm34", "char_wb", (3, 4), j, min_df=1, svd=200),
        SimSpace("jm56", "char_wb", (5, 6), j, min_df=2),
        SimSpace("w1", "word", (1, 1), n, min_df=1, svd=150),
        SimSpace("w12", "word", (1, 2), n, min_df=1),
        SimSpace("s2", "word", (1, 1), st2, min_df=1, svd=150),
        SimSpace("s3", "word", (1, 1), st3, min_df=1),
        SimSpace("s2b", "word", (1, 2), st2, min_df=1, binary=True, sublinear=False),
    ]


def soft_align_feats(s1, s2, token_vec, idf_map, default_idf):
    """Greedy soft-alignment between token sets using char-ngram token vectors."""
    out = []
    for a, b in zip(s1, s2):
        ta = tokens(stem_tokens(norm(a), 3))
        tb = tokens(stem_tokens(norm(b), 3))
        ta_f = tokens(norm(a)); tb_f = tokens(norm(b))
        if not ta_f or not tb_f:
            out.append([0.0] * 6)
            continue
        A = token_vec.transform(ta_f)
        B = token_vec.transform(tb_f)
        S = (A @ B.T).toarray()
        wa = np.array([idf_map.get(t, default_idf) for t in ta_f])
        wb = np.array([idf_map.get(t, default_idf) for t in tb_f])
        ma = S.max(1); mb = S.max(0)
        sa = float((ma * wa).sum() / max(wa.sum(), 1e-9))
        sb = float((mb * wb).sum() / max(wb.sum(), 1e-9))
        out.append([
            sa, sb, min(sa, sb), max(sa, sb),
            float(ma.mean()), float(mb.mean()),
        ])
    return np.array(out)


def lexical_feats(s1, s2):
    rows = []
    for a, b in zip(s1, s2):
        na, nb = norm(a), norm(b)
        ta, tb = tokens(na), tokens(nb)
        sa2, sb2 = tokens(stem_tokens(na, 2)), tokens(stem_tokens(nb, 2))
        sa3, sb3 = tokens(stem_tokens(na, 3)), tokens(stem_tokens(nb, 3))
        f = []
        f += set_feats(ta, tb)
        f += set_feats(sa2, sb2)
        f += set_feats(sa3, sb3)
        f += set_feats(ngrams(na.replace(" ", ""), 2), ngrams(nb.replace(" ", ""), 2))
        f += set_feats(ngrams(na.replace(" ", ""), 3), ngrams(nb.replace(" ", ""), 3))
        ja, jb = to_jamo(na), to_jamo(nb)
        f += set_feats(ngrams(ja.replace(" ", ""), 4), ngrams(jb.replace(" ", ""), 4))
        # length
        la, lb = len(na), len(nb)
        f += [la, lb, abs(la - lb), _safe_div(min(la, lb), max(la, lb) or 1),
              len(ta), len(tb), abs(len(ta) - len(tb)),
              _safe_div(min(len(ta), len(tb)), max(len(ta), len(tb)) or 1)]
        # string similarity
        f.append(difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio())
        f.append(difflib.SequenceMatcher(None, ja, jb, autojunk=False).ratio())
        L = lcs_len(na, nb)
        f += [L, _safe_div(L, min(la, lb) or 1), _safe_div(2.0 * L, la + lb or 1)]
        M = lcsubstr_len(na, nb)
        f += [M, _safe_div(M, min(la, lb) or 1)]
        # numbers
        numa = set(NUM.findall(str(a))); numb = set(NUM.findall(str(b)))
        f += [len(numa & numb), len(numa ^ numb),
              float(bool(numa) != bool(numb)),
              _safe_div(len(numa & numb), len(numa | numb) or 1)]
        # latin tokens (proper nouns / brands often kept)
        lat_a = set(re.findall(r"[a-z]{2,}", na)); lat_b = set(re.findall(r"[a-z]{2,}", nb))
        f += [len(lat_a & lat_b), len(lat_a ^ lat_b),
              _safe_div(len(lat_a & lat_b), len(lat_a | lat_b) or 1)]
        # first / last token match (topic & predicate cues)
        f += [float(ta[:1] == tb[:1]), float(ta[-1:] == tb[-1:]),
              float(sa2[:1] == sb2[:1]), float(sa2[-1:] == sb2[-1:])]
        # negation cues
        neg = ["안", "않", "없", "못", "아니", "no", "not"]
        na_neg = sum(na.count(w) for w in neg); nb_neg = sum(nb.count(w) for w in neg)
        f += [na_neg, nb_neg, abs(na_neg - nb_neg), float((na_neg > 0) != (nb_neg > 0))]
        # question cues
        f += [float("?" in str(a)), float("?" in str(b)),
              float(("?" in str(a)) != ("?" in str(b)))]
        rows.append(f)
    return np.array(rows, dtype=np.float64)


LEX_N = None


def make_features(tr, te, return_svd=True):
    corpus = pd.concat([tr.sentence1, tr.sentence2, te.sentence1, te.sentence2]).astype(str).tolist()
    spaces = build_spaces()
    for sp in spaces:
        sp.fit(corpus)

    # token-level vectorizer for soft alignment
    all_tokens = sorted({t for s in corpus for t in tokens(norm(s))})
    token_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1,
                                sublinear_tf=True, dtype=np.float32)
    token_vec.fit([to_jamo(t) for t in all_tokens])

    class JamoTokVec:
        def transform(self, toks):
            return normalize(token_vec.transform([to_jamo(t) for t in toks]))
    jtv = JamoTokVec()

    # token idf from sentence corpus
    tv = TfidfVectorizer(analyzer="word", min_df=1)
    tv.fit([norm(s) for s in corpus])
    idf_map = {w: tv.idf_[i] for w, i in tv.vocabulary_.items()}
    default_idf = float(tv.idf_.max())

    out = {}
    svd_pack = {}
    for split, df in (("tr", tr), ("te", te)):
        s1 = df.sentence1.astype(str).tolist()
        s2 = df.sentence2.astype(str).tolist()
        feats = {}
        for sp in spaces:
            feats.update(sp.transform(s1, s2))
            if sp.svd is not None:
                svd_pack.setdefault(sp.name, {})[split] = (sp.last_U, sp.last_V)
        F = pd.DataFrame(feats)
        SA = soft_align_feats(s1, s2, jtv, idf_map, default_idf)
        F = pd.concat([F, pd.DataFrame(SA, columns=[f"align{i}" for i in range(SA.shape[1])])], axis=1)
        LX = lexical_feats(s1, s2)
        F = pd.concat([F, pd.DataFrame(LX, columns=[f"lex{i}" for i in range(LX.shape[1])])], axis=1)
        out[split] = F
    return out["tr"], out["te"], svd_pack
