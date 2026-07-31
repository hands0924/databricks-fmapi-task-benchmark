"""Feature engineering for Korean STS (KLUE-STS) without pretrained models."""
import re
import difflib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------- text utils
CHO = list("\u3131\u3132\u3134\u3137\u3138\u3139\u3141\u3142\u3143\u3145\u3146\u3147\u3148\u3149\u314a\u314b\u314c\u314d\u314e")
JUNG = list("\u314f\u3150\u3151\u3152\u3153\u3154\u3155\u3156\u3157\u3158\u3159\u315a\u315b\u315c\u315d\u315e\u315f\u3160\u3161\u3162\u3163")
JONG = [""] + list("\u3131\u3132\u3133\u3134\u3135\u3136\u3137\u3139\u313a\u313b\u313c\u313d\u313e\u313f\u3140\u3141\u3142\u3144\u3145\u3146\u3147\u3148\u314a\u314b\u314c\u314d\u314e")


def jamo(text):
    out = []
    for ch in text:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            i = o - 0xAC00
            out.append(CHO[i // 588])
            out.append(JUNG[(i % 588) // 28])
            j = JONG[i % 28]
            if j:
                out.append(j)
        else:
            out.append(ch)
    return "".join(out)


PUNCT = re.compile(r"[^\w\s]")
SPACES = re.compile(r"\s+")
NUM = re.compile(r"\d+")


def clean(text):
    t = str(text).strip()
    t = PUNCT.sub(" ", t)
    t = SPACES.sub(" ", t)
    return t.lower()


def tokens(text):
    return clean(text).split()


def stems(text, n=2):
    """Crude Korean stemming: keep the leading n..4 chars of each eojeol.

    Korean particles/endings are suffixes, so prefixes approximate the stem.
    """
    out = []
    for w in tokens(text):
        if len(w) <= n:
            out.append(w)
        else:
            out.append(w[: max(n, len(w) - 1)] if len(w) <= 3 else w[:3])
    return out


def stem_text(text):
    return " ".join(stems(text))


# ---------------------------------------------------------------- set metrics
def _set_feats(a, b):
    A, B = set(a), set(b)
    inter = len(A & B)
    union = len(A | B) or 1
    mn = min(len(A), len(B)) or 1
    mx = max(len(A), len(B)) or 1
    return [inter / union, 2 * inter / (len(A) + len(B) + 1e-9), inter / mn, inter / mx, inter]


def ngrams(s, n):
    s = clean(s).replace(" ", "")
    return [s[i:i + n] for i in range(max(1, len(s) - n + 1))]


def lcs_len(a, b):
    """Length of longest common subsequence (chars)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def lcsubstr_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b):
            if ca == cb:
                cur[j + 1] = prev[j] + 1
                if cur[j + 1] > best:
                    best = cur[j + 1]
        prev = cur
    return best


# ---------------------------------------------------------------- main builder
class FeatureBuilder:
    def __init__(self, n_svd=200, verbose=True):
        self.n_svd = n_svd
        self.verbose = verbose

    def log(self, *a):
        if self.verbose:
            print("[feat]", *a, flush=True)

    def fit_transform(self, df_list):
        """df_list: list of DataFrames with sentence1/sentence2. Fit on all of them
        (unsupervised / transductive over the provided texts only)."""
        s1 = pd.concat([d.sentence1 for d in df_list]).astype(str).tolist()
        s2 = pd.concat([d.sentence2 for d in df_list]).astype(str).tolist()
        texts = s1 + s2
        self.log("texts", len(texts))

        # pre-computed views of the text
        c_all = [clean(t) for t in texts]
        j_all = [jamo(t) for t in c_all]
        st_all = [" ".join(stems(t)) for t in texts]

        blocks = []
        names = []

        # ------------- vectorizer family: cosine + SVD-cosine per representation
        specs = [
            ("cw24", dict(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2), c_all),
            ("cw35", dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2), c_all),
            ("c23", dict(analyzer="char", ngram_range=(2, 3), sublinear_tf=True, min_df=2), c_all),
            ("w11", dict(analyzer="word", ngram_range=(1, 1), sublinear_tf=True, min_df=1), c_all),
            ("w12", dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1), c_all),
            ("st11", dict(analyzer="word", ngram_range=(1, 1), sublinear_tf=True, min_df=1), st_all),
            ("jm34", dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2), j_all),
        ]
        n = len(s1)
        self.vecs = {}
        self.svds = {}
        svd_pair_blocks = []
        for tag, kw, view in specs:
            v = TfidfVectorizer(**kw)
            X = v.fit_transform(view)
            Xn = normalize(X)
            A, B = Xn[:n], Xn[n:]
            cos = np.asarray(A.multiply(B).sum(1)).ravel()
            blocks.append(cos.reshape(-1, 1))
            names.append(f"cos_{tag}")
            # asymmetric idf-weighted coverage using binary presence
            Xb = (X > 0).astype(np.float32)
            idf = v.idf_
            wa = Xb[:n].multiply(idf)
            wb = Xb[n:].multiply(idf)
            both = wa.minimum(wb).sum(1)
            sa = wa.sum(1) + 1e-9
            sb = wb.sum(1) + 1e-9
            both = np.asarray(both).ravel()
            sa = np.asarray(sa).ravel()
            sb = np.asarray(sb).ravel()
            blocks.append(np.c_[both / sa, both / sb, both / np.maximum(sa, sb), both / np.minimum(sa, sb)])
            names += [f"cov_{tag}_a", f"cov_{tag}_b", f"cov_{tag}_max", f"cov_{tag}_min"]
            self.vecs[tag] = v

            if tag in ("cw24", "w12", "st11"):
                k = min(self.n_svd, X.shape[1] - 1, X.shape[0] - 1)
                sv = TruncatedSVD(n_components=k, random_state=0)
                Z = sv.fit_transform(X)
                Z = normalize(Z)
                U, V = Z[:n], Z[n:]
                blocks.append((U * V).sum(1).reshape(-1, 1))
                names.append(f"svdcos_{tag}")
                blocks.append(np.linalg.norm(U - V, axis=1).reshape(-1, 1))
                names.append(f"svdl2_{tag}")
                self.svds[tag] = sv
                if tag == "cw24":
                    kk = min(64, k)
                    svd_pair_blocks.append(np.abs(U[:, :kk] - V[:, :kk]))
                    svd_pair_blocks.append(U[:, :kk] * V[:, :kk])
            self.log("done", tag, X.shape)

        # ------------- LSA word vectors -> soft alignment (BERTScore-like)
        cvz = CountVectorizer(analyzer="word", min_df=1, token_pattern=r"\S+")
        D = cvz.fit_transform(c_all)  # docs x terms
        tf = TfidfVectorizer(analyzer="word", min_df=1, sublinear_tf=True, token_pattern=r"\S+")
        Dt = tf.fit_transform(c_all)
        TD = Dt.T.tocsr()  # terms x docs
        k = min(200, TD.shape[1] - 1, TD.shape[0] - 1)
        svw = TruncatedSVD(n_components=k, random_state=0)
        W = svw.fit_transform(TD)
        W = normalize(W)
        vocab = tf.vocabulary_
        idfw = dict(zip(tf.get_feature_names_out(), tf.idf_))
        default_idf = float(np.max(tf.idf_))
        self.log("word vecs", W.shape)

        align = np.zeros((n, 6), dtype=np.float64)
        for i in range(n):
            ta = clean(s1[i]).split()
            tb = clean(s2[i]).split()
            ia = [vocab[w] for w in ta if w in vocab]
            ib = [vocab[w] for w in tb if w in vocab]
            if not ia or not ib:
                continue
            Va = W[ia]
            Vb = W[ib]
            S = Va @ Vb.T
            wa = np.array([idfw.get(w, default_idf) for w in ta if w in vocab])
            wb = np.array([idfw.get(w, default_idf) for w in tb if w in vocab])
            ra = S.max(1)
            rb = S.max(0)
            p = float(np.dot(ra, wa) / wa.sum())
            q = float(np.dot(rb, wb) / wb.sum())
            f = 2 * p * q / (p + q) if (p + q) > 1e-9 else 0.0
            align[i] = [p, q, f, float(ra.mean()), float(rb.mean()), float(S.max())]
        blocks.append(align)
        names += ["al_p", "al_q", "al_f", "al_ra", "al_rb", "al_max"]

        # idf-weighted centroid cosine from word vectors
        cent = np.zeros((len(texts), W.shape[1]))
        for i, t in enumerate(c_all):
            ws = [w for w in t.split() if w in vocab]
            if ws:
                wt = np.array([idfw.get(w, default_idf) for w in ws])[:, None]
                cent[i] = (W[[vocab[w] for w in ws]] * wt).sum(0)
        cent = normalize(cent)
        CU, CV = cent[:n], cent[n:]
        blocks.append((CU * CV).sum(1).reshape(-1, 1))
        names.append("cent_cos")
        svd_pair_blocks.append(np.abs(CU[:, :64] - CV[:, :64]))
        svd_pair_blocks.append(CU[:, :64] * CV[:, :64])

        # ------------- surface / string features
        surf = []
        for i in range(n):
            a, b = str(s1[i]), str(s2[i])
            ca, cb = clean(a), clean(b)
            ta, tb = ca.split(), cb.split()
            row = []
            row += [len(a), len(b), abs(len(a) - len(b)), min(len(a), len(b)) / max(len(a), len(b), 1)]
            row += [len(ta), len(tb), abs(len(ta) - len(tb)), min(len(ta), len(tb)) / max(len(ta), len(tb), 1)]
            row += _set_feats(ta, tb)
            row += _set_feats(stems(a), stems(b))
            row += _set_feats(ngrams(a, 2), ngrams(b, 2))
            row += _set_feats(ngrams(a, 3), ngrams(b, 3))
            row += _set_feats(ngrams(jamo(a), 3), ngrams(jamo(b), 3))
            row.append(difflib.SequenceMatcher(None, ca, cb).ratio())
            row.append(difflib.SequenceMatcher(None, ta, tb).ratio())
            L = lcs_len(ca.replace(" ", ""), cb.replace(" ", ""))
            row += [L, L / max(1, min(len(ca), len(cb))), L / max(1, max(len(ca), len(cb)))]
            LS = lcsubstr_len(ca.replace(" ", ""), cb.replace(" ", ""))
            row += [LS, LS / max(1, min(len(ca), len(cb)))]
            na, nb = set(NUM.findall(a)), set(NUM.findall(b))
            row += [len(na), len(nb), len(na & nb), 1.0 if na == nb else 0.0,
                    1.0 if (na or nb) and not (na & nb) else 0.0]
            row += [1.0 if "?" in a else 0.0, 1.0 if "?" in b else 0.0,
                    1.0 if ("?" in a) == ("?" in b) else 0.0]
            row += [1.0 if "!" in a else 0.0, 1.0 if "!" in b else 0.0]
            # first / last token match
            row += [1.0 if ta and tb and ta[0] == tb[0] else 0.0,
                    1.0 if ta and tb and ta[-1] == tb[-1] else 0.0]
            # negation cues
            neg = ["\uc548", "\uc9c0 \uc54a", "\uc5c6", "\ubabb", "\uc54a"]
            va = sum(1 for w in neg if w in a)
            vb = sum(1 for w in neg if w in b)
            row += [va, vb, abs(va - vb)]
            surf.append(row)
        surf = np.array(surf, dtype=np.float64)
        blocks.append(surf)
        names += (["len_a", "len_b", "len_d", "len_r", "ntok_a", "ntok_b", "ntok_d", "ntok_r"]
                  + [f"tok_{x}" for x in ["jac", "dice", "cmin", "cmax", "int"]]
                  + [f"stem_{x}" for x in ["jac", "dice", "cmin", "cmax", "int"]]
                  + [f"c2_{x}" for x in ["jac", "dice", "cmin", "cmax", "int"]]
                  + [f"c3_{x}" for x in ["jac", "dice", "cmin", "cmax", "int"]]
                  + [f"j3_{x}" for x in ["jac", "dice", "cmin", "cmax", "int"]]
                  + ["sm_char", "sm_tok", "lcs", "lcs_rmin", "lcs_rmax", "lcsub", "lcsub_r",
                     "n_a", "n_b", "n_int", "n_eq", "n_conflict",
                     "q_a", "q_b", "q_same", "e_a", "e_b", "first_eq", "last_eq",
                     "neg_a", "neg_b", "neg_d"])

        # ------------- char-similarity soft token alignment (inflection robust)
        def tok3(w):
            w2 = f"^{w}$"
            return set(w2[i:i + 3] for i in range(max(1, len(w2) - 2)))

        cache3 = {}

        def g3(w):
            if w not in cache3:
                cache3[w] = tok3(w)
            return cache3[w]

        soft = np.zeros((n, 5))
        for i in range(n):
            ta = clean(s1[i]).split()
            tb = clean(s2[i]).split()
            if not ta or not tb:
                continue
            wa = np.array([idfw.get(w, default_idf) for w in ta])
            wb = np.array([idfw.get(w, default_idf) for w in tb])
            ga = [g3(w) for w in ta]
            gb = [g3(w) for w in tb]
            S = np.empty((len(ta), len(tb)))
            for x, sa_ in enumerate(ga):
                for yy, sb_ in enumerate(gb):
                    inter = len(sa_ & sb_)
                    S[x, yy] = inter / (len(sa_) + len(sb_) - inter)
            ra = S.max(1)
            rb = S.max(0)
            p = float(np.dot(ra, wa) / wa.sum())
            q = float(np.dot(rb, wb) / wb.sum())
            f = 2 * p * q / (p + q) if (p + q) > 1e-9 else 0.0
            soft[i] = [p, q, f, float(ra.mean()), float(rb.mean())]
        blocks.append(soft)
        names += ["sft_p", "sft_q", "sft_f", "sft_ra", "sft_rb"]

        # ------------- BM25 (word + stem space)
        for tag, view in (("w", c_all), ("st", st_all)):
            cv2 = CountVectorizer(analyzer="word", min_df=1, token_pattern=r"\S+")
            C = cv2.fit_transform(view)
            ndoc = C.shape[0]
            dfq = np.asarray((C > 0).sum(0)).ravel()
            idf = np.log(1 + (ndoc - dfq + 0.5) / (dfq + 0.5))
            dl = np.asarray(C.sum(1)).ravel()
            avdl = dl.mean()
            k1, b = 1.5, 0.75
            Cc = C.tocsr().astype(np.float64)
            Ct = Cc.copy()
            # bm25 term weights per doc
            rows = np.repeat(np.arange(ndoc), np.diff(Cc.indptr))
            denom = Cc.data + k1 * (1 - b + b * dl[rows] / avdl)
            Ct.data = (Cc.data * (k1 + 1) / denom) * idf[Cc.indices]
            A, B = Ct[:n], Ct[n:]
            Ab = (Cc[:n] > 0).astype(np.float64)
            Bb = (Cc[n:] > 0).astype(np.float64)
            sab = np.asarray(A.multiply(Bb).sum(1)).ravel()
            sba = np.asarray(B.multiply(Ab).sum(1)).ravel()
            sa = np.asarray(A.sum(1)).ravel() + 1e-9
            sb = np.asarray(B.sum(1)).ravel() + 1e-9
            blocks.append(np.c_[sab, sba, sab / sa, sba / sb,
                                (sab + sba) / (sa + sb)])
            names += [f"bm25_{tag}_ab", f"bm25_{tag}_ba", f"bm25_{tag}_rab",
                      f"bm25_{tag}_rba", f"bm25_{tag}_sym"]

        # ------------- extra string / numeric features
        ex = []
        for i in range(n):
            a, b = str(s1[i]), str(s2[i])
            ca, cb = clean(a), clean(b)
            ja, jb = jamo(ca), jamo(cb)
            row = [difflib.SequenceMatcher(None, ja, jb).ratio()]
            row.append(difflib.SequenceMatcher(None, " ".join(stems(a)), " ".join(stems(b))).ratio())
            # numeric value comparison
            na = [float(x) for x in NUM.findall(a)][:4]
            nb = [float(x) for x in NUM.findall(b)][:4]
            if na and nb:
                d = min(abs(x - y) / max(abs(x), abs(y), 1.0) for x in na for y in nb)
                row += [d, 1.0]
            else:
                row += [0.0, 0.0]
            # high-idf (content) token overlap
            ta = sorted(set(ca.split()), key=lambda w: -idfw.get(w, default_idf))[:4]
            tb = sorted(set(cb.split()), key=lambda w: -idfw.get(w, default_idf))[:4]
            inter = len(set(ta) & set(tb))
            row += [inter, inter / max(1, min(len(ta), len(tb)))]
            # unmatched idf mass
            sa_ = set(ca.split()); sb_ = set(cb.split())
            ua = sum(idfw.get(w, default_idf) for w in sa_ - sb_)
            ub = sum(idfw.get(w, default_idf) for w in sb_ - sa_)
            tot = sum(idfw.get(w, default_idf) for w in sa_ | sb_) + 1e-9
            row += [ua / tot, ub / tot, (ua + ub) / tot, max(ua, ub) / tot]
            # unmatched idf mass in stem space
            sa2 = set(stems(a)); sb2 = set(stems(b))
            ua2 = sum(idfw.get(w, default_idf) for w in sa2 - sb2)
            ub2 = sum(idfw.get(w, default_idf) for w in sb2 - sa2)
            tot2 = sum(idfw.get(w, default_idf) for w in sa2 | sb2) + 1e-9
            row += [(ua2 + ub2) / tot2, max(ua2, ub2) / tot2]
            ex.append(row)
        blocks.append(np.array(ex, dtype=np.float64))
        names += ["sm_jamo", "sm_stem", "num_reldiff", "num_both", "top_int", "top_r",
                  "unm_a", "unm_b", "unm_sum", "unm_max", "unm_stem_sum", "unm_stem_max"]

        F = np.hstack(blocks)
        Fpair = np.hstack(svd_pair_blocks) if svd_pair_blocks else np.zeros((n, 0))
        assert F.shape[1] == len(names), (F.shape, len(names))
        self.feature_names = names
        self.log("features", F.shape, "pair-emb", Fpair.shape)
        return F, Fpair
