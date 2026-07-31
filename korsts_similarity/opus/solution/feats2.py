"""Second-generation features: BM25, soft-cosine, alignment coverage with idf
mass, edit-opcode statistics, word-order."""
import os, sys, math, difflib
import numpy as np, pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats as FT

n_ = FT.norm
jm = FT.to_jamo


def toks(s, k=None):
    t = n_(s).split()
    return [x[:k] if k and len(x) > k else x for x in t]


class Extra:
    def __init__(self, corpus):
        self.corpus = [n_(s) for s in corpus]
        # idf over stem-2 and full tokens
        self.idf = {}
        for k, tag in [(None, 'f'), (2, 's2'), (3, 's3')]:
            docs = [" ".join(toks(s, k)) for s in self.corpus]
            tv = TfidfVectorizer(analyzer='word', min_df=1).fit(docs)
            self.idf[tag] = ({w: tv.idf_[i] for w, i in tv.vocabulary_.items()},
                             float(tv.idf_.max()))
        # avg doc length for BM25
        self.avgdl = {tag: np.mean([len(toks(s, k)) for s in self.corpus])
                      for k, tag in [(None, 'f'), (2, 's2'), (3, 's3')]}
        # token-level jamo char-ngram vectors for soft similarity
        vocab = sorted({t for s in self.corpus for t in s.split()})
        self.tvec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1,
                                    sublinear_tf=True, dtype=np.float32)
        self.tvec.fit([jm(t) for t in vocab])

    def tmat(self, tokens):
        if not tokens:
            return None
        return normalize(self.tvec.transform([jm(t) for t in tokens]))

    def bm25(self, q, d, tag, k1=1.5, b=0.75):
        idf, dflt = self.idf[tag]
        dc = Counter(d); dl = len(d)
        avgdl = self.avgdl[tag]
        sc = 0.0; norm_ = 0.0
        for t in set(q):
            w = idf.get(t, dflt)
            norm_ += w
            f = dc.get(t, 0)
            if f:
                sc += w * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / max(avgdl, 1e-9)))
        return sc / max(norm_, 1e-9)

    def row(self, a, b):
        f = []
        ta, tb = toks(a), toks(b)
        sa, sb = toks(a, 2), toks(b, 2)
        s3a, s3b = toks(a, 3), toks(b, 3)
        for (qa, qb, tag) in [(ta, tb, 'f'), (sa, sb, 's2'), (s3a, s3b, 's3')]:
            x = self.bm25(qa, qb, tag); yv = self.bm25(qb, qa, tag)
            f += [x, yv, min(x, yv), max(x, yv), 2 * x * yv / max(x + yv, 1e-9)]
        # ---- soft alignment with idf mass ----
        idf, dflt = self.idf['f']
        A = self.tmat(ta); B = self.tmat(tb)
        if A is None or B is None:
            f += [0.0] * 26
        else:
            S = (A @ B.T).toarray()
            wa = np.array([idf.get(t, dflt) for t in ta])
            wb = np.array([idf.get(t, dflt) for t in tb])
            pa = S.max(1); pb = S.max(0)
            for th in (0.9, 0.75, 0.6, 0.45):
                ca = float((wa * (pa > th)).sum() / max(wa.sum(), 1e-9))
                cb = float((wb * (pb > th)).sum() / max(wb.sum(), 1e-9))
                f += [min(ca, cb), max(ca, cb), 2 * ca * cb / max(ca + cb, 1e-9)]
            # unmatched idf mass (absolute + relative)
            um_a = float((wa * (1 - pa)).sum()); um_b = float((wb * (1 - pb)).sum())
            f += [um_a, um_b, min(um_a, um_b), max(um_a, um_b), um_a + um_b]
            # hardest unmatched token weight
            f += [float((wa * (pa < 0.4)).max() if len(wa) else 0.0),
                  float((wb * (pb < 0.4)).max() if len(wb) else 0.0)]
            # soft cosine measure
            num = float(wa @ S @ wb)
            da = float(wa @ (A @ A.T).toarray() @ wa)
            db = float(wb @ (B @ B.T).toarray() @ wb)
            f.append(num / max(math.sqrt(da * db), 1e-9))
            # word order via best-match positions
            good = pa > 0.6
            if good.sum() >= 3:
                pos_a = np.arange(len(ta))[good]
                pos_b = S.argmax(1)[good]
                from scipy.stats import spearmanr
                rho = spearmanr(pos_a, pos_b).statistic
                f += [0.0 if np.isnan(rho) else float(rho), float(good.mean())]
            else:
                f += [0.0, float(good.mean())]
            # count of strong matches, normalized
            f += [float((pa > 0.85).sum()), float((pa > 0.85).mean()),
                  float((pb > 0.85).mean()),
                  float(np.mean(pa) * np.mean(pb))]
        # ---- edit opcode statistics ----
        na, nb = n_(a), n_(b)
        for x, yv in ((na, nb), (jm(na), jm(nb))):
            sm = difflib.SequenceMatcher(None, x, yv, autojunk=False)
            eq = rep = ins = dele = 0
            nblocks = 0
            for tagop, i1, i2, j1, j2 in sm.get_opcodes():
                if tagop == 'equal':
                    eq += i2 - i1; nblocks += 1
                elif tagop == 'replace':
                    rep += max(i2 - i1, j2 - j1)
                elif tagop == 'insert':
                    ins += j2 - j1
                else:
                    dele += i2 - i1
            tot = max(len(x) + len(yv), 1)
            f += [eq / tot, rep / tot, ins / tot, dele / tot, float(nblocks),
                  eq / max(min(len(x), len(yv)), 1)]
        return f


def build(tr, te):
    corpus = pd.concat([tr.sentence1, tr.sentence2, te.sentence1, te.sentence2]).astype(str).tolist()
    ex = Extra(corpus)
    out = []
    for df in (tr, te):
        rows = [ex.row(a, b) for a, b in zip(df.sentence1.astype(str), df.sentence2.astype(str))]
        M = np.array(rows, dtype=np.float64)
        out.append(pd.DataFrame(M, columns=[f"x{i}" for i in range(M.shape[1])]))
    return out[0], out[1]


if __name__ == '__main__':
    import time
    from scipy.stats import pearsonr
    t0 = time.time()
    tr = pd.read_csv('train.csv'); te = pd.read_csv('test.csv')
    Xtr, Xte = build(tr, te)
    Xtr.to_pickle('work/Xtr.pkl'); Xte.to_pickle('work/Xte.pkl')
    y = tr.score.values
    cors = sorted(((abs(pearsonr(Xtr[c], y)[0]) if Xtr[c].std() > 0 else 0, c) for c in Xtr.columns), reverse=True)
    print("dim", Xtr.shape)
    for v, c in cors[:15]: print("  %-6s %.4f" % (c, v))
    print("%.0fs" % (time.time() - t0))
