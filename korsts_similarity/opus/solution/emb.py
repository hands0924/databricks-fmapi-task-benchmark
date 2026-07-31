"""Corpus-derived word embeddings (PPMI + SVD) from train+test sentences.
Label-free: uses only sentence text and the pair structure (available at test time).
"""
import os, sys, time
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats as FT


def stems(s, k=3):
    return [t[:k] for t in FT.norm(s).split() if t]


class WordEmb:
    def __init__(self, dim=150, k=3, cross_w=1.0, min_count=2, power=0.5):
        self.dim, self.k, self.cross_w, self.min_count, self.power = dim, k, cross_w, min_count, power

    def fit(self, pairs):
        toks = [(stems(a, self.k), stems(b, self.k)) for a, b in pairs]
        from collections import Counter
        cnt = Counter()
        for a, b in toks:
            cnt.update(a); cnt.update(b)
        self.vocab = {w: i for i, w in enumerate(w for w, c in cnt.items() if c >= self.min_count)}
        V = len(self.vocab)
        rows, cols, vals = [], [], []

        def add(x, y, w):
            i = self.vocab.get(x); j = self.vocab.get(y)
            if i is None or j is None or i == j:
                return
            rows.append(i); cols.append(j); vals.append(w)
            rows.append(j); cols.append(i); vals.append(w)

        for a, b in toks:
            for sent in (a, b):
                u = list(dict.fromkeys(sent))
                for x in range(len(u)):
                    for y in range(x + 1, len(u)):
                        add(u[x], u[y], 1.0)
            ua = list(dict.fromkeys(a)); ub = list(dict.fromkeys(b))
            for x in ua:
                for y in ub:
                    add(x, y, self.cross_w)
        C = sp.coo_matrix((vals, (rows, cols)), shape=(V, V)).tocsr()
        # PPMI
        total = max(C.sum(), 1e-9)
        rowsum = np.maximum(np.asarray(C.sum(1)).ravel(), 1e-9)
        ctx = rowsum ** self.power
        ctx_total = max(ctx.sum(), 1e-9)
        Cc = C.tocoo()
        pj = ctx[Cc.col] / ctx_total
        pmi = np.log(np.maximum(Cc.data, 1e-12) / np.maximum(rowsum[Cc.row] * pj, 1e-12))
        pmi = np.nan_to_num(np.maximum(pmi, 0), nan=0.0, posinf=0.0, neginf=0.0)
        M = sp.coo_matrix((pmi, (Cc.row, Cc.col)), shape=(V, V)).tocsr()
        M.eliminate_zeros()
        svd = TruncatedSVD(n_components=min(self.dim, V - 1), random_state=0)
        E = svd.fit_transform(M)
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        self.E = E
        self.counts = np.array([cnt[w] for w in sorted(self.vocab, key=self.vocab.get)], dtype=float)
        return self

    def vecs(self, sent_stems, idf_map, default_idf):
        idx = []; w = []
        for t in sent_stems:
            i = self.vocab.get(t)
            if i is not None:
                idx.append(i); w.append(idf_map.get(t, default_idf))
        return (self.E[idx] if idx else np.zeros((0, self.E.shape[1]))), np.array(w)


def emb_features(pairs, we, idf_map, default_idf, tag):
    cols = [f"{tag}_{n}" for n in ["cos", "wcos", "alignmin", "alignmax", "alignhm",
                                   "alignmean", "cov", "l2", "maxmin", "cov90"]]
    out = []
    for a, b in pairs:
        sa, sb = stems(a, we.k), stems(b, we.k)
        Ea, wa = we.vecs(sa, idf_map, default_idf)
        Eb, wb = we.vecs(sb, idf_map, default_idf)
        if len(Ea) == 0 or len(Eb) == 0:
            out.append([0.0] * len(cols)); continue
        ma = Ea.mean(0); mb = Eb.mean(0)
        cos = float(ma @ mb / max(np.linalg.norm(ma) * np.linalg.norm(mb), 1e-9))
        va = (Ea * wa[:, None]).sum(0) / max(wa.sum(), 1e-9)
        vb = (Eb * wb[:, None]).sum(0) / max(wb.sum(), 1e-9)
        wcos = float(va @ vb / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-9))
        S = Ea @ Eb.T
        pa = S.max(1); pb = S.max(0)
        aa = float((pa * wa).sum() / max(wa.sum(), 1e-9))
        ab = float((pb * wb).sum() / max(wb.sum(), 1e-9))
        nva = normalize(va.reshape(1, -1))[0]; nvb = normalize(vb.reshape(1, -1))[0]
        out.append([
            cos, wcos, min(aa, ab), max(aa, ab),
            2 * aa * ab / max(aa + ab, 1e-9), (aa + ab) / 2,
            float((pa > 0.6).mean() + (pb > 0.6).mean()) / 2,
            float(np.linalg.norm(nva - nvb)),
            float(min(pa.min(), pb.min())),
            float(np.percentile(np.concatenate([pa, pb]), 10)),
        ])
    return pd.DataFrame(out, columns=cols)


def build_emb_feats(tr, te):
    pairs_tr = list(zip(tr.sentence1.astype(str), tr.sentence2.astype(str)))
    pairs_te = list(zip(te.sentence1.astype(str), te.sentence2.astype(str)))
    allp = pairs_tr + pairs_te
    from sklearn.feature_extraction.text import TfidfVectorizer
    Ftr = []; Fte = []
    for k, dim, cw, tag in [(3, 150, 1.0, "e3"), (2, 120, 1.0, "e2"), (3, 150, 0.0, "e3n")]:
        we = WordEmb(dim=dim, k=k, cross_w=cw).fit(allp)
        docs = [" ".join(stems(a, k)) for a, b in allp] + [" ".join(stems(b, k)) for a, b in allp]
        tv = TfidfVectorizer(analyzer='word', min_df=1).fit(docs)
        idf_map = {w: tv.idf_[i] for w, i in tv.vocabulary_.items()}
        dflt = float(tv.idf_.max())
        Ftr.append(emb_features(pairs_tr, we, idf_map, dflt, tag))
        Fte.append(emb_features(pairs_te, we, idf_map, dflt, tag))
        print(f"  {tag}: vocab={len(we.vocab)}")
    return pd.concat(Ftr, axis=1), pd.concat(Fte, axis=1)


if __name__ == '__main__':
    from scipy.stats import pearsonr
    t0 = time.time()
    tr = pd.read_csv('train.csv'); te = pd.read_csv('test.csv')
    Etr, Ete = build_emb_feats(tr, te)
    Etr.to_pickle('work/Etr.pkl'); Ete.to_pickle('work/Ete.pkl')
    y = tr.score.values
    for c in Etr.columns:
        print("  %-14s %.4f" % (c, pearsonr(Etr[c].values, y)[0]))
    print("%.0fs" % (time.time()-t0))
