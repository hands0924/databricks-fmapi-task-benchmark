"""Soft token-alignment features from LSA term embeddings (no external data).

Term embeddings come from an SVD of the term-document matrix over the task's own
sentences.  For each premise/hypothesis pair we compute, for every hypothesis
token, its best cosine match among premise tokens, and aggregate those scores
(IDF weighted).  Low coverage -> neutral/contradiction, high coverage ->
entailment; unmatched high-IDF tokens are a strong contradiction/neutral cue.
"""
import sys, time
import numpy as np, pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
sys.path.insert(0, 'solution')
import features as F

t0 = time.time()
tr = pd.read_csv('train.csv'); te = pd.read_csv('test.csv')
ntr = len(tr)
s1 = pd.concat([tr.sentence1, te.sentence1]).tolist()
s2 = pd.concat([tr.sentence2, te.sentence2]).tolist()

vec = TfidfVectorizer(analyzer='word', tokenizer=F.stem_tokens, min_df=2,
                      sublinear_tf=True, lowercase=False)
M = vec.fit_transform(s1 + s2)               # sentences x vocab
vocab = vec.vocabulary_
idf = vec.idf_
svd = TruncatedSVD(n_components=128, random_state=0, n_iter=6)
W = svd.fit_transform(M.T.tocsr())           # vocab x k  (term embeddings)
W = normalize(W).astype(np.float32)
print('term emb', W.shape, time.time() - t0, flush=True)

MED = float(np.median(idf))


def ids(s):
    out = []
    for t in F.stem_tokens(s):
        j = vocab.get(t)
        if j is not None:
            out.append(j)
    return out


def pair_feats(a, b):
    """a: premise token ids, b: hypothesis token ids"""
    if not a or not b:
        return [0.0] * 12
    Wa = W[a]; Wb = W[b]
    S = Wb @ Wa.T                            # |b| x |a|
    best = S.max(1)
    wb = np.array([idf[j] for j in b], dtype=np.float32)
    wsum = wb.sum() + 1e-6
    aset = set(a)
    exact = np.array([1.0 if j in aset else 0.0 for j in b], dtype=np.float32)
    soft = np.where(exact > 0, 1.0, best)
    # weighted coverage on the premise side too
    bestp = S.max(0)
    wa = np.array([idf[j] for j in a], dtype=np.float32)
    return [
        float(best.mean()), float(best.min()), float(np.median(best)),
        float((best * wb).sum() / wsum),
        float((soft * wb).sum() / wsum),
        float(((soft < 0.4) * wb).sum() / wsum),
        float(((soft < 0.6) * wb).sum() / wsum),
        float(wb[soft < 0.5].max(initial=0.0)),
        float((exact * wb).sum() / wsum),
        float(bestp.mean()), float((bestp * wa).sum() / (wa.sum() + 1e-6)),
        float(S.mean()),
    ]


rows = []
A1 = [ids(s) for s in s1]
B1 = [ids(s) for s in s2]
print('tokenized', time.time() - t0, flush=True)
for a, b in zip(A1, B1):
    rows.append(pair_feats(a, b))
Z = np.asarray(rows, dtype=np.float32)
np.save('cache/Atr.npy', Z[:ntr]); np.save('cache/Ate.npy', Z[ntr:])
print('align', Z.shape, time.time() - t0)
