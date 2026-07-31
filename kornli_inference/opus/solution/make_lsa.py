"""LSA sentence embeddings in a shared space -> pair interaction features."""
import time, sys
import numpy as np, pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
sys.path.insert(0, 'solution')
import features as F

t0 = time.time()
K = int(sys.argv[1]) if len(sys.argv) > 1 else 200
tr = pd.read_csv('train.csv'); te = pd.read_csv('test.csv')
s1 = pd.concat([tr.sentence1, te.sentence1]).tolist()
s2 = pd.concat([tr.sentence2, te.sentence2]).tolist()
ntr = len(tr)

vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=3,
                      sublinear_tf=True, max_features=300000)
vec.fit(s1 + s2)
A = vec.transform(s1); B = vec.transform(s2)
vec2 = TfidfVectorizer(analyzer='word', tokenizer=F.stem_tokens, min_df=2,
                       sublinear_tf=True, lowercase=False)
vec2.fit(s1 + s2)
A2 = vec2.transform(s1); B2 = vec2.transform(s2)
A = sp.hstack([A, A2], format='csr'); B = sp.hstack([B, B2], format='csr')
print('tfidf', A.shape, time.time() - t0, flush=True)

svd = TruncatedSVD(n_components=K, random_state=0, n_iter=5)
svd.fit(sp.vstack([A, B]))
U = normalize(svd.transform(A)).astype(np.float32)
V = normalize(svd.transform(B)).astype(np.float32)
print('svd', svd.explained_variance_ratio_.sum(), time.time() - t0, flush=True)

Z = np.hstack([U * V, np.abs(U - V), U, V,
               (U * V).sum(1, keepdims=True)]).astype(np.float32)
np.save('cache/Ltr.npy', Z[:ntr]); np.save('cache/Lte.npy', Z[ntr:])
print('lsa', Z.shape, time.time() - t0)
