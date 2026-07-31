import time, pickle, os
import numpy as np, pandas as pd
from scipy import sparse as sp
import features as F

os.makedirs('cache', exist_ok=True)
t0 = time.time()
tr = pd.read_csv('train.csv'); te = pd.read_csv('test.csv')
Xtr, Xte, names = F.build_sparse(tr, te)
print('sparse', Xtr.shape, Xtr.nnz, time.time() - t0, flush=True)
sp.save_npz('cache/Xtr.npz', Xtr); sp.save_npz('cache/Xte.npz', Xte)
with open('cache/names.pkl', 'wb') as f:
    pickle.dump(names, f)
Dtr, Dte = F.build_dense(tr, te)
print('dense', Dtr.shape, time.time() - t0, flush=True)
np.save('cache/Dtr.npy', Dtr); np.save('cache/Dte.npy', Dte)
np.save('cache/y.npy', tr.label.values)
print('done', time.time() - t0)
