"""End-to-end: base models -> OOF blend weights -> premise-group constraint ->
outputs/submission.csv.

Run from the task directory after make_cache.py / make_lsa.py / make_align.py:
    python solution/run_final.py
"""
import pickle, sys, time
from collections import Counter
import numpy as np, pandas as pd
from scipy import sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

sys.path.insert(0, 'solution')
from constraint import apply_constraint, known_from, CLASSES

NFOLD = 5
SEEDS = (0, 1, 2)
ALPHA = 3e-7
DSCALE = 0.3
t0 = time.time()

tr = pd.read_csv('train.csv'); te = pd.read_csv('test.csv')
y = np.load('cache/y.npy', allow_pickle=True)
names = pickle.load(open('cache/names.pkl', 'rb'))
sizes = [int(n.split(':')[1]) for n in names]; offs = np.cumsum([0] + sizes)
B = {n.split(':')[0]: (offs[i], offs[i + 1]) for i, n in enumerate(names)}
COLS = ['char_ovl', 'char_nov', 'word_ovl', 'word_nov', 'stem_ovl', 'stem_nov']

Dtr = np.load('cache/Dtr.npy'); Dte = np.load('cache/Dte.npy')
Atr = np.load('cache/Atr.npy'); Ate = np.load('cache/Ate.npy')
Ltr = np.load('cache/Ltr.npy'); Lte = np.load('cache/Lte.npy')
Gtr = np.hstack([Dtr, Atr]); Gte = np.hstack([Dte, Ate])
qt = QuantileTransformer(output_distribution='normal', subsample=20000, random_state=0)
qt.fit(np.vstack([Gtr, Gte]))
Qtr = qt.transform(Gtr).astype(np.float32); Qte = qt.transform(Gte).astype(np.float32)

Xs_tr = sp.load_npz('cache/Xtr.npz'); Xs_te = sp.load_npz('cache/Xte.npz')
Xtr = sp.hstack([Xs_tr[:, B[c][0]:B[c][1]] for c in COLS] +
                [sp.csr_matrix(Qtr * DSCALE)], format='csr').astype(np.float32)
Xte = sp.hstack([Xs_te[:, B[c][0]:B[c][1]] for c in COLS] +
                [sp.csr_matrix(Qte * DSCALE)], format='csr').astype(np.float32)
del Xs_tr, Xs_te
ss = StandardScaler().fit(np.vstack([np.hstack([Qtr, Ltr]), np.hstack([Qte, Lte])]))
Mtr = ss.transform(np.hstack([Qtr, Ltr])).astype(np.float32)
Mte = ss.transform(np.hstack([Qte, Lte])).astype(np.float32)
print('matrices', Xtr.shape, Gtr.shape, Mtr.shape, '%.0fs' % (time.time() - t0), flush=True)


def fit_lin(Xa, ya, Xb, Xc):
    """averaged multi-seed SGD logistic model"""
    pb = np.zeros((Xb.shape[0], 3)); pc = np.zeros((Xc.shape[0], 3))
    for s in SEEDS:
        m = SGDClassifier(loss='log_loss', alpha=ALPHA, max_iter=40, tol=None,
                          average=True, random_state=s)
        m.fit(Xa, ya)
        assert list(m.classes_) == list(CLASSES)
        pb += m.predict_proba(Xb); pc += m.predict_proba(Xc)
    return pb / len(SEEDS), pc / len(SEEDS)


def fit_gb(Xa, ya, Xb, Xc):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
    m.fit(Xa, ya)
    assert list(m.classes_) == list(CLASSES)
    return m.predict_proba(Xb), m.predict_proba(Xc)


def fit_mlp(Xa, ya, Xb, Xc):
    m = MLPClassifier(hidden_layer_sizes=(384, 128), alpha=3e-4, batch_size=256,
                      learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                      n_iter_no_change=5, random_state=0)
    m.fit(Xa, ya)
    assert list(m.classes_) == list(CLASSES)
    return m.predict_proba(Xb), m.predict_proba(Xc)


MODELS = [('lin', fit_lin, Xtr, Xte), ('gb', fit_gb, Gtr, Gte), ('mlp', fit_mlp, Mtr, Mte)]
n = len(y)
oof = {k: np.zeros((n, 3)) for k, *_ in MODELS}
tep = {k: np.zeros((len(te), 3)) for k, *_ in MODELS}
skf = StratifiedKFold(NFOLD, shuffle=True, random_state=42)
for k, (ia, ib) in enumerate(skf.split(np.zeros(n), y)):
    for name, fn, Xa, Xb in MODELS:
        pv, pt = fn(Xa[ia], y[ia], Xa[ib], Xb)
        oof[name][ib] = pv
        tep[name] += pt / NFOLD
    print('fold', k, {nm: round(float((CLASSES[oof[nm][ib].argmax(1)] == y[ib]).mean()), 4)
                      for nm, *_ in MODELS}, '%.0fs' % (time.time() - t0), flush=True)

acc = lambda P: float((CLASSES[P.argmax(1)] == y).mean())
for nm, *_ in MODELS:
    print('oof %s %.4f' % (nm, acc(oof[nm])))
OO = np.nan_to_num(np.stack([oof[nm] for nm, *_ in MODELS]), nan=1 / 3)
TT = np.nan_to_num(np.stack([tep[nm] for nm, *_ in MODELS]), nan=1 / 3)
OO /= OO.sum(2, keepdims=True); TT /= TT.sum(2, keepdims=True)
np.save('cache/oof_all.npy', OO); np.save('cache/te_all.npy', TT)

print('base model probabilities cached; run blend_submit.py next (%.0fs)' % (time.time() - t0))
