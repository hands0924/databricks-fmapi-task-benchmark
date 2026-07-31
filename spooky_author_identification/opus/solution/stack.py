"""Level-2 stacking on cached OOF predictions."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F, blend as B
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

SEED2 = 202


def meta_matrix(O, T, use_hand=False, logit=True):
    """(M,N,3) OOF + (M,Nt,3) test -> level-2 design matrices."""
    def enc(P):
        P = np.clip(P, 1e-7, 1 - 1e-7)
        if logit:
            V = np.log(P)
            V = V - V.mean(2, keepdims=True)      # center per row -> drop redundancy scale
        else:
            V = P
        return np.concatenate([V[i] for i in range(len(V))], axis=1)
    A, Bm = enc(O), enc(T)
    if use_hand:
        H = F.hand_feats(); n = A.shape[0]
        A = np.hstack([A, H[:n]]); Bm = np.hstack([Bm, H[n:]])
    return A, Bm


def cv_stack(A, y, make, nfold=5, seed=SEED2, Bm=None):
    skf = StratifiedKFold(nfold, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 3))
    tst = np.zeros((Bm.shape[0], 3)) if Bm is not None else None
    for ia, ib in skf.split(A, y):
        m = make()
        sc = StandardScaler().fit(A[ia])
        m.fit(sc.transform(A[ia]), y[ia])
        oof[ib] = m.predict_proba(sc.transform(A[ib]))
        if Bm is not None:
            tst += m.predict_proba(sc.transform(Bm)) / nfold
    return oof, tst, log_loss(y, oof)
