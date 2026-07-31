"""Model zoo: compute OOF + test predictions for each base model, cached to disk."""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone

CLASSES = F.CLASSES
OOFDIR = 'work/oof'
os.makedirs(OOFDIR, exist_ok=True)
NFOLD = 5
SEED = 42

_tr, _te, Y = F.load()
NTR = len(_tr)


def folds():
    return list(StratifiedKFold(NFOLD, shuffle=True, random_state=SEED).split(np.zeros(NTR), Y))


class NBFeatures(BaseEstimator):
    """Naive-Bayes log-count-ratio reweighting (one-vs-rest, per class) -> NBSVM."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.rs_ = []
        for c in np.unique(y):
            p = np.asarray(X[y == c].sum(0)).ravel() + self.alpha
            q = np.asarray(X[y != c].sum(0)).ravel() + self.alpha
            r = np.log((p / p.sum()) / (q / q.sum()))
            self.rs_.append(sparse.diags(r))
        return self

    def transform(self, X):
        return sparse.hstack([X @ r for r in self.rs_]).tocsr()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class DenseWrap(BaseEstimator, ClassifierMixin):
    """Fit a dense-input estimator; handles scaling."""
    def __init__(self, est, scale=True):
        self.est = est; self.scale = scale

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.sc_ = StandardScaler().fit(X) if self.scale else None
        if self.sc_ is not None:
            X = self.sc_.transform(X)
        self.e_ = clone(self.est).fit(X, y)
        self.classes_ = self.e_.classes_
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.sc_ is not None:
            X = self.sc_.transform(X)
        return self.e_.predict_proba(X)


class SoftmaxDecision(BaseEstimator, ClassifierMixin):
    """Wrap a decision_function estimator into probabilities via temperature softmax
    fit on an internal holdout."""
    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        from scipy.optimize import minimize_scalar
        from scipy.special import softmax
        idx = np.arange(X.shape[0])
        rng = np.random.RandomState(0); rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        a, b = idx[:cut], idx[cut:]
        e0 = clone(self.est).fit(X[a], y[a])
        d = e0.decision_function(X[b])

        def f(logT):
            return log_loss(y[b], softmax(d * np.exp(-logT), axis=1), labels=np.unique(y))
        r = minimize_scalar(f, bounds=(-3, 3), method='bounded')
        self.logT_ = r.x
        self.e_ = clone(self.est).fit(X, y)
        self.classes_ = self.e_.classes_
        return self

    def predict_proba(self, X):
        from scipy.special import softmax
        return softmax(self.e_.decision_function(X) * np.exp(-self.logT_), axis=1)


# ------------------------------------------------------------------
def feat(name):
    """Assemble a feature matrix by name."""
    if name == 'word':
        return F.word_tfidf()
    if name == 'word1':
        return F.word_tfidf1()
    if name == 'char':
        return F.char_tfidf()
    if name == 'charfull':
        return F.char_full_tfidf()
    if name == 'pos':
        return F.pos_tfidf()
    if name == 'wc':
        return F.word_counts()
    if name == 'cc':
        return F.char_counts()
    if name == 'wc_bin':
        X = F.word_counts().copy(); X.data[:] = 1.0; return X
    if name == 'cc_bin':
        X = F.char_counts().copy(); X.data[:] = 1.0; return X
    if name == 'all':
        return sparse.hstack([F.word_tfidf(), F.char_tfidf(), F.char_full_tfidf()]).tocsr()
    if name == 'allpos':
        return sparse.hstack([F.word_tfidf(), F.char_tfidf(), F.char_full_tfidf(), F.pos_tfidf()]).tocsr()
    if name == 'wordchar':
        return sparse.hstack([F.word_tfidf(), F.char_tfidf()]).tocsr()
    if name == 'svd':
        return F.svd_feats()
    if name == 'svdhand':
        return np.hstack([F.svd_feats(), F.hand_feats()])
    if name == 'stopw':
        return F.stopw_tfidf()
    if name == 'charcase':
        return F.charcase_tfidf()
    if name == 'word3':
        return F.word3_tfidf()
    if name == 'hand':
        return F.hand_feats()
    raise KeyError(name)


def run(name, model_fn, featname, force=False, verbose=True):
    """Compute OOF and test predictions for one model spec."""
    fo = f'{OOFDIR}/{name}_oof.npy'; ft = f'{OOFDIR}/{name}_test.npy'
    if os.path.exists(fo) and not force:
        oof, tst = np.load(fo), np.load(ft)
        if verbose:
            print(f'{name:24s} {log_loss(Y, oof):.5f}  (cached)', flush=True)
        return oof, tst
    t0 = time.time()
    X = feat(featname)
    Xtr, Xte = X[:NTR], X[NTR:]
    oof = np.zeros((NTR, 3))
    tst = np.zeros((Xte.shape[0], 3))
    for k, (ia, ib) in enumerate(folds()):
        m = model_fn()
        m.fit(Xtr[ia], Y[ia])
        oof[ib] = m.predict_proba(Xtr[ib])
        tst += m.predict_proba(Xte) / NFOLD
    np.save(fo, oof); np.save(ft, tst)
    if verbose:
        print(f'{name:24s} {log_loss(Y, oof):.5f}  ({time.time()-t0:.0f}s)', flush=True)
    return oof, tst
