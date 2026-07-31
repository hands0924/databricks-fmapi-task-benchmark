"""NB-weighted one-vs-rest logistic regression (NBSVM-style, Wang & Manning 2012)."""
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class NBLR(BaseEstimator, ClassifierMixin):
    """For each class c: reweight features by the NB log-count ratio r_c computed from the
    training fold, then fit a binary L2 logistic regression on x * r_c.
    Scores across classes are combined one-vs-rest.
    """

    def __init__(self, C=2.0, alpha=0.25, class_weight="balanced", max_iter=3000):
        self.C = C
        self.alpha = alpha              # Laplace smoothing for the count ratio
        self.class_weight = class_weight
        self.max_iter = max_iter

    def fit(self, X, y):
        X = sp.csr_matrix(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.r_, self.est_ = [], []
        for c in self.classes_:
            pos = y == c
            p = np.asarray(X[np.where(pos)[0]].sum(0)).ravel() + self.alpha
            q = np.asarray(X[np.where(~pos)[0]].sum(0)).ravel() + self.alpha
            r = np.log((p / p.sum()) / (q / q.sum()))
            est = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                     class_weight=self.class_weight)
            est.fit(X.multiply(r).tocsr(), pos.astype(int))
            self.r_.append(r)
            self.est_.append(est)
        return self

    def decision_function(self, X):
        X = sp.csr_matrix(X)
        return np.column_stack([e.decision_function(X.multiply(r).tocsr())
                                for e, r in zip(self.est_, self.r_)])

    def predict_proba(self, X):
        """OvR sigmoid, row-normalised (argmax-equivalent to decision_function)."""
        d = self.decision_function(X)
        s = 1.0 / (1.0 + np.exp(-np.clip(d, -30, 30)))
        return s / np.clip(s.sum(1, keepdims=True), 1e-12, None)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(1)]
