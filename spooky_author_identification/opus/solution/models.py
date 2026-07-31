"""Definition of every level-1 base model used in the final ensemble.

Each entry is (name, feature_block, factory). `name` must match the cache key so
runs are resumable. Costs are noted where relevant; the whole zoo takes roughly
50-70 CPU-minutes on 4 cores.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from zoo import NBFeatures, SoftmaxDecision, DenseWrap


class NBLR:
    """NBSVM-style: Naive-Bayes log-count-ratio reweighting + logistic regression."""

    def __init__(self, C=1.0, alpha=1.0):
        self.C = C
        self.alpha = alpha

    def fit(self, X, y):
        self.nb_ = NBFeatures(self.alpha).fit(X, y)
        self.m_ = LogisticRegression(C=self.C, solver='liblinear').fit(self.nb_.transform(X), y)
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(self.nb_.transform(X))


def specs():
    S = []
    A = S.append

    # ---- Multinomial NB (very strong on sublinear TF-IDF with tiny alpha) ----
    for fn in ['word', 'word1', 'char', 'charfull', 'pos', 'all', 'wordchar']:
        for a in [0.003, 0.01, 0.02, 0.05]:
            A((f'mnb_{fn}_a{a}', fn, lambda a=a: MultinomialNB(alpha=a)))
    for fn in ['wc', 'cc', 'word', 'char', 'wc_bin', 'cc_bin', 'pos']:
        for a in [0.03, 0.1, 0.3, 1.0]:
            A((f'mnb_{fn}_a{a}', fn, lambda a=a: MultinomialNB(alpha=a)))

    # ---- Complement / Bernoulli NB ----
    for fn in ['wc', 'cc', 'word', 'char']:
        for a in [0.1, 0.3, 1.0]:
            A((f'cnb_{fn}_a{a}', fn, lambda a=a: ComplementNB(alpha=a)))
    for fn in ['wc_bin', 'cc_bin']:
        for a in [0.03, 0.1, 0.3]:
            A((f'bnb_{fn}_a{a}', fn, lambda a=a: BernoulliNB(alpha=a)))

    # ---- Logistic regression on TF-IDF blocks (needs large C on sublinear TF-IDF) ----
    for fn in ['word', 'char', 'charfull', 'all']:
        for C in [1, 4, 12, 30, 80]:
            A((f'lr_{fn}_C{C}', fn, lambda C=C: LogisticRegression(C=C, solver='liblinear')))
    for C in [1, 4, 12]:
        A((f'lr_pos_C{C}', 'pos', lambda C=C: LogisticRegression(C=C, solver='liblinear')))
    for fn in ['allpos', 'wordchar']:
        for C in [30, 80]:
            A((f'lr_{fn}_C{C}', fn, lambda C=C: LogisticRegression(C=C, solver='liblinear')))
    A(('lrmn_word_C1', 'word', lambda: LogisticRegression(C=1, solver='lbfgs', max_iter=800)))

    # ---- NBSVM ----
    for fn in ['wc_bin', 'wc', 'word']:
        for C in [0.1, 0.5, 2.0]:
            A((f'nblr_{fn}_C{C}', fn, lambda C=C: NBLR(C=C, alpha=1.0)))

    # ---- Linear SVM turned into probabilities via temperature-scaled softmax ----
    for fn in ['all', 'char', 'word', 'charfull']:
        for C in [0.3, 1.0]:
            A((f'svc_{fn}_C{C}', fn, lambda C=C: SoftmaxDecision(LinearSVC(C=C, max_iter=3000))))

    # ---- extra stylometric views: function-word stream, case-sensitive chars, 1-3 grams ----
    for fn in ['stopw', 'charcase', 'word3']:
        for a in [0.003, 0.01, 0.02, 0.05, 0.1]:
            A((f'mnb_{fn}_a{a}', fn, lambda a=a: MultinomialNB(alpha=a)))
        for C in [4, 12, 30]:
            A((f'lr_{fn}_C{C}', fn, lambda C=C: LogisticRegression(C=C, solver='liblinear')))
        A((f'svc_{fn}_C0.5', fn, lambda: SoftmaxDecision(LinearSVC(C=0.5, max_iter=3000))))

    # ---- one dense non-linear member on 180 TruncatedSVD components (weak alone,
    #      but adds a little diversity to the stack) ----
    A(('mlp_svd_h256', 'svd', lambda: DenseWrap(MLPClassifier(
        (256,), alpha=1e-4, max_iter=60, early_stopping=True, n_iter_no_change=6,
        random_state=0, learning_rate_init=3e-3))))

    return S
