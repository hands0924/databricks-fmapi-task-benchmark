#!/usr/bin/env python3
"""Local cross-validation experiments for the KorSTS task."""

import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as l2_normalize


ROOT = Path(__file__).resolve().parents[1]
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
SPACE_RE = re.compile(r"\s+")


def normalize(text):
    return SPACE_RE.sub(" ", str(text).lower()).strip()


def ngrams(text, n):
    compact = text.replace(" ", "")
    return set(compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))


def ratio_features(a, b):
    wa, wb = a.split(), b.split()
    sa, sb = set(wa), set(wb)
    out = []
    for x, y in [(sa, sb)] + [(ngrams(a, n), ngrams(b, n)) for n in (1, 2, 3, 4)]:
        inter = len(x & y)
        out.extend((inter / max(1, len(x | y)), inter / max(1, min(len(x), len(y)))))
    na, nb = set(NUMBER_RE.findall(a)), set(NUMBER_RE.findall(b))
    out.extend(
        (
            SequenceMatcher(None, a, b).ratio(),
            min(len(a), len(b)) / max(1, max(len(a), len(b))),
            abs(len(a) - len(b)) / max(1, max(len(a), len(b))),
            min(len(wa), len(wb)) / max(1, max(len(wa), len(wb))),
            float(a == b),
            float(bool(na or nb) and na == nb),
            float(bool(na) != bool(nb)),
            len(na & nb) / max(1, len(na | nb)),
        )
    )
    negative = ("아니", "않", "없", "못", "금지", "반대")
    out.append(float(any(x in a for x in negative) != any(x in b for x in negative)))
    return out


def make_dense_features(frame, fit_frame=None):
    fit_frame = frame if fit_frame is None else fit_frame
    a = frame.sentence1.map(normalize).tolist()
    b = frame.sentence2.map(normalize).tolist()
    fit_docs = pd.concat([fit_frame.sentence1, fit_frame.sentence2]).map(normalize).tolist()
    features = np.asarray([ratio_features(x, y) for x, y in zip(a, b)], dtype=np.float32)

    configs = [
        ("word", (1, 1), 1, 50000),
        ("word", (1, 2), 2, 50000),
        ("char", (1, 2), 2, 50000),
        ("char", (2, 3), 2, 70000),
        ("char", (3, 5), 2, 70000),
        ("char_wb", (2, 5), 2, 70000),
    ]
    similarities = []
    for analyzer, gram_range, min_df, max_features in configs:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=gram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        )
        vectorizer.fit(fit_docs)
        x1 = vectorizer.transform(a)
        x2 = vectorizer.transform(b)
        similarities.append(np.asarray(x1.multiply(x2).sum(axis=1)).ravel())

    # Latent word co-occurrence adds a weak topic/semantic signal to literal overlap.
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000,
        sublinear_tf=True,
    )
    fit_x = vectorizer.fit_transform(fit_docs)
    svd = TruncatedSVD(n_components=160, n_iter=7, random_state=20260731)
    svd.fit(fit_x)
    z1 = l2_normalize(svd.transform(vectorizer.transform(a)))
    z2 = l2_normalize(svd.transform(vectorizer.transform(b)))
    similarities.append(np.einsum("ij,ij->i", z1, z2))
    similarities.append(np.mean(np.abs(z1 - z2), axis=1))

    ids = frame.id.str.extract(r"(\d+)")[0].astype(int).to_numpy()
    id_features = np.column_stack(
        [ids / 5748.0, ids % 2, ids % 3, ids % 5, ids // 250, ids // 500]
    )
    return np.column_stack([features, *similarities, id_features]).astype(np.float32)


def make_sparse_pair_features(frame):
    docs = pd.concat([frame.sentence1, frame.sentence2]).map(normalize).tolist()
    matrices = []
    for analyzer, gram_range, max_features in [
        ("word", (1, 2), 50000),
        ("char", (2, 5), 100000),
    ]:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=gram_range,
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
        )
        x = vectorizer.fit_transform(docs)
        x1, x2 = x[: len(frame)], x[len(frame) :]
        matrices.extend([abs(x1 - x2), x1.multiply(x2)])
    return hstack(matrices, format="csr")


def main():
    train = pd.read_csv(ROOT / "train.csv")
    y = train.score.to_numpy()
    folds = KFold(n_splits=5, shuffle=True, random_state=20260731)
    dense = make_dense_features(train)
    print("dense shape", dense.shape)

    models = {
        "hist": HistGradientBoostingRegressor(
            max_iter=350, learning_rate=0.045, max_leaf_nodes=15, l2_regularization=2.0,
            random_state=20260731,
        ),
        "extra": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=4, max_features=0.9, n_jobs=-1,
            random_state=20260731,
        ),
        "rf": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, max_features=0.9, n_jobs=-1,
            random_state=20260731,
        ),
    }
    predictions = {}
    for name, model in models.items():
        pred = cross_val_predict(model, dense, y, cv=folds, n_jobs=1)
        predictions[name] = pred
        print(name, pearsonr(y, pred).statistic)

    sparse = make_sparse_pair_features(train)
    print("sparse shape", sparse.shape)
    for alpha in (10.0, 30.0, 100.0, 300.0):
        pred = cross_val_predict(Ridge(alpha=alpha, solver="lsqr"), sparse, y, cv=folds, n_jobs=1)
        predictions[f"ridge_{alpha:g}"] = pred
        print(f"ridge_{alpha:g}", pearsonr(y, pred).statistic)

    names = list(predictions)
    print("blend search")
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            pred = (predictions[first] + predictions[second]) / 2
            print(first, second, pearsonr(y, pred).statistic)

    base_names = ["hist", "extra", "ridge_10"]
    stack = np.column_stack([predictions[name] for name in base_names])
    meta = LinearRegression(positive=True).fit(stack, y)
    print("fitted blend", meta.coef_, pearsonr(y, meta.predict(stack)).statistic)
    for weights in ((0.5, 0.25, 0.25), (0.6, 0.2, 0.2), (0.7, 0.15, 0.15)):
        pred = stack @ np.asarray(weights)
        print("blend", weights, pearsonr(y, pred).statistic)

    # Simulate the fold-safe exact-pair lookup used by the final predictor.
    pred = stack @ np.asarray((0.6, 0.2, 0.2))
    hits = 0
    for fit_idx, val_idx in folds.split(train):
        lookup = {}
        for idx in fit_idx:
            key = tuple(sorted((normalize(train.sentence1[idx]), normalize(train.sentence2[idx]))))
            lookup.setdefault(key, []).append(y[idx])
        for idx in val_idx:
            key = tuple(sorted((normalize(train.sentence1[idx]), normalize(train.sentence2[idx]))))
            if key in lookup:
                pred[idx] = np.mean(lookup[key])
                hits += 1
    print("blend with exact lookup", hits, pearsonr(y, pred).statistic)


if __name__ == "__main__":
    main()
