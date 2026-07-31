"""Model zoo for t5_bike."""
import numpy as np
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.compose import ColumnTransformer


def make(name, seed=0):
    if name.startswith("hgb"):
        # hgb_L<leaf>_i<iter>_r<lr*100>
        p = dict(max_leaf_nodes=15, max_iter=800, learning_rate=0.05,
                 min_samples_leaf=30, l2_regularization=1.0)
        for tok in name.split("_")[1:]:
            if tok.startswith("L"):
                p["max_leaf_nodes"] = int(tok[1:])
            elif tok.startswith("i"):
                p["max_iter"] = int(tok[1:])
            elif tok.startswith("r"):
                p["learning_rate"] = int(tok[1:]) / 1000
            elif tok.startswith("m"):
                p["min_samples_leaf"] = int(tok[1:])
            elif tok.startswith("f"):
                p["max_features"] = int(tok[1:]) / 100
        return HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **p)
    if name == "et":
        return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                   max_features=0.6, n_jobs=4, random_state=seed)
    if name == "et_l5":
        return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                                   max_features=0.6, n_jobs=4, random_state=seed)
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     max_features=0.5, n_jobs=4, random_state=seed)
    if name == "gbr":
        return GradientBoostingRegressor(n_estimators=600, learning_rate=0.04,
                                         max_depth=5, subsample=0.8,
                                         min_samples_leaf=20, random_state=seed)
    if name == "spline":
        # smooth additive model; extrapolates linearly outside observed range
        return make_pipeline(
            SplineTransformer(n_knots=8, degree=3, extrapolation="linear"),
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    raise ValueError(name)
