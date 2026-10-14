"""Regularized quadratic logistic surrogate for phase-region visualization.

The labels it is trained on are an exact arg-min over a dense (T, P) grid, so
the surrogate can only smooth -- never improve -- the boundaries. It is kept as
an optional, *scored* view: every fit reports how many grid points it disagrees
with, and misclassified points are drawn on the figure.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

#: default inverse regularization strength. C=0.1 erased ~30% of a small pocket
#: phase on a synthetic 10k-point grid; C=100 recovered ~98% of it.
DEFAULT_C = 100.0


def fit_phase_classifier(X, y, *, C: float = DEFAULT_C, degree: int = 2):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 2 or len(X) != len(y):
        raise ValueError("Expected X with shape (n, 2) and one label per row")
    if not np.isfinite(X).all():
        raise ValueError("Pressure and temperature must be finite")
    if len(np.unique(y)) < 2:
        raise ValueError("At least two distinct phases are required for classification")
    model = make_pipeline(
        StandardScaler(),                       # scale T and P first: better-conditioned polynomial
        PolynomialFeatures(degree=degree, include_bias=False),
        LogisticRegression(C=C, max_iter=10000),
    )
    return model.fit(X, y)


def surrogate_disagreement(model, X, y) -> float:
    """Fraction of grid points whose surrogate label differs from the exact label."""
    return float(np.mean(model.predict(np.asarray(X, dtype=float)) != np.asarray(y)))


def per_class_recall(model, X, y) -> dict:
    y = np.asarray(y)
    pred = model.predict(np.asarray(X, dtype=float))
    return {str(c): float(np.mean(pred[y == c] == c)) for c in np.unique(y)}
