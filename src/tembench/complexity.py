from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd


@dataclass
class FitResult:
    model: str
    k: int  # number of parameters
    a: float  # intercept in log space
    b: float  # slope in log space
    aic: float
    n: int

    def predict(self, n_values: Iterable[float]) -> List[float]:
        # predict in original space: y = exp(a) * f(n)^b, but since we linearize as log y = a + b * log f(n)
        import math

        vals = []
        for n in n_values:
            vals.append(math.exp(self.a + self.b * 1.0 * 1.0 * 1.0) * 1.0)  # placeholder, replaced below
        return vals


def _basis_functions() -> Dict[str, Callable[[float], float]]:
    # positive n only
    return {
        "O(1)": lambda n: 1.0,
        "O(log n)": lambda n: log(max(n, 2)),
        "O(n)": lambda n: float(n),
        "O(n log n)": lambda n: float(n) * log(max(n, 2)),
        "O(n^2)": lambda n: float(n) ** 2,
        "O(n^3)": lambda n: float(n) ** 3,
    }


def _fit_single(x: List[float], y: List[float], basis: Callable[[float], float]) -> Tuple[float, float, float]:
    # Fit log y = a + b * log f(n) using simple least squares
    import math

    X = []
    Y = []
    for n, val in zip(x, y):
        fv = basis(n)
        if fv <= 0 or val <= 0:
            continue
        X.append(math.log(fv))
        Y.append(math.log(val))
    if len(X) < 2:
        return float("inf"), 0.0, 0.0
    # linear regression: Y = a + b X
    n = len(X)
    mean_x = sum(X) / n
    mean_y = sum(Y) / n
    sxx = sum((xi - mean_x) ** 2 for xi in X)
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(X, Y))
    if sxx == 0:
        return float("inf"), 0.0, 0.0
    b = sxy / sxx
    a = mean_y - b * mean_x
    # residuals
    rss = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(X, Y))
    # AIC for Gaussian errors with RSS (up to additive constant): AIC = n*log(RSS/n) + 2k
    k = 2  # a and b
    aic = n * math.log(rss / n) + 2 * k if rss > 0 else -float("inf")
    return aic, a, b


def fit_models(df: pd.DataFrame, x_col: str, y_col: str, by: List[str]) -> pd.DataFrame:
    """Fit candidate complexity models per group.

    Returns a DataFrame with columns: by..., model, a, b, aic, nobs
    """
    results = []
    bases = _basis_functions()
    for keys, group in df.groupby(by, dropna=False):
        x = group[x_col].tolist()
        y = group[y_col].tolist()
        best = (float("inf"), None, None, None)
        for name, fn in bases.items():
            aic, a, b = _fit_single(x, y, fn)
            if aic < best[0]:
                best = (aic, name, a, b)
        aic, name, a, b = best
        if name is None:
            continue
        rec = {}
        if isinstance(keys, tuple):
            for k, v in zip(by, keys):
                rec[k] = v
        else:
            rec[by[0]] = keys
        rec.update({"model": name, "a": a, "b": b, "aic": aic, "nobs": len(group)})
        results.append(rec)
    return pd.DataFrame(results)


def predict_series(df: pd.DataFrame, fits: pd.DataFrame, x_col: str, by: List[str]) -> pd.DataFrame:
    """Generate predictions per group across observed x using fitted model."""
    import math

    bases = _basis_functions()
    preds = []
    for _, row in fits.iterrows():
        key = {k: row[k] for k in by}
        model = row["model"]
        a = row["a"]
        b = row["b"]
        fn = bases[model]
        sub = df
        for k, v in key.items():
            sub = sub[sub[k] == v]
        # ensure unique sorted x
        xs = sorted(pd.unique(sub[x_col].values))
        for xv in xs:
            fv = fn(xv)
            if fv <= 0:
                continue
            yhat = math.exp(a + b * math.log(fv))
            rec = {**key, x_col: xv, "yhat": yhat, "model": model}
            preds.append(rec)
    return pd.DataFrame(preds)
