"""Public fit/predict API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .fitting import _bootstrap_exponent_ci, _ols_fit, _upper_bound_offset
from .formatting import _format_formula, _format_model_label
from .models import _MODEL_ORDER, _basis_functions
from .selection import _select_model


@dataclass
class FitResult:
    """Result of fitting a complexity model to observed data.

    The model is: y = C·f(n) + baseline
    where f(n) is the basis function for the complexity class.
    The upper-bound curve is shifted up by `offset` so that
    y_bound = C·f(n) + baseline + offset ≥ y_i for all i.
    """

    model: str
    C: float
    baseline: float
    offset: float
    rss: float
    nobs: int

    @property
    def formula(self) -> str:
        return _format_formula(self.model, self.C, self.baseline + self.offset)

    def predict(self, n_values) -> List[float]:
        fn = _basis_functions()[self.model]
        b = self.baseline + self.offset
        return [self.C * fn(n) + b for n in n_values]


def fit_models(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    by: List[str],
    strategy: str = "heuristic",
) -> pd.DataFrame:
    """Fit Big-O complexity models per group.

    Algorithm:
    1. Select the complexity class via outlier-robust constant detection,
       log-log slope fallback (≤3 points), and step-up OLS with adaptive
       threshold (≥4 points).
    2. Fit y = C·f(n) + baseline via OLS for the selected model.
    3. If C < 0, fall back to a simpler model.
    4. Shift the curve up by `offset` to form a proper upper bound.

    Returns DataFrame: by…, model, display_model, C, baseline, offset, formula,
    rss, nobs, empirical_exponent, exponent_ci_low, exponent_ci_high
    """
    if strategy not in {"heuristic", "strict"}:
        raise ValueError("strategy must be one of: heuristic, strict")

    results = []
    bases = _basis_functions()

    for keys, group in df.groupby(by, dropna=False):
        x = group[x_col].astype(float).tolist()
        y = group[y_col].astype(float).tolist()

        if len(x) < 2:
            continue

        # Step 1: Select model based on growth pattern
        model = _select_model(x, y)

        # Step 2: Fit OLS for the selected model
        C, baseline, rss = _ols_fit(x, y, bases[model])

        # If C is negative, try simpler models
        if C < 0:
            idx = _MODEL_ORDER.index(model)
            while idx > 0 and C < 0:
                idx -= 1
                model = _MODEL_ORDER[idx]
                C, baseline, rss = _ols_fit(x, y, bases[model])
            if C < 0:
                model = "O(1)"
                C = 0.0
                baseline = max(y)
                rss = sum((yi - baseline) ** 2 for yi in y)

        # Step 3: Compute upper-bound offset
        offset = _upper_bound_offset(x, y, bases[model], C, baseline)
        empirical_exponent, exponent_ci_low, exponent_ci_high = _bootstrap_exponent_ci(
            x, y
        )
        display_model = _format_model_label(
            model,
            empirical_exponent,
            exponent_ci_low,
            exponent_ci_high,
            strategy=strategy,
        )

        rec = {}
        if isinstance(keys, tuple):
            for k, v in zip(by, keys):
                rec[k] = v
        else:
            rec[by[0]] = keys

        eff_baseline = baseline + offset
        rec.update(
            {
                "model": model,
                "display_model": display_model,
                "C": C,
                "baseline": baseline,
                "offset": offset,
                "formula": _format_formula(model, C, eff_baseline),
                "rss": rss,
                "nobs": len(group),
                "empirical_exponent": empirical_exponent,
                "exponent_ci_low": exponent_ci_low,
                "exponent_ci_high": exponent_ci_high,
            }
        )
        results.append(rec)

    return pd.DataFrame(results)


def predict_series(
    df: pd.DataFrame, fits: pd.DataFrame, x_col: str, by: List[str]
) -> pd.DataFrame:
    """Generate upper-bound predictions per group.

    The curve is y = C·f(n) + baseline + offset, which guarantees the
    fit line sits at or above all measured data points.
    Interpolates 50 points for smooth rendering.
    """
    if df.empty or fits.empty:
        return pd.DataFrame()

    bases = _basis_functions()
    x_rows = []
    grouped = df.groupby(by, dropna=False) if by else [((), df)]
    for keys, group in grouped:
        xs = sorted(pd.unique(group[x_col].astype(float).values))
        if not xs:
            continue
        if len(xs) < 2:
            smooth_xs = xs
        else:
            x_min, x_max = xs[0], xs[-1]
            n_interp = 50
            step = (x_max - x_min) / n_interp
            smooth_xs = [x_min + i * step for i in range(n_interp + 1)]

        key_values = keys if isinstance(keys, tuple) else (keys,)
        row_key = dict(zip(by, key_values)) if by else {}
        for xv in smooth_xs:
            x_rows.append({**row_key, x_col: xv})

    if not x_rows:
        return pd.DataFrame()

    x_grid = pd.DataFrame(x_rows)
    pred_df = x_grid.merge(fits, on=by, how="inner") if by else x_grid.merge(
        fits, how="cross"
    )

    pred_parts = []
    for model, part in pred_df.groupby("model", dropna=False):
        fn = bases[model]
        out = part.copy()
        x_vals = out[x_col].astype(float).map(fn)
        out["yhat"] = (
            out["C"].astype(float) * x_vals
            + out["baseline"].astype(float)
            + out["offset"].astype(float)
        )
        pred_parts.append(out)

    if not pred_parts:
        return pd.DataFrame()

    preds = pd.concat(pred_parts, ignore_index=True)
    keep_cols = list(by) + [x_col, "yhat", "model", "formula"]
    for col in [
        "display_model",
        "empirical_exponent",
        "exponent_ci_low",
        "exponent_ci_high",
    ]:
        if col in preds.columns:
            keep_cols.append(col)
    return preds[keep_cols]
