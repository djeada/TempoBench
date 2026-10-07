"""Public fit/predict API."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List

import pandas as pd

from .fitting import (
    _bootstrap_exponent_ci,
    _constant_only_without_outlier,
    _upper_bound_offset,
    _upper_bound_scale,
    _wls_fit,
)
from .formatting import _format_formula, _format_model_label
from .models import _MODEL_ORDER, _basis_functions
from .quality import assess_fit
from .selection import _rank_models, runner_up


@dataclass
class FitResult:
    """Result of fitting a complexity model to observed data.

    The model is: y = C·f(n) + baseline
    where f(n) is the basis function for the complexity class.  `fit_models`
    reports C and baseline already scaled up by the smallest factor that puts
    the curve over every point, and `offset` covers whatever scaling cannot
    (points the curve predicts as non-positive), so that
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
    count_col: str | None = None,
    spread_cols: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Fit Big-O complexity models per group.

    Algorithm:
    1. Select the complexity class (see `selection`): outlier-robust constant
       detection, log-log slope fallback for very short series, otherwise the
       lowest AIC of relative-error fits.
    2. Report y = C·f(n) + baseline from that same relative-error fit, so the
       curve is sane at small n as well as large.
    3. If C < 0 (only possible on the short-series path), fall back to a
       simpler model.
    4. Scale the curve up (and, where that cannot help, shift it by
       `offset`) until it sits over every point — a proper upper bound.

    Rows whose size or duration is missing or non-finite are ignored; a group
    left with fewer than two points is not fitted.

    Returns DataFrame: by…, model, display_model, C, baseline, offset, formula,
    rss, nobs, empirical_exponent, exponent_ci_low, exponent_ci_high,
    runner_up, model_margin, confidence, confidence_notes, caveats (the
    machine-readable codes behind confidence_notes, comma-separated)

    A class is always returned; `confidence` says whether the measurements could
    support it.  See `tembench.complexity.quality`.
    """
    if strategy not in {"heuristic", "strict"}:
        raise ValueError("strategy must be one of: heuristic, strict")

    results = []
    bases = _basis_functions()

    for keys, group in df.groupby(by, dropna=False):
        # A grid point with no successful trial stays in the summary so its
        # failure is visible, but it has no duration to fit.  An infinite
        # reading has none either, and would turn every coefficient into nan
        # while the quality checks still rated the fit high.
        xs_raw = pd.to_numeric(group[x_col], errors="coerce").astype(float)
        ys_raw = pd.to_numeric(group[y_col], errors="coerce").astype(float)
        finite = xs_raw.map(math.isfinite) & ys_raw.map(math.isfinite)
        group = group[finite]
        x = xs_raw[finite].tolist()
        y = ys_raw[finite].tolist()

        if len(x) < 2:
            continue

        # How many trials the thinnest point in this series was built from.
        min_samples = None
        if count_col and count_col in group.columns:
            counts = pd.to_numeric(group[count_col], errors="coerce").dropna()
            if not counts.empty:
                min_samples = float(counts.min())

        # The least repeatable point in the series, as a fraction of its median.
        max_relative_spread = None
        if spread_cols and all(c in group.columns for c in spread_cols):
            low = pd.to_numeric(group[spread_cols[0]], errors="coerce")
            high = pd.to_numeric(group[spread_cols[1]], errors="coerce")
            centre = pd.to_numeric(group[y_col], errors="coerce")
            relative = ((high - low) / centre.where(centre > 0)).dropna()
            if not relative.empty:
                max_relative_spread = float(relative.max())

        # Step 1: Select model based on growth pattern
        model, scores = _rank_models(x, y)

        # Step 2: the reported curve comes from the fit the class was scored on
        C, baseline, _ = _wls_fit(x, y, bases[model])

        # The scored path never picks a negative C; the short-series slope hint
        # can.  Walking down always terminates: O(1) fits C = 0.
        idx = _MODEL_ORDER.index(model)
        while C < 0 and idx > 0:
            idx -= 1
            model = _MODEL_ORDER[idx]
            C, baseline, _ = _wls_fit(x, y, bases[model])
        fn = bases[model]
        rss = sum((yi - (C * fn(xi) + baseline)) ** 2 for xi, yi in zip(x, y))

        # Resolved after the negative-coefficient fallback, so the margin always
        # describes the class that is actually being reported.
        rival, margin = runner_up(model, scores)

        # Step 3: lift the fitted curve into an upper bound
        fitted_C, fitted_baseline = C, baseline
        scale = _upper_bound_scale(x, y, fn, C, baseline)
        C, baseline = C * scale, baseline * scale
        offset = _upper_bound_offset(x, y, fn, C, baseline)
        # The exponent describes the work, so the fitted overhead comes off
        # first: a constant flattens the raw log-log slope (O(n²) plus startup
        # measures about n^0.5) and would make every class look simpler.
        net_y = y if model == "O(1)" else [yi - fitted_baseline for yi in y]
        empirical_exponent, exponent_ci_low, exponent_ci_high = _bootstrap_exponent_ci(
            x, net_y
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
        quality = assess_fit(
            x,
            y,
            effective_baseline=eff_baseline,
            exponent_ci_low=exponent_ci_low,
            exponent_ci_high=exponent_ci_high,
            model_margin=margin,
            model=model,
            relative_error=_relative_rms(x, y, fn, fitted_C, fitted_baseline),
            dropped_outlier=model == "O(1)" and _constant_only_without_outlier(y, x),
            min_samples=min_samples,
            max_relative_spread=max_relative_spread,
        )
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
                "runner_up": rival,
                "model_margin": margin,
                "confidence": quality.confidence,
                "confidence_notes": quality.summary,
                "caveats": ",".join(quality.notes),
            }
        )
        results.append(rec)

    return pd.DataFrame(results)


def _relative_rms(
    x: List[float], y: List[float], fn: Callable[[float], float], C: float, baseline: float
) -> float:
    """Typical relative miss of the fitted (unscaled) curve, over positive readings."""
    misses = [(yi - (C * fn(xi) + baseline)) / yi for xi, yi in zip(x, y) if yi > 0]
    return math.sqrt(sum(m * m for m in misses) / len(misses)) if misses else 0.0


def predict_series(
    df: pd.DataFrame, fits: pd.DataFrame, x_col: str, by: List[str]
) -> pd.DataFrame:
    """Generate upper-bound predictions per group.

    The curve is y = C·f(n) + baseline + offset, which guarantees the
    fit line sits at or above all measured data points.
    Samples 50 geometrically spaced sizes (linear when the range includes
    n ≤ 0) plus every observed size, for smooth rendering.
    """
    if df.empty or fits.empty:
        return pd.DataFrame()

    bases = _basis_functions()
    x_rows = []
    grouped = df.groupby(by, dropna=False) if by else [((), df)]
    for keys, group in grouped:
        numeric = pd.to_numeric(group[x_col], errors="coerce").astype(float)
        xs = sorted({v for v in numeric if math.isfinite(v)})
        if not xs:
            continue
        if len(xs) < 2:
            smooth_xs = xs
        else:
            x_min, x_max = xs[0], xs[-1]
            n_interp = 50
            # Sizes are usually swept geometrically and drawn on a log axis,
            # where linear steps leave the whole low end as one straight chord.
            if x_min > 0:
                ratio = (x_max / x_min) ** (1.0 / n_interp)
                grid = [x_min * ratio**i for i in range(n_interp)]
            else:
                step = (x_max - x_min) / n_interp
                grid = [x_min + i * step for i in range(n_interp)]
            # The observed sizes themselves are always sampled: the offset only
            # guarantees the bound at those points, so a chord between two
            # samples could otherwise pass beneath a measurement.
            smooth_xs = sorted(set(grid) | set(xs))

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
        fn = bases[str(model)]
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
        "confidence",
        "confidence_notes",
    ]:
        if col in preds.columns:
            keep_cols.append(col)
    return preds[keep_cols]
