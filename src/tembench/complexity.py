from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import pandas as pd


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


# ---------------------------------------------------------------------------
# Basis functions  (each represents a growth class, simplest first)
# ---------------------------------------------------------------------------

_MODEL_ORDER = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)", "O(n³)"]

_ALL_MODELS: List[Tuple[str, Callable[[float], float]]] = [
    ("O(1)", lambda n: 1.0),
    ("O(log n)", lambda n: math.log(max(n, 2))),
    ("O(n)", lambda n: float(n)),
    ("O(n log n)", lambda n: float(n) * math.log(max(n, 2))),
    ("O(n²)", lambda n: float(n) ** 2),
    ("O(n³)", lambda n: float(n) ** 3),
]


def _basis_functions() -> Dict[str, Callable[[float], float]]:
    return dict(_ALL_MODELS)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_coeff(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    if abs(v) >= 0.001:
        return f"{v:.4g}"
    return f"{v:.3e}"


_BASIS_STR = {
    "O(1)": "1",
    "O(log n)": "log(n)",
    "O(n)": "n",
    "O(n log n)": "n·log(n)",
    "O(n²)": "n²",
    "O(n³)": "n³",
}


def _format_formula(model: str, C: float, effective_baseline: float = 0.0) -> str:
    """Render e.g. 'T(n) ≤ 3.21e-04·n·log(n) + 12.5'."""
    b = _BASIS_STR.get(model, "?")
    c_str = _format_coeff(C)

    if b == "1":
        total = C + effective_baseline
        return f"T(n) ≤ {_format_coeff(total)}"

    term = f"{c_str}·{b}"
    if abs(effective_baseline) < 1e-9:
        return f"T(n) ≤ {term}"
    sign = "+" if effective_baseline >= 0 else "−"
    return f"T(n) ≤ {term} {sign} {_format_coeff(abs(effective_baseline))}"


# ---------------------------------------------------------------------------
# OLS fitting
# ---------------------------------------------------------------------------


def _ols_fit(
    x: List[float], y: List[float], basis: Callable[[float], float]
) -> tuple[float, float, float]:
    """Fit y = C·f(n) + baseline via OLS.  Returns (C, baseline, rss)."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0, float("inf")

    F = [basis(xi) for xi in x]
    sum_f = sum(F)
    sum_y = sum(y)
    sum_ff = sum(fi * fi for fi in F)
    sum_fy = sum(fi * yi for fi, yi in zip(F, y))

    denom = n * sum_ff - sum_f * sum_f
    if abs(denom) < 1e-30:
        return 0.0, sum_y / n, float("inf")

    C = (n * sum_fy - sum_f * sum_y) / denom
    baseline = (sum_y - C * sum_f) / n
    rss = sum((yi - (C * fi + baseline)) ** 2 for fi, yi in zip(F, y))
    return C, baseline, rss


# ---------------------------------------------------------------------------
# Model selection  (step-up with outlier detection and log-log fallback)
# ---------------------------------------------------------------------------

# Coefficient of variation below which data is classified as O(1).
_CV_CONST = 0.08


def _is_effectively_constant(y: List[float]) -> bool:
    """Check if data is constant, using outlier-robust analysis for n≥4.

    For ≥4 points, also checks whether removing any single point brings
    the CV below the threshold.  This detects "flat + one outlier" data
    that would otherwise be misclassified as polynomial.
    """
    n = len(y)
    y_mean = sum(y) / n
    if abs(y_mean) < 1e-30:
        return all(abs(yi) < 1e-30 for yi in y)

    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    cv = math.sqrt(ss_tot / n) / abs(y_mean)
    if cv < _CV_CONST:
        return True

    if n >= 4:
        for skip in range(n):
            subset = [yi for i, yi in enumerate(y) if i != skip]
            m = sum(subset) / len(subset)
            if abs(m) < 1e-30:
                continue
            ss = sum((yi - m) ** 2 for yi in subset)
            if math.sqrt(ss / len(subset)) / abs(m) < _CV_CONST:
                return True
    return False


def _log_log_slope(x: List[float], y: List[float]) -> float:
    """Compute empirical exponent from log-log linear regression."""
    pairs = sorted(zip(x, y))
    lx = [math.log(p[0]) for p in pairs]
    ly = [math.log(p[1]) for p in pairs]
    n = len(lx)
    sx, sy = sum(lx), sum(ly)
    sxx = sum(a * a for a in lx)
    sxy = sum(a * b for a, b in zip(lx, ly))
    d = n * sxx - sx * sx
    return (n * sxy - sx * sy) / d if abs(d) > 1e-30 else 0.0


def _slope_to_model(slope: float) -> str:
    """Map a log-log slope to the nearest complexity class."""
    if slope < 0.15:
        return "O(log n)"
    if slope < 0.75:
        return "O(n)" if slope > 0.5 else "O(log n)"
    if slope < 1.25:
        return "O(n)"
    if slope < 1.75:
        return "O(n log n)"
    if slope < 2.5:
        return "O(n²)"
    return "O(n³)"


def _tail_ratio_favors_linear(x: List[float], y: List[float]) -> bool:
    """Check whether the growth ratio at the largest n pair is closer to O(n)
    than to O(n log n).

    Uses the last pair of data points (largest n) where subprocess overhead
    is minimal relative to computation.  If the observed y ratio is at least
    as close to the pure O(n) prediction as to O(n log n), returns True.

    Only applied with ≥ 4 data points — with 3 points, baseline overhead
    can still distort the tail ratio.
    """
    if len(x) < 4 or x[-1] <= x[-2] or x[-2] <= 0 or y[-2] <= 0:
        return False
    n_ratio = x[-1] / x[-2]
    observed = y[-1] / y[-2]
    expected_n = n_ratio
    expected_nln = n_ratio * math.log(x[-1]) / math.log(x[-2])
    return abs(observed - expected_n) <= abs(observed - expected_nln)


def _select_model(x: List[float], y: List[float]) -> str:
    """Select the best Big-O complexity class.

    Algorithm
    ---------
    1. **Constant check**: outlier-robust CV test — if the data (or the
       data minus any single outlier) has CV < 8 %, classify as O(1).
    2. **≤ 2 points**: fall back to log-log slope (OLS is degenerate).
    3. **3 points, low dynamic range**: use log-log slope — with only
       1 degree of freedom, OLS cannot distinguish models.
    4. **≥ 4 points (or 3 with high DR)**: step-up OLS — start from the
       simplest valid model and accept a more complex one only if it
       reduces RSS by a factor that depends on the dynamic range.
       Low DR requires a larger improvement (avoids overfitting noise),
       high DR is more permissive (real growth is clear).
    5. **O(n) → O(n log n) guard**: when OLS prefers O(n log n) over O(n),
       verify using the growth ratio at the two largest n values.  If the
       observed ratio is closer to pure O(n), keep O(n).
    """
    y_mean = sum(y) / len(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)

    # (1) Constant check (with outlier robustness)
    if ss_tot < 1e-30 or _is_effectively_constant(y):
        return "O(1)"

    # Sort by x
    pairs = sorted(zip(x, y))
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    # (2) Two-point degeneracy
    if len(x) <= 2:
        if all(v > 0 for v in x) and all(v > 0 for v in y):
            return _slope_to_model(_log_log_slope(x, y))
        return "O(n)"

    y_min, y_max = min(y), max(y)
    dr = y_max / y_min if y_min > 0 else float("inf")

    # (3) Three points with low dynamic range → log-log slope
    if len(x) == 3 and dr < 10 and all(v > 0 for v in y):
        return _slope_to_model(_log_log_slope(x, y))

    # (4) Step-up OLS with adaptive threshold
    fits = []
    for name, f in _ALL_MODELS:
        C, b, rss = _ols_fit(x, y, f)
        r2 = 1.0 - rss / ss_tot
        fits.append((name, C, b, rss, r2))

    # Adaptive step-up ratio: harder to justify more complexity
    # when the dynamic range is low (baseline-dominated data).
    if dr < 2:
        step_ratio = 20.0
    elif dr < 5:
        step_ratio = 10.0
    else:
        step_ratio = 2.5

    selected: str | None = None
    selected_rss = float("inf")

    for name, C, _, rss, r2 in fits[1:]:  # skip O(1)
        if C < 0 or r2 < 0.5:
            continue
        if selected is None:
            selected = name
            selected_rss = rss
            continue
        # (5) Guard O(n)→O(n log n): these differ only by a log factor,
        # so even small overhead curvature can fool OLS.  Use tail ratio.
        if selected == "O(n)" and name == "O(n log n)":
            if _tail_ratio_favors_linear(x, y):
                continue
        if selected_rss > 0 and rss < selected_rss / step_ratio:
            selected = name
            selected_rss = rss

    return selected or "O(1)"


def _upper_bound_offset(
    x: List[float],
    y: List[float],
    basis: Callable[[float], float],
    C: float,
    baseline: float,
) -> float:
    """Compute offset so that C·f(n) + baseline + offset ≥ y_i for all points."""
    max_above = 0.0
    for xi, yi in zip(x, y):
        predicted = C * basis(xi) + baseline
        shortfall = yi - predicted
        if shortfall > max_above:
            max_above = shortfall
    return max_above


def fit_models(df: pd.DataFrame, x_col: str, y_col: str, by: List[str]) -> pd.DataFrame:
    """Fit Big-O complexity models per group.

    Algorithm:
    1. Select the complexity class via outlier-robust constant detection,
       log-log slope fallback (≤3 points), and step-up OLS with adaptive
       threshold (≥4 points).
    2. Fit y = C·f(n) + baseline via OLS for the selected model.
    3. If C < 0, fall back to a simpler model.
    4. Shift the curve up by `offset` to form a proper upper bound.

    Returns DataFrame: by…, model, C, baseline, offset, formula, rss, nobs
    """
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
                "C": C,
                "baseline": baseline,
                "offset": offset,
                "formula": _format_formula(model, C, eff_baseline),
                "rss": rss,
                "nobs": len(group),
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
    bases = _basis_functions()
    preds = []

    for _, row in fits.iterrows():
        key = {k: row[k] for k in by}
        model = row["model"]
        C = row["C"]
        baseline = row["baseline"]
        offset = row["offset"]
        formula = row["formula"]
        fn = bases[model]
        eff_baseline = baseline + offset

        sub = df
        for k, v in key.items():
            sub = sub[sub[k] == v]

        xs = sorted(pd.unique(sub[x_col].astype(float).values))
        if len(xs) < 2:
            for xv in xs:
                preds.append(
                    {
                        **key,
                        x_col: xv,
                        "yhat": C * fn(xv) + eff_baseline,
                        "model": model,
                        "formula": formula,
                    }
                )
            continue

        x_min, x_max = xs[0], xs[-1]
        n_interp = 50
        step = (x_max - x_min) / n_interp
        smooth_xs = [x_min + i * step for i in range(n_interp + 1)]

        for xv in smooth_xs:
            yhat = C * fn(xv) + eff_baseline
            preds.append(
                {**key, x_col: xv, "yhat": yhat, "model": model, "formula": formula}
            )

    return pd.DataFrame(preds)
