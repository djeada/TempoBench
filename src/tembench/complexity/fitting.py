"""Numerical primitives used by model selection and bound construction."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Callable, List

from .models import _basis_functions

# Legacy CV fallback kept for non-positive series where ratio tests are unusable.
_CV_CONST = 0.08
_CONST_FLAT_RATIO = 1.15
_CONST_RELAXED_RATIO = 1.5
_CONST_RHO_MAX = 0.6
_EXPONENT_BOOTSTRAP_SAMPLES = 200


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


def _rankdata(values: List[float]) -> List[float]:
    """Return average ranks for Spearman correlation."""
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation for two equal-length series."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    rx = _rankdata(x)
    ry = _rankdata(y)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x < 1e-30 or var_y < 1e-30:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _cv_is_flat(y: List[float]) -> bool:
    """Legacy CV-based flatness check used when values are not strictly positive."""
    n = len(y)
    y_mean = sum(y) / n
    if abs(y_mean) < 1e-30:
        return all(abs(yi) < 1e-30 for yi in y)

    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    return math.sqrt(ss_tot / n) / abs(y_mean) < _CV_CONST


def _flat_ratio_and_rho(x: List[float], y: List[float]) -> tuple[float, float]:
    """Return (max/min ratio, |Spearman rho|) for a positive series."""
    positive_values = [yi for yi in y if yi > 0]
    if not positive_values:
        return float("inf"), 0.0
    ratio = max(y) / min(positive_values)
    rho = abs(_spearman_rho(x, y))
    return ratio, rho


def _is_effectively_constant(y: List[float], x: List[float] | None = None) -> bool:
    """Check if data is effectively constant, with outlier robustness for n>=4.

    For positive series, prefer a scale-free rule:
    - treat globally flat data as O(1) when max/min is small, or
    - allow one-point-outlier robustness only when the remaining points are
      both low-range and low-trend (to avoid classifying slow monotone growth
      like [1.0, 1.05, ..., 1.2] as constant).

    For non-positive series, fall back to the legacy CV heuristic.
    """
    if not y:
        return True

    if x is None:
        x = list(range(len(y)))

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(y)
    if all(abs(yi) < 1e-30 for yi in y):
        return True

    if any(yi <= 0 for yi in y):
        if _cv_is_flat(y):
            return True
        if len(y) >= 4:
            for skip in range(len(y)):
                subset = [yi for i, yi in enumerate(y) if i != skip]
                if _cv_is_flat(subset):
                    return True
        return False

    ratio, rho = _flat_ratio_and_rho(x, y)
    if ratio <= _CONST_FLAT_RATIO or (
        ratio <= _CONST_RELAXED_RATIO and rho <= _CONST_RHO_MAX
    ):
        return True

    if n >= 4:
        for skip in range(n):
            subset_x = [xi for i, xi in enumerate(x) if i != skip]
            subset = [yi for i, yi in enumerate(y) if i != skip]
            ratio, rho = _flat_ratio_and_rho(subset_x, subset)
            if ratio <= _CONST_RELAXED_RATIO and rho <= _CONST_RHO_MAX:
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
    if slope < 0.08:
        return "O(1)"
    if slope < 0.45:
        return "O(log n)"
    if slope < 1.05:
        return "O(n)"
    if slope < 1.55:
        return "O(n log n)"
    if slope < 2.4:
        return "O(n²)"
    return "O(n³)"


def _quantile(values: List[float], q: float) -> float:
    """Compute a linear-interpolated quantile for a sorted list."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _bootstrap_exponent_ci(
    x: List[float], y: List[float], samples: int = _EXPONENT_BOOTSTRAP_SAMPLES
) -> tuple[float, float, float]:
    """Estimate a deterministic bootstrap CI for the empirical log-log slope."""
    if len(x) != len(y) or len(x) < 2 or any(v <= 0 for v in x) or any(v <= 0 for v in y):
        return float("nan"), float("nan"), float("nan")

    exponent = _log_log_slope(x, y)
    if len(x) < 3:
        return exponent, exponent, exponent

    seed_material = repr(list(zip(x, y))).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(seed)

    slopes: List[float] = []
    n = len(x)
    for _ in range(samples):
        for _attempt in range(8):
            idxs = [rng.randrange(n) for _ in range(n)]
            sample_x = [x[i] for i in idxs]
            if len(set(sample_x)) >= 2:
                sample_y = [y[i] for i in idxs]
                slopes.append(_log_log_slope(sample_x, sample_y))
                break

    if not slopes:
        return exponent, exponent, exponent

    slopes.sort()
    return exponent, _quantile(slopes, 0.025), _quantile(slopes, 0.975)


def _log_space_aic(
    x: List[float], y: List[float], basis: Callable[[float], float]
) -> float:
    """Compute an AIC-like score from log-space residuals of an OLS fit."""
    C, baseline, _ = _ols_fit(x, y, basis)
    if C < 0:
        return float("inf")

    positive_values = [yi for yi in y if yi > 0]
    if not positive_values:
        return float("inf")

    eps = max(1e-12, min(positive_values) * 1e-9)
    preds = [max(C * basis(xi) + baseline, eps) for xi in x]
    sse = sum(
        (math.log(max(yi, eps)) - math.log(pi)) ** 2 for yi, pi in zip(y, preds)
    )
    n = len(x)
    return n * math.log(max(sse / n, 1e-30)) + 4.0


def _tail_ratio_favors_simpler(
    x: List[float], y: List[float], simpler_model: str, complexer_model: str
) -> bool:
    """Return True when the tail growth is closer to the simpler model."""
    if len(x) < 4 or x[-1] <= x[-2] or x[-2] <= 0 or y[-2] <= 0:
        return False

    bases = _basis_functions()
    observed = y[-1] / y[-2]
    expected_simpler = bases[simpler_model](x[-1]) / bases[simpler_model](x[-2])
    expected_complexer = bases[complexer_model](x[-1]) / bases[complexer_model](x[-2])

    return abs(math.log(observed / expected_simpler)) <= abs(
        math.log(observed / expected_complexer)
    )


def _tail_ratio_favors_linear(x: List[float], y: List[float]) -> bool:
    """Backward-compatible wrapper for the O(n) vs O(n log n) tail check."""
    return _tail_ratio_favors_simpler(x, y, "O(n)", "O(n log n)")


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
