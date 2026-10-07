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
# Relative residuals below this are timer resolution, not model misfit.
_REL_RESIDUAL_FLOOR = 1e-4


def _relative_scales(y: List[float]) -> List[float]:
    """Return the magnitude each residual is measured against.

    Timings are compared by relative error, so each point is scaled by its own
    value.  Zero or negative readings (below timer resolution) borrow the
    smallest positive one; a series with no positive reading falls back to
    plain least squares.
    """
    positive = [v for v in y if v > 0]
    if not positive:
        return [1.0] * len(y)
    floor = min(positive)
    return [max(v, floor) for v in y]


def _wls_fit(
    x: List[float], y: List[float], basis: Callable[[float], float]
) -> tuple[float, float, float]:
    """Fit y = C·f(n) + baseline minimising relative error.

    Returns ``(C, baseline, sse)`` where ``sse`` is the sum of squared
    relative residuals.  Plain OLS on timings spanning several decades is
    decided by the largest few points alone: it drives the intercept negative
    and misses every small-n point by orders of magnitude, so the class it
    favours is whichever happens to get a lucky intercept sign.  Weighting each
    point by 1/y² makes every reading count equally, in the same relative terms
    the model comparison scores them in.
    """
    n = len(x)
    if n < 2:
        return 0.0, (y[0] if y else 0.0), float("inf")

    scales = _relative_scales(y)
    w = [1.0 / (s * s) for s in scales]
    F = [basis(xi) for xi in x]
    # Normalise the basis so n²·2ⁿ or n³ at large n cannot overflow the sums.
    f_max = max(abs(fi) for fi in F)
    if f_max > 0:
        F = [fi / f_max for fi in F]

    W = sum(w)
    f_mean = sum(wi * fi for wi, fi in zip(w, F)) / W
    y_mean = sum(wi * yi for wi, yi in zip(w, y)) / W
    s_ff = sum(wi * (fi - f_mean) ** 2 for wi, fi in zip(w, F))
    s_fy = sum(wi * (fi - f_mean) * (yi - y_mean) for wi, fi, yi in zip(w, F, y))

    # Degenerate (constant) basis, judged relative to the basis's own weighted
    # magnitude so that tiny weights on huge timings do not trip it.
    s_f2 = sum(wi * fi * fi for wi, fi in zip(w, F))
    if s_ff <= 1e-12 * s_f2:
        C, baseline = 0.0, y_mean
    else:
        C = s_fy / s_ff
        baseline = y_mean - C * f_mean
        if baseline < 0:
            # Overhead cannot be negative.  A negative intercept is how a class
            # that grows too slowly imitates a faster one over a finite range
            # (C·n − b passing for n·log n), so it is refitted through zero.
            C = sum(wi * fi * yi for wi, fi, yi in zip(w, F, y)) / s_f2
            baseline = 0.0
    sse = sum(
        ((yi - (C * fi + baseline)) / si) ** 2 for fi, yi, si in zip(F, y, scales)
    )
    if f_max > 0:
        C /= f_max
    return C, baseline, sse


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


def _is_effectively_constant(
    y: List[float], x: List[float] | None = None, allow_outlier: bool = True
) -> bool:
    """Check if data is effectively constant, with outlier robustness for n>=4.

    For positive series, prefer a scale-free rule:
    - treat globally flat data as O(1) when max/min is small, or
    - allow one-point-outlier robustness only when the remaining points are
      both low-range and low-trend (to avoid classifying slow monotone growth
      like [1.0, 1.05, ..., 1.2] as constant).

    The point at the largest input size is never discarded as an outlier: it
    is the best evidence of asymptotic growth there is, and a series that is
    flat until its last reading jumps is exactly what growth looks like.

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

    x_max = max(x)
    droppable = [i for i in range(n) if x[i] < x_max] if allow_outlier and n >= 4 else []

    if any(yi <= 0 for yi in y):
        if _cv_is_flat(y):
            return True
        for skip in droppable:
            subset = [yi for i, yi in enumerate(y) if i != skip]
            if _cv_is_flat(subset):
                return True
        return False

    ratio, rho = _flat_ratio_and_rho(x, y)
    if ratio <= _CONST_FLAT_RATIO or (
        ratio <= _CONST_RELAXED_RATIO and rho <= _CONST_RHO_MAX
    ):
        return True

    for skip in droppable:
        subset_x = [xi for i, xi in enumerate(x) if i != skip]
        subset = [yi for i, yi in enumerate(y) if i != skip]
        ratio, rho = _flat_ratio_and_rho(subset_x, subset)
        if ratio <= _CONST_RELAXED_RATIO and rho <= _CONST_RHO_MAX:
            return True
    return False


def _constant_only_without_outlier(y: List[float], x: List[float]) -> bool:
    """Return True when the series is O(1) only once a reading is discarded."""
    return _is_effectively_constant(y, x) and not _is_effectively_constant(
        y, x, allow_outlier=False
    )


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
    """Map a log-log slope to a coarse complexity class.

    This is a hint, not a classifier: it is consulted only for series too short
    for model comparison.

    It deliberately does not return O(√n), whose true slope of 0.5 sits inside
    the band this function assigns to O(n).  Carving out a √n band would take
    that range away from O(n), and a linear series carrying a fixed additive
    cost — process startup being the common case — measures a depressed slope
    that lands squarely in it.  Mislabelling ordinary linear code as O(√n) is
    the worse error, and O(√n) remains reachable through the AIC comparison in
    `selection._select_model`, which is not fooled by an additive offset.
    """
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
    """Estimate a deterministic bootstrap CI for the empirical log-log slope.

    Points with a non-positive size or duration have no logarithm and are left
    out, rather than disabling the estimate for the whole series.
    """
    if len(x) != len(y):
        return float("nan"), float("nan"), float("nan")
    pairs = [(xi, yi) for xi, yi in zip(x, y) if xi > 0 and yi > 0]
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    if len(set(x)) < 2:
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


def _model_score(
    x: List[float], y: List[float], basis: Callable[[float], float]
) -> float:
    """Return the AIC of a relative-error fit of one class; lower is better.

    ``n·log(SSE/n) + 2k`` with k = 2 (C and baseline) is the Gaussian AIC of
    the relative residuals, so score differences are ordinary ΔAIC values.
    Relative errors below `_REL_RESIDUAL_FLOOR` are treated as that floor:
    timers do not resolve better, and without it noise-free data would win by
    arbitrarily large margins that mean nothing.
    """
    C, _, sse = _wls_fit(x, y, basis)
    if C < 0 or not math.isfinite(sse):
        return float("inf")
    n = len(x)
    return n * math.log(max(sse / n, _REL_RESIDUAL_FLOOR**2)) + 4.0


def _tail_ratio_favors_simpler(
    x: List[float],
    y: List[float],
    simpler_model: str,
    complexer_model: str,
    baseline: float = 0.0,
) -> bool:
    """Return True when the tail growth is closer to the simpler model.

    ``baseline`` is subtracted first: a constant overhead flattens the raw
    ratio of the last two readings and makes every class look simpler than it
    is.
    """
    if len(x) < 4 or x[-1] <= x[-2] or x[-2] <= 0:
        return False
    low, high = y[-2] - baseline, y[-1] - baseline
    if low <= 0 or high <= 0:
        return False

    bases = _basis_functions()
    f_simple = bases[simpler_model]
    f_complex = bases[complexer_model]
    if f_simple(x[-2]) <= 0 or f_complex(x[-2]) <= 0:
        return False
    observed = high / low
    expected_simpler = f_simple(x[-1]) / f_simple(x[-2])
    expected_complexer = f_complex(x[-1]) / f_complex(x[-2])

    return abs(math.log(observed / expected_simpler)) <= abs(
        math.log(observed / expected_complexer)
    )


def _upper_bound_scale(
    x: List[float],
    y: List[float],
    basis: Callable[[float], float],
    C: float,
    baseline: float,
) -> float:
    """Return the smallest factor ≥ 1 lifting the curve over every point.

    The curve is fitted by relative error, so the misses it leaves are
    relative too.  Covering them additively would lift the whole curve by the
    absolute miss at the largest n — hundreds of ms over readings of one — so
    the curve is scaled instead, which keeps the bound as tight at small n as
    at large.  Points the curve predicts as non-positive are left to
    `_upper_bound_offset`.
    """
    scale = 1.0
    for xi, yi in zip(x, y):
        predicted = C * basis(xi) + baseline
        if predicted > 0 and yi > predicted * scale:
            scale = yi / predicted
    return scale


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
