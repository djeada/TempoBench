from __future__ import annotations

import math

import pandas as pd
import pytest

from tembench.complexity import FitResult, _select_model, fit_models, predict_series

# ---------------------------------------------------------------------------
# Model selection tests  (deterministic, cover all canonical classes)
# ---------------------------------------------------------------------------

_SELECT_CASES = [
    # Pure complexity classes (no baseline)
    ("pure_n2_5pt", [100, 500, 1000, 5000, 10000], [n ** 2 for n in [100, 500, 1000, 5000, 10000]], "O(n²)"),
    ("pure_n_4pt", [100, 1000, 10000, 100000], [float(n) for n in [100, 1000, 10000, 100000]], "O(n)"),
    ("pure_logn_5pt", [10, 100, 1000, 10000, 100000], [5 * math.log(n) for n in [10, 100, 1000, 10000, 100000]], "O(log n)"),
    ("pure_n3_5pt", [10, 50, 100, 500, 1000], [n ** 3 for n in [10, 50, 100, 500, 1000]], "O(n³)"),
    ("pure_nlogn_4pt", [100, 1000, 10000, 100000], [n * math.log(n) for n in [100, 1000, 10000, 100000]], "O(n log n)"),

    # With additive baseline (constant overhead)
    ("n2_base_5pt", [100, 500, 1000, 5000, 10000], [50 + 0.001 * n ** 2 for n in [100, 500, 1000, 5000, 10000]], "O(n²)"),
    ("n2_base_3pt", [1000, 10000, 100000], [50 + 0.001 * n ** 2 for n in [1000, 10000, 100000]], "O(n²)"),
    ("n_base_4pt", [100, 1000, 10000, 100000], [50 + 0.5 * n for n in [100, 1000, 10000, 100000]], "O(n)"),
    ("nlogn_base_3pt", [1000, 10000, 100000], [35 + 0.003 * n * math.log(n) for n in [1000, 10000, 100000]], "O(n log n)"),
    ("logn_base_5pt", [10, 100, 1000, 10000, 100000], [50 + 5 * math.log(n) for n in [10, 100, 1000, 10000, 100000]], "O(log n)"),
    ("n3_base_4pt", [10, 50, 100, 500], [100 + n ** 3 for n in [10, 50, 100, 500]], "O(n³)"),

    # Constant / near-constant
    ("const_flat", [100, 1000, 10000, 100000], [42.0, 42.0, 42.0, 42.0], "O(1)"),
    ("const_noise", [100, 1000, 10000, 100000], [42, 43, 41.5, 42.5], "O(1)"),
    ("const_slight_rise", [1000, 10000, 100000], [37.0, 38.0, 41.0], "O(1)"),
    ("hash_lookup", [1000, 10000, 100000, 1000000], [0.5, 0.5, 0.5, 0.5], "O(1)"),

    # Realistic benchmark scenarios
    ("timsort_random", [1000, 10000, 100000], [37.4, 38.3, 75.5], "O(log n)"),
    ("timsort_sorted", [1000, 10000, 100000], [36.8, 37.7, 41.3], "O(1)"),
    ("binary_search", [1000, 10000, 100000, 1000000], [0.3, 0.4, 0.5, 0.6], "O(log n)"),
    ("bubble_sort_3pt", [1000, 10000, 100000], [5.0, 500.0, 50000.0], "O(n²)"),

    # User's actual benchmark data (unique_bench.yaml)
    ("unique_quadratic", [10000, 50000, 100000], [310.98, 6416.88, 25261.63], "O(n²)"),
    ("unique_sort_scan", [10000, 50000, 100000, 500000, 1000000, 5000000], [60.60, 83.00, 98.58, 386.22, 755.61, 4175.26], "O(n log n)"),
    ("unique_hash_set", [10000, 50000, 100000, 500000, 1000000, 5000000], [58.36, 70.88, 82.57, 267.70, 507.20, 2612.83], "O(n)"),

    # Classic algorithms (realistic timings)
    ("selection_sort", [500, 1000, 2000, 4000, 8000], [n ** 2 * 0.00001 for n in [500, 1000, 2000, 4000, 8000]], "O(n²)"),
    ("merge_sort", [1000, 10000, 100000, 1000000], [n * math.log2(n) * 0.001 for n in [1000, 10000, 100000, 1000000]], "O(n log n)"),
    ("linear_scan", [1000, 10000, 100000, 1000000], [0.001 * n for n in [1000, 10000, 100000, 1000000]], "O(n)"),
    ("matrix_mult", [10, 20, 50, 100], [n ** 3 * 0.001 for n in [10, 20, 50, 100]], "O(n³)"),
    ("dict_lookup", [100, 1000, 10000, 100000, 1000000], [0.01] * 5, "O(1)"),

    # Edge cases
    ("n2_3pt_clean", [10, 100, 1000], [100.0, 10000.0, 1000000.0], "O(n²)"),
    ("sqrt_like", [100, 1000, 10000, 100000], [10.0, 31.6, 100.0, 316.2], "O(n)"),
    ("nlogn_5pt", [100, 500, 1000, 5000, 10000], [n * math.log(n) for n in [100, 500, 1000, 5000, 10000]], "O(n log n)"),
    ("n2_noisy", [100, 500, 1000, 5000], [n ** 2 * (1 + 0.05 * (-1) ** i) for i, n in enumerate([100, 500, 1000, 5000])], "O(n²)"),
    ("linear_2pt", [1000, 100000], [10.0, 1000.0], "O(n)"),
    ("nlogn_wide", [10, 100, 1000, 10000, 100000, 1000000],
     [n * math.log(n) for n in [10, 100, 1000, 10000, 100000, 1000000]], "O(n log n)"),

    # 3-point with high dynamic range (uses OLS, not log-log)
    ("n2_3pt_high_dr", [100, 1000, 10000], [10, 1000, 100000], "O(n²)"),
    ("n3_3pt", [10, 100, 1000], [1, 1000, 1000000], "O(n³)"),
    ("n_3pt", [100, 1000, 10000], [100, 1000, 10000], "O(n)"),

    # Outlier patterns (4+ points)
    ("outlier_start", [100, 1000, 10000, 100000], [100, 50, 50, 50], "O(1)"),
    ("outlier_mid", [100, 1000, 10000, 100000], [50, 100, 50, 50], "O(1)"),

    # Subtle growth
    ("subtle_n2", [1000, 2000, 4000, 8000, 16000], [10, 12, 18, 34, 82], "O(n²)"),
    ("n2_large_base_5pt", [100, 500, 1000, 5000, 10000], [1000 + 0.001 * n ** 2 for n in [100, 500, 1000, 5000, 10000]], "O(n²)"),
    ("nlogn_noisy", [100, 500, 1000, 5000, 10000],
     [n * math.log(n) * (1 + 0.03 * (-1) ** i) for i, n in enumerate([100, 500, 1000, 5000, 10000])], "O(n log n)"),
    ("exp_like_4pt", [1, 2, 3, 4], [2, 4, 8, 16], "O(n³)"),
    ("n2_w_noise_5pt", [100, 500, 1000, 5000, 10000],
     [n ** 2 + 500 * (-1) ** i for i, n in enumerate([100, 500, 1000, 5000, 10000])], "O(n²)"),
]


@pytest.mark.parametrize("label,x,y,expected", _SELECT_CASES, ids=[c[0] for c in _SELECT_CASES])
def test_select_model(label, x, y, expected):
    """_select_model must return the correct Big-O class for known data."""
    assert _select_model(x, y) == expected


def test_select_model_deterministic():
    """Repeated calls with the same input must always produce the same result."""
    for _, x, y, expected in _SELECT_CASES:
        for _ in range(5):
            assert _select_model(x, y) == expected


def test_select_model_monotonic_ranking():
    """Faster-growing series must get higher-or-equal complexity classes.

    Given three series over the same n values where series A grows faster
    than B which grows faster than C, their assigned complexities must be
    ordered A ≥ B ≥ C.
    """
    order = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)", "O(n³)"]
    ns = [1000, 10000, 100000]
    series = [
        ([5, 500, 50000], "fast"),     # quadratic-like
        ([37, 38, 75], "medium"),       # sub-linear
        ([37, 38, 41], "slow"),         # near-constant
    ]
    models = [_select_model(ns, ys) for ys, _ in series]
    ranks = [order.index(m) for m in models]
    assert ranks[0] >= ranks[1] >= ranks[2], f"Non-monotonic: {list(zip(models, [n for _, n in series]))}"


# ---------------------------------------------------------------------------
# fit_models integration tests
# ---------------------------------------------------------------------------

def test_fit_linear_data():
    """Linear data (y = n) should be classified as O(n)."""
    df = pd.DataFrame({
        "impl": ["a"] * 5,
        "n": [100, 1000, 10000, 100000, 1000000],
        "y": [100, 1000, 10000, 100000, 1000000],
    })
    fits = fit_models(df, x_col="n", y_col="y", by=["impl"])
    assert len(fits) == 1
    assert fits.iloc[0]["model"] == "O(n)"
    assert fits.iloc[0]["C"] > 0
    assert "formula" in fits.columns
    assert "T(n)" in fits.iloc[0]["formula"]


def test_fit_quadratic_data():
    """Quadratic data (y = n²) should be classified as O(n²)."""
    ns = [10, 100, 1000, 10000]
    df = pd.DataFrame({
        "impl": ["a"] * len(ns),
        "n": ns,
        "y": [n * n for n in ns],
    })
    fits = fit_models(df, x_col="n", y_col="y", by=["impl"])
    assert fits.iloc[0]["model"] == "O(n²)"


def test_fit_log_data():
    """Logarithmic data should be classified as O(log n)."""
    ns = [10, 100, 1000, 10000, 100000]
    df = pd.DataFrame({
        "impl": ["a"] * len(ns),
        "n": ns,
        "y": [5.0 * math.log(n) + 2.0 for n in ns],
    })
    fits = fit_models(df, x_col="n", y_col="y", by=["impl"])
    assert fits.iloc[0]["model"] == "O(log n)"
    assert abs(fits.iloc[0]["C"] - 5.0) < 0.5


def test_fit_nlogn_data():
    """n·log(n) data should be classified as O(n log n)."""
    ns = [100, 1000, 10000, 100000]
    df = pd.DataFrame({
        "impl": ["a"] * len(ns),
        "n": ns,
        "y": [n * math.log(n) for n in ns],
    })
    fits = fit_models(df, x_col="n", y_col="y", by=["impl"])
    assert fits.iloc[0]["model"] == "O(n log n)"


def test_fit_cubic_data():
    """Cubic data should be classified as O(n³)."""
    ns = [10, 50, 100, 500, 1000]
    df = pd.DataFrame({
        "impl": ["a"] * len(ns),
        "n": ns,
        "y": [n ** 3 for n in ns],
    })
    fits = fit_models(df, x_col="n", y_col="y", by=["impl"])
    assert fits.iloc[0]["model"] == "O(n³)"


def test_fit_constant_data():
    """Constant data should be classified as O(1)."""
    ns = [100, 1000, 10000, 100000]
    df = pd.DataFrame({
        "impl": ["a"] * len(ns),
        "n": ns,
        "y": [42.0, 42.0, 42.0, 42.0],
    })
    fits = fit_models(df, x_col="n", y_col="y", by=["impl"])
    assert fits.iloc[0]["model"] == "O(1)"


# ---------------------------------------------------------------------------
# Predict / formula / upper-bound tests
# ---------------------------------------------------------------------------

def test_predict_series():
    df = pd.DataFrame({
        "impl": ["a", "a", "a"],
        "n": [10, 100, 1000],
        "y": [10, 100, 1000],
    })
    fits = fit_models(df, x_col="n", y_col="y", by=["impl"])
    preds = predict_series(df, fits, x_col="n", by=["impl"])
    assert not preds.empty
    assert "yhat" in preds.columns
    assert "formula" in preds.columns
    assert len(preds) > 3


def test_fit_result_predict():
    fr = FitResult(model="O(n)", C=1.0, baseline=0.0, offset=0.0, rss=0.0, nobs=5)
    vals = fr.predict([10, 100])
    assert len(vals) == 2
    assert vals[0] == 10.0
    assert vals[1] == 100.0


def test_fit_result_formula():
    fr = FitResult(model="O(n log n)", C=0.0003, baseline=12.5, offset=0.5, rss=0.0, nobs=5)
    f = fr.formula
    assert "n·log(n)" in f
    assert "T(n)" in f


def test_negative_C_rejected():
    """Models with negative scaling coefficient should be rejected."""
    df = pd.DataFrame({
        "impl": ["a"] * 3,
        "n": [10, 100, 1000],
        "y": [100, 50, 25],
    })
    fits = fit_models(df, x_col="n", y_col="y", by=["impl"])
    if not fits.empty:
        assert fits.iloc[0]["C"] >= 0


def test_upper_bound_covers_all_points():
    """The fit curve must sit at or above every observed data point."""
    ns = [100, 1000, 10000, 100000]
    ys = [10, 80, 600, 5000]
    df = pd.DataFrame({"impl": ["a"] * len(ns), "n": ns, "y": ys})
    fits = fit_models(df, x_col="n", y_col="y", by=["impl"])
    assert not fits.empty
    row = fits.iloc[0]
    from tembench.complexity import _basis_functions
    fn = _basis_functions()[row["model"]]
    eff_baseline = row["baseline"] + row["offset"]
    for ni, yi in zip(ns, ys):
        predicted = row["C"] * fn(ni) + eff_baseline
        assert predicted >= yi - 1e-9, f"Upper bound violated at n={ni}: {predicted} < {yi}"
