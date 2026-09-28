"""Regression tests for Big-O fitting bugs found by auditing synthetic sweeps."""

from __future__ import annotations

import math
import random

import pandas as pd
import pytest

from tembench.complexity import _basis_functions, _select_model, assess_fit, fit_models, predict_series
from tembench.complexity.fitting import _tail_ratio_favors_simpler, _wls_fit
from tembench.complexity.selection import _STEP_DOWN_AIC_TOL, _rank_models, runner_up


def _geomspace(lo: float, hi: float, count: int) -> list[float]:
    return [lo * (hi / lo) ** (i / (count - 1)) for i in range(count)]


def _fit(x, y, **kwargs):
    df = pd.DataFrame({"s": "a", "n": x, "t": y})
    fits = fit_models(df, "n", "t", ["s"], **kwargs)
    return fits.iloc[0] if len(fits) else None


def _bound_covers(row, x, y) -> bool:
    fn = _basis_functions()[row["model"]]
    level = row["baseline"] + row["offset"]
    return all(row["C"] * fn(xi) + level >= yi - 1e-9 * abs(yi) for xi, yi in zip(x, y))


# --- Relative-error fitting (plain OLS was decided by the largest-n points) ---


def test_cubic_over_two_decades_is_not_mistaken_for_log():
    x = _geomspace(10, 1000, 10)
    for seed in range(10):
        rng = random.Random(seed)
        y = [v**3 * (1 + rng.gauss(0, 0.01)) for v in x]
        assert _select_model(x, y) == "O(n³)"


def test_noisy_linear_over_three_decades_is_never_confidently_n_log_n():
    x = _geomspace(1e3, 1e6, 10)
    for seed in range(40):
        rng = random.Random(seed)
        y = [1000 * v / 1e6 * (1 + rng.gauss(0, 0.05)) for v in x]
        row = _fit(x, y)
        assert row["model"] == "O(n)" or row["confidence"] != "high", seed


# Five-repeat medians from a real sweep; both were fitted with plain OLS as
# O(n²) (HIGH) and "O(n log n) ≈ O(n²)".  Measured growth is ~n^1.25.
_SORT_SCAN = ([1e4, 5e4, 1e5, 5e5, 1e6], [1.057, 6.82, 22.5, 119.0, 350.3])
_HASH_SET = ([1e4, 5e4, 1e5, 5e5, 1e6], [0.198, 1.69, 6.06, 27.2, 64.3])


@pytest.mark.parametrize("series", [_SORT_SCAN, _HASH_SET], ids=["sort_scan", "hash_set"])
def test_real_cache_bound_series_is_n_log_n_not_quadratic(series):
    x, y = series
    df = pd.DataFrame({"s": "a", "n": x, "t": y, "count": 5})
    row = fit_models(df, "n", "t", ["s"], count_col="count").iloc[0]

    assert row["model"] == "O(n log n)"
    assert row["runner_up"] != "O(n²)"
    # The bound comes from the same relative fit, so it is tight at small n
    # too — not a curve 30x above the first reading.
    fn = _basis_functions()[row["model"]]
    first = row["C"] * fn(x[0]) + row["baseline"] + row["offset"]
    assert y[0] <= first < 2 * y[0]
    assert _bound_covers(row, x, y)


def test_exponent_interval_outside_the_class_band_is_flagged():
    x = [1e4, 5e4, 1e5, 5e5, 1e6]
    y = [v**1.25 * 1e-5 for v in x]
    quality = assess_fit(
        x, y, effective_baseline=0.0, exponent_ci_low=1.2, exponent_ci_high=1.3,
        model_margin=50.0, model="O(n²)", fitted_baseline=0.0,
    )
    assert "exponent-mismatch" in quality.notes
    assert quality.confidence != "high"


def test_overhead_can_explain_an_exponent_below_the_band():
    x = [1e3, 1e4, 1e5, 1e6]
    y = [400 + 600 * (v / 1e6) ** 2 for v in x]
    quality = assess_fit(
        x, y, effective_baseline=400.0, exponent_ci_low=0.1, exponent_ci_high=0.4,
        model_margin=50.0, model="O(n²)", fitted_baseline=400.0,
    )
    assert "exponent-mismatch" not in quality.notes


def test_log_class_band_follows_its_local_exponent_at_small_n():
    x = list(range(2, 9))
    y = [math.log(v) for v in x]
    quality = assess_fit(
        x, y, effective_baseline=0.0, exponent_ci_low=0.7, exponent_ci_high=0.9,
        model_margin=50.0, model="O(log n)", fitted_baseline=0.0,
    )
    assert "exponent-mismatch" not in quality.notes


# --- Tail-ratio step-down (overrode the score however badly it disagreed) ---


def test_noise_free_n_log_n_with_overhead_is_not_stepped_down():
    x = _geomspace(1e3, 1e6, 10)
    top = 1e6 * math.log(1e6)
    for share in (0.1, 0.3, 0.6):
        y = [share * 1000 + (1 - share) * 1000 * v * math.log(v) / top for v in x]
        row = _fit(x, y)
        assert row["model"] == "O(n log n)", share
        assert row["model_margin"] > 10


def test_sqrt_with_overhead_is_not_stepped_down_to_log():
    x = _geomspace(1e3, 1e6, 10)
    y = [400 + 600 * math.sqrt(v / 1e6) for v in x]
    row = _fit(x, y)
    assert row["model"] == "O(√n)"
    assert "another class" not in row["confidence_notes"]


def test_step_down_never_costs_more_than_the_tolerance():
    for seed in range(30):
        rng = random.Random(seed)
        x = _geomspace(10, 1000, 8)
        y = [(200 + v * math.log(v)) * (1 + rng.gauss(0, 0.1)) for v in x]
        selected, scores = _rank_models(x, y)
        if scores:
            _, margin = runner_up(selected, scores)
            assert margin >= -_STEP_DOWN_AIC_TOL


def test_tail_ratio_discounts_the_baseline():
    x = [1e3, 1e4, 1e5, 1e6]
    y = [500 + v * math.log(v) * 1e-3 for v in x]
    assert _tail_ratio_favors_simpler(x, y, "O(n)", "O(n log n)")
    assert not _tail_ratio_favors_simpler(x, y, "O(n)", "O(n log n)", baseline=500)


# --- O(1) outlier rule ---


def test_jump_at_the_largest_input_is_not_discarded_as_an_outlier():
    x = [1e3, 1e4, 1e5, 1e6, 1e7]
    row = _fit(x, [102, 98, 101, 99, 400])
    assert row["model"] != "O(1)"
    assert "largest input" in row["confidence_notes"]
    assert row["confidence"] != "high"


def test_constant_after_dropping_an_early_outlier_says_so():
    x = [1e3, 1e4, 1e5, 1e6, 1e7]
    row = _fit(x, [400, 100, 98, 101, 99])
    assert row["model"] == "O(1)"
    assert "outlying reading" in row["confidence_notes"]
    assert math.isfinite(row["rss"])


# --- Non-finite and non-positive readings ---


def test_infinite_reading_is_dropped_not_fitted():
    row = _fit([10, 100, 1000, 10000, 100000], [1, 10, 100, 1000, float("inf")])
    assert row["model"] == "O(n)"
    assert row["nobs"] == 4
    assert "nan" not in row["formula"] and "inf" not in row["formula"]
    assert all(math.isfinite(row[c]) for c in ["C", "baseline", "offset", "rss"])


def test_group_left_with_one_finite_point_is_skipped():
    df = pd.DataFrame(
        {"s": ["a", "a", "b", "b", "b"], "n": [1, 2, 1, 2, 4],
         "t": [1.0, float("nan"), 1.0, 2.0, 4.0]}
    )
    fits = fit_models(df, "n", "t", ["s"])
    assert fits["s"].tolist() == ["b"]


def test_zero_size_does_not_switch_the_range_checks_off():
    x = [0, 1, 2, 3]
    row = _fit(x, [v * v for v in x])
    assert row["confidence"] != "high"
    assert "span less than" in row["confidence_notes"]


def test_all_zero_durations_are_low_confidence():
    row = _fit([10, 100, 1000, 10000], [0.0, 0.0, 0.0, 0.0])
    assert row["model"] == "O(1)"
    assert row["confidence"] == "low"
    assert "zero or negative" in row["confidence_notes"]


# --- log bases at n = 1 ---


def test_log_bases_are_zero_at_one():
    bases = _basis_functions()
    assert bases["O(log n)"](1) == 0.0
    assert bases["O(n log n)"](1) == 0.0


def test_exact_log_and_n_log_n_from_one_are_recognised():
    x = list(range(1, 9))
    assert _select_model(x, [math.log(v) + 0.5 for v in x]) == "O(log n)"
    assert _select_model(x, [v * math.log(v) for v in x]) == "O(n log n)"
    row = _fit(x, [math.log(v) + 0.5 for v in x])
    assert row["model"] == "O(log n)" and _bound_covers(row, x, [math.log(v) + 0.5 for v in x])


# --- Minor numerics ---


def test_constant_basis_has_finite_residuals():
    C, baseline, sse = _wls_fit([1, 2, 3], [4.0, 5.0, 6.0], lambda n: 1.0)
    assert C == 0.0 and math.isfinite(sse) and sse > 0


def test_exponential_series_is_fitted_despite_huge_basis_values():
    x = [10, 20, 30, 40, 50]
    assert _select_model(x, [v * v * 2**v * 1e-12 for v in x]) == "O(n² 2^n)"


def test_predictions_are_geometric_and_cover_every_point():
    x = [1e3, 1e4, 1e5, 1e6]
    y = [2.0, 11.0, 130.0, 1500.0]
    df = pd.DataFrame({"s": "a", "n": x, "t": y})
    fits = fit_models(df, "n", "t", ["s"])
    preds = predict_series(df, fits, "n", ["s"])

    xs = preds["n"].tolist()
    assert all(v in xs for v in x)
    grid = [v for v in xs if v not in x]
    ratios = [b / a for a, b in zip(grid, grid[1:])]
    assert max(ratios) / min(ratios) < 1.01
    by_n = dict(zip(preds["n"], preds["yhat"]))
    assert all(by_n[xi] >= yi - 1e-9 for xi, yi in zip(x, y))
