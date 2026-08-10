from __future__ import annotations

import math

import pandas as pd

from tembench.complexity import assess_fit, fit_models


def _clean_linear() -> tuple[list[float], list[float]]:
    xs = [1000.0, 4000.0, 16000.0, 64000.0, 256000.0]
    return xs, [x * 0.001 for x in xs]


def test_clean_wide_range_series_is_high_confidence():
    x, y = _clean_linear()
    quality = assess_fit(
        x, y, effective_baseline=0.0, exponent_ci_low=0.98, exponent_ci_high=1.02
    )
    assert quality.confidence == "high"
    assert quality.notes == ()
    assert quality.summary == ""


def test_too_few_points_is_flagged():
    quality = assess_fit(
        [1.0, 100.0, 10000.0],
        [1.0, 100.0, 10000.0],
        effective_baseline=0.0,
        exponent_ci_low=0.99,
        exponent_ci_high=1.01,
    )
    assert "few-points" in quality.notes
    assert quality.confidence == "medium"


def test_narrow_input_range_is_flagged():
    x = [1000.0, 1500.0, 2000.0, 3000.0]
    quality = assess_fit(
        x,
        [v * 0.001 for v in x],
        effective_baseline=0.0,
        exponent_ci_low=0.99,
        exponent_ci_high=1.01,
    )
    assert "narrow-n-range" in quality.notes


def test_flat_durations_are_flagged():
    x = [1000.0, 10000.0, 100000.0, 1000000.0]
    quality = assess_fit(
        x, [10.0, 10.5, 11.0, 11.5], effective_baseline=0.0,
        exponent_ci_low=0.01, exponent_ci_high=0.03,
    )
    assert "flat-signal" in quality.notes


def test_a_constant_series_is_never_rated_high():
    """O(1) is never established, only unfalsified over the range measured.

    A flat curve is exactly what a sweep that did not reach interesting input
    sizes looks like, so the caveat stays attached even when the readings are
    clean.
    """
    x = [1000.0, 10000.0, 100000.0, 1000000.0]
    quality = assess_fit(
        x, [5.0, 5.0, 5.0, 5.0], effective_baseline=0.0,
        exponent_ci_low=0.0, exponent_ci_high=0.0,
    )
    assert quality.confidence != "high"
    assert "flat-signal" in quality.notes


def test_wide_exponent_interval_is_flagged():
    x, y = _clean_linear()
    quality = assess_fit(
        x, y, effective_baseline=0.0, exponent_ci_low=0.4, exponent_ci_high=1.8
    )
    assert "wide-exponent-ci" in quality.notes


def test_constant_overhead_dominating_the_signal_is_flagged():
    """The exact failure mode of timing a whole process: startup swamps the work."""
    x, y = _clean_linear()
    quality = assess_fit(
        x,
        y,
        effective_baseline=max(y) * 0.9,
        exponent_ci_low=0.98,
        exponent_ci_high=1.02,
    )
    assert "overhead-dominated" in quality.notes
    assert "constant overhead" in quality.summary


def test_two_independent_caveats_downgrade_to_low():
    quality = assess_fit(
        [100.0, 150.0, 200.0],
        [10.0, 10.2, 10.4],
        effective_baseline=0.0,
        exponent_ci_low=0.0,
        exponent_ci_high=0.05,
    )
    assert quality.confidence == "low"
    assert len(quality.notes) >= 2


def test_non_finite_exponent_interval_does_not_add_a_caveat():
    x, y = _clean_linear()
    quality = assess_fit(
        x,
        y,
        effective_baseline=0.0,
        exponent_ci_low=float("nan"),
        exponent_ci_high=float("nan"),
    )
    assert "wide-exponent-ci" not in quality.notes


def test_an_undecided_class_is_flagged_and_names_its_rival():
    quality = assess_fit(
        *_clean_linear(),
        effective_baseline=0.0,
        exponent_ci_low=0.98,
        exponent_ci_high=1.02,
        model_margin=0.4,
    )
    assert "ambiguous-class" in quality.notes
    assert "about as well" in quality.summary


def test_a_decisive_margin_adds_no_caveat():
    quality = assess_fit(
        *_clean_linear(),
        effective_baseline=0.0,
        exponent_ci_low=0.98,
        exponent_ci_high=1.02,
        model_margin=50.0,
    )
    assert quality.confidence == "high"


def test_clean_curves_win_by_a_wide_margin():
    """The ambiguity caveat must not fire on data that plainly decides itself."""
    rows = []
    for n in [1000, 4000, 16000, 64000, 256000]:
        rows.append({"impl": "linear", "n": n, "t": n * 1e-3})
        rows.append({"impl": "nlogn", "n": n, "t": n * math.log(n) * 1e-5})
    fits = fit_models(pd.DataFrame(rows), "n", "t", ["impl"]).set_index("impl")

    assert fits.loc["linear", "model"] == "O(n)"
    assert fits.loc["nlogn", "model"] == "O(n log n)"
    assert (fits["model_margin"] > 10).all()
    assert (fits["confidence"] == "high").all()


def test_runner_up_is_reported_for_every_compared_fit():
    rows = [{"impl": "a", "n": n, "t": n * 1e-3} for n in [1000, 4000, 16000, 64000]]
    fits = fit_models(pd.DataFrame(rows), "n", "t", ["impl"])
    assert fits.loc[0, "runner_up"] not in (None, fits.loc[0, "model"])
    assert (fits["model_margin"] > 0).all()


def test_series_too_short_to_compare_has_no_runner_up():
    rows = [{"impl": "a", "n": n, "t": n * 1e-3} for n in [1000, 100000]]
    fits = fit_models(pd.DataFrame(rows), "n", "t", ["impl"])
    assert fits.loc[0, "runner_up"] is None
    # No comparison happened, so ambiguity is not claimed either; the series is
    # already called out for having too few points.
    assert "ambiguous-class" not in fits.loc[0, "confidence_notes"]
    assert "fewer than" in fits.loc[0, "confidence_notes"]


def test_fit_models_reports_confidence_per_series():
    rows = []
    for n in [1000, 4000, 16000, 64000, 256000]:
        rows.append({"impl": "linear", "n": n, "t": n * 0.001})
        # Startup-dominated: a large constant plus a tiny linear term.
        rows.append({"impl": "startup_bound", "n": n, "t": 100.0 + n * 1e-6})
    fits = fit_models(pd.DataFrame(rows), "n", "t", ["impl"]).set_index("impl")

    assert fits.loc["linear", "confidence"] == "high"
    assert fits.loc["startup_bound", "confidence"] == "low"
    assert "overhead" in str(fits.loc["startup_bound", "confidence_notes"])


def test_confidence_columns_survive_into_predictions():
    from tembench.complexity import predict_series

    df = pd.DataFrame(
        [{"impl": "a", "n": n, "t": n * 0.001} for n in [1000, 4000, 16000, 64000]]
    )
    fits = fit_models(df, "n", "t", ["impl"])
    preds = predict_series(df, fits, "n", ["impl"])
    assert "confidence" in preds.columns
    assert preds["confidence"].notna().all()


def test_upper_bound_still_covers_every_point_regardless_of_confidence():
    """Confidence is advisory; it must not change the fitted bound itself."""
    df = pd.DataFrame(
        [{"impl": "a", "n": n, "t": 100.0 + n * 1e-6} for n in [1000, 4000, 16000, 64000]]
    )
    fits = fit_models(df, "n", "t", ["impl"])
    row = fits.iloc[0]
    from tembench.complexity import _basis_functions

    basis = _basis_functions()[row["model"]]
    for _, point in df.iterrows():
        bound = row["C"] * basis(point["n"]) + row["baseline"] + row["offset"]
        assert bound >= point["t"] - 1e-9 or math.isclose(bound, point["t"])


def test_a_thin_sample_count_is_flagged():
    """Two trials per point leave noise that bends the curve rather than scattering it.

    None of the other checks can see that: the points sit on a smooth wrong
    curve, so the spread is small and the exponent interval is narrow.
    """
    x, y = _clean_linear()
    quality = assess_fit(
        x, y, effective_baseline=0.0, exponent_ci_low=0.98, exponent_ci_high=1.02,
        min_samples=2,
    )
    assert "thin-samples" in quality.notes
    assert quality.confidence == "medium"


def test_an_adequate_sample_count_adds_no_caveat():
    x, y = _clean_linear()
    quality = assess_fit(
        x, y, effective_baseline=0.0, exponent_ci_low=0.98, exponent_ci_high=1.02,
        min_samples=5,
    )
    assert quality.confidence == "high"


def test_fit_models_reads_the_sample_count_from_the_summary():
    rows = []
    for n in [1000, 4000, 16000, 64000, 256000]:
        rows.append({"impl": "thin", "n": n, "t": n * 1e-3, "t_count": 2})
        rows.append({"impl": "thorough", "n": n, "t": n * 1e-3, "t_count": 8})
    fits = fit_models(
        pd.DataFrame(rows), "n", "t", ["impl"], count_col="t_count"
    ).set_index("impl")

    assert "trials per input size" in str(fits.loc["thin", "confidence_notes"])
    assert fits.loc["thorough", "confidence"] == "high"


def test_a_summary_without_counts_is_not_penalised():
    """Older summaries carry no count column; absence is not evidence of thinness."""
    rows = [{"impl": "a", "n": n, "t": n * 1e-3} for n in [1000, 4000, 16000, 64000]]
    fits = fit_models(pd.DataFrame(rows), "n", "t", ["impl"], count_col="t_count")
    assert "trials per input size" not in str(fits.loc[0, "confidence_notes"])


def test_unrepeatable_trials_are_flagged():
    """A busy machine leaves medians that still trace a smooth, wrong curve."""
    x, y = _clean_linear()
    quality = assess_fit(
        x, y, effective_baseline=0.0, exponent_ci_low=0.98, exponent_ci_high=1.02,
        max_relative_spread=1.8,
    )
    assert "unstable-timings" in quality.notes
    assert "machine busy" in quality.summary


def test_repeatable_trials_add_no_caveat():
    x, y = _clean_linear()
    quality = assess_fit(
        x, y, effective_baseline=0.0, exponent_ci_low=0.98, exponent_ci_high=1.02,
        max_relative_spread=0.05,
    )
    assert quality.confidence == "high"


def test_fit_models_reads_the_percentile_spread():
    rows = []
    for n in [1000, 4000, 16000, 64000, 256000]:
        t = n * 1e-3
        # Same medians, but one series' trials disagreed wildly.
        rows.append({"impl": "steady", "n": n, "t": t, "t_p10": t * 0.98, "t_p90": t * 1.02})
        rows.append({"impl": "jittery", "n": n, "t": t, "t_p10": t * 0.2, "t_p90": t * 2.0})
    fits = fit_models(
        pd.DataFrame(rows), "n", "t", ["impl"], spread_cols=("t_p10", "t_p90")
    ).set_index("impl")

    assert fits.loc["steady", "confidence"] == "high"
    assert "machine busy" in str(fits.loc["jittery", "confidence_notes"])


def test_spread_columns_are_optional():
    rows = [{"impl": "a", "n": n, "t": n * 1e-3} for n in [1000, 4000, 16000, 64000]]
    fits = fit_models(
        pd.DataFrame(rows), "n", "t", ["impl"], spread_cols=("t_p10", "t_p90")
    )
    assert "machine busy" not in str(fits.loc[0, "confidence_notes"])
