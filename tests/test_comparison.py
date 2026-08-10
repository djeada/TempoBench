from __future__ import annotations

from pathlib import Path

import pandas as pd

from tembench.reporting import compare_summaries
from tembench.reporting.comparison import comparison_tally


def _summary(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_regression_is_flagged_above_the_threshold(tmp_path: Path):
    current = _summary(tmp_path / "cur.csv", [
        {"bench": "b", "impl": "a", "n": 100, "time_ms_median": 120.0},
    ])
    baseline = _summary(tmp_path / "base.csv", [
        {"bench": "b", "impl": "a", "n": 100, "time_ms_median": 100.0},
    ])
    result = compare_summaries(current, baseline, threshold_pct=5.0)
    assert result.loc[0, "time_ms_median_delta"] == 20.0
    assert result.loc[0, "time_ms_median_delta_pct"] == 20.0
    assert bool(result.loc[0, "time_ms_median_regression"]) is True


def test_improvement_is_not_flagged(tmp_path: Path):
    current = _summary(tmp_path / "cur.csv", [
        {"bench": "b", "impl": "a", "n": 100, "time_ms_median": 80.0},
    ])
    baseline = _summary(tmp_path / "base.csv", [
        {"bench": "b", "impl": "a", "n": 100, "time_ms_median": 100.0},
    ])
    result = compare_summaries(current, baseline, threshold_pct=5.0)
    assert bool(result.loc[0, "time_ms_median_regression"]) is False


def test_only_the_canonical_duration_decides_regressions(tmp_path: Path):
    """Wall time carries startup jitter; it must not raise regressions of its own.

    Otherwise a summary carrying both families double-counts every regression,
    and ordinary variation in process startup fails the build.
    """
    rows_current = [{
        "bench": "b", "impl": "a", "n": 100,
        "time_ms_median": 100.0,   # the work itself did not change
        "wall_ms_median": 400.0,   # startup happened to be slow this time
    }]
    rows_baseline = [{
        "bench": "b", "impl": "a", "n": 100,
        "time_ms_median": 100.0,
        "wall_ms_median": 200.0,
    }]
    result = compare_summaries(
        _summary(tmp_path / "cur.csv", rows_current),
        _summary(tmp_path / "base.csv", rows_baseline),
        threshold_pct=5.0,
    )

    assert "time_ms_median_regression" in result.columns
    assert "wall_ms_median_regression" not in result.columns
    assert bool(result.loc[0, "time_ms_median_regression"]) is False
    # The wall-clock delta is still reported, just not decisive.
    assert result.loc[0, "wall_ms_median_delta_pct"] == 100.0


def test_older_summaries_with_only_wall_time_still_decide(tmp_path: Path):
    result = compare_summaries(
        _summary(tmp_path / "cur.csv", [
            {"bench": "b", "impl": "a", "n": 100, "wall_ms_median": 200.0},
        ]),
        _summary(tmp_path / "base.csv", [
            {"bench": "b", "impl": "a", "n": 100, "wall_ms_median": 100.0},
        ]),
        threshold_pct=5.0,
    )
    assert bool(result.loc[0, "wall_ms_median_regression"]) is True


def test_memory_deltas_are_reported_without_flagging_regressions(tmp_path: Path):
    result = compare_summaries(
        _summary(tmp_path / "cur.csv", [
            {"bench": "b", "impl": "a", "n": 100, "time_ms_median": 100.0,
             "peak_rss_mb_median": 20.0},
        ]),
        _summary(tmp_path / "base.csv", [
            {"bench": "b", "impl": "a", "n": 100, "time_ms_median": 100.0,
             "peak_rss_mb_median": 10.0},
        ]),
        threshold_pct=5.0,
    )
    assert result.loc[0, "peak_rss_mb_median_delta_pct"] == 100.0
    assert "peak_rss_mb_median_regression" not in result.columns


def test_no_shared_grouping_columns_yields_nothing(tmp_path: Path):
    result = compare_summaries(
        _summary(tmp_path / "cur.csv", [{"algo": "a", "time_ms_median": 1.0}]),
        _summary(tmp_path / "base.csv", [{"variant": "a", "time_ms_median": 1.0}]),
    )
    assert result.empty


def test_comparison_works_for_a_grid_that_is_not_named_impl_and_n(tmp_path: Path):
    """The grid is user-defined; `impl`/`n` is only the bundled examples' habit."""
    current = _summary(tmp_path / "cur.csv", [
        {"bench": "b", "algorithm": "dijkstra", "vertices": 100, "time_ms_median": 130.0},
        {"bench": "b", "algorithm": "bellman", "vertices": 100, "time_ms_median": 200.0},
    ])
    baseline = _summary(tmp_path / "base.csv", [
        {"bench": "b", "algorithm": "dijkstra", "vertices": 100, "time_ms_median": 100.0},
        {"bench": "b", "algorithm": "bellman", "vertices": 100, "time_ms_median": 200.0},
    ])
    result = compare_summaries(current, baseline, threshold_pct=5.0).set_index("algorithm")

    assert len(result) == 2, "each grid point must be compared against its own baseline"
    assert bool(result.loc["dijkstra", "time_ms_median_regression"]) is True
    assert bool(result.loc["bellman", "time_ms_median_regression"]) is False


def test_status_counts_are_not_mistaken_for_grid_keys(tmp_path: Path):
    """`summarize` appends per-status counts; joining on them would drop rows."""
    current = _summary(tmp_path / "cur.csv", [
        {"bench": "b", "impl": "a", "n": 100, "time_ms_median": 120.0,
         "time_source": "reported", "ok": 3},
    ])
    baseline = _summary(tmp_path / "base.csv", [
        {"bench": "b", "impl": "a", "n": 100, "time_ms_median": 100.0,
         "time_source": "wall", "ok": 5},
    ])
    result = compare_summaries(current, baseline, threshold_pct=5.0)

    assert len(result) == 1
    assert bool(result.loc[0, "time_ms_median_regression"]) is True


def test_each_win_is_counted_once_not_once_per_metric(tmp_path: Path):
    """One faster configuration is one improvement, not one per metric column."""
    current = _summary(tmp_path / "cur.csv", [
        {"bench": "b", "impl": "a", "n": 100,
         "time_ms_median": 50.0, "wall_ms_median": 150.0, "peak_rss_mb_median": 5.0},
    ])
    baseline = _summary(tmp_path / "base.csv", [
        {"bench": "b", "impl": "a", "n": 100,
         "time_ms_median": 100.0, "wall_ms_median": 300.0, "peak_rss_mb_median": 10.0},
    ])
    df = compare_summaries(current, baseline, threshold_pct=5.0)

    # Time, wall time and memory all improved, but only one configuration did.
    assert comparison_tally(df, 5.0) == {
        "compared": 1,
        "regressions": 0,
        "improvements": 1,
    }


def test_tally_never_reports_more_events_than_configurations(tmp_path: Path):
    rows_current = [
        {"bench": "b", "impl": "a", "n": 100, "time_ms_median": 200.0,
         "wall_ms_median": 400.0},
        {"bench": "b", "impl": "c", "n": 100, "time_ms_median": 10.0,
         "wall_ms_median": 20.0},
    ]
    rows_baseline = [
        {"bench": "b", "impl": "a", "n": 100, "time_ms_median": 100.0,
         "wall_ms_median": 200.0},
        {"bench": "b", "impl": "c", "n": 100, "time_ms_median": 100.0,
         "wall_ms_median": 200.0},
    ]
    df = compare_summaries(
        _summary(tmp_path / "cur.csv", rows_current),
        _summary(tmp_path / "base.csv", rows_baseline),
        threshold_pct=5.0,
    )
    tally = comparison_tally(df, 5.0)
    assert tally == {"compared": 2, "regressions": 1, "improvements": 1}


def test_comparison_report_renders_the_tally(tmp_path: Path):
    from tembench.reporting.comparison import generate_comparison_report

    df = compare_summaries(
        _summary(tmp_path / "cur.csv", [
            {"bench": "b", "impl": "a", "n": 100, "time_ms_median": 50.0},
        ]),
        _summary(tmp_path / "base.csv", [
            {"bench": "b", "impl": "a", "n": 100, "time_ms_median": 100.0},
        ]),
        threshold_pct=5.0,
    )
    html = generate_comparison_report(df, threshold_pct=5.0)
    assert "No regressions detected" in html
    assert "Improvements" in html
