from __future__ import annotations

import json
from pathlib import Path

from tembench.summarize import preferred_time_column, read_jsonl, summarize_runs


def _write_runs(path: Path, records: list[dict]):
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_read_jsonl(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    _write_runs(p, [{"a": 1}, {"a": 2}])
    rows = read_jsonl(p)
    assert len(rows) == 2


def test_read_jsonl_skips_bad_lines(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a":1}\nnot json\n{"a":2}\n')
    rows = read_jsonl(p)
    assert len(rows) == 2


def test_summarize_runs_basic(tmp_path: Path):
    p = tmp_path / "runs.jsonl"
    records = [
        {"bench": "t", "status": "ok", "wall_ms": 10.0, "peak_rss_mb": 1.0, "params": {"impl": "a", "n": 100}},
        {"bench": "t", "status": "ok", "wall_ms": 12.0, "peak_rss_mb": 1.5, "params": {"impl": "a", "n": 100}},
        {"bench": "t", "status": "ok", "wall_ms": 11.0, "peak_rss_mb": 1.2, "params": {"impl": "a", "n": 100}},
    ]
    _write_runs(p, records)
    df = summarize_runs(p)
    assert len(df) == 1
    assert "wall_ms_median" in df.columns
    assert "wall_ms_p10" in df.columns
    assert "wall_ms_p90" in df.columns
    assert df.iloc[0]["wall_ms_median"] == 11.0


def test_summarize_falls_back_to_wall_time_without_markers(tmp_path: Path):
    p = tmp_path / "runs.jsonl"
    _write_runs(p, [
        {"bench": "t", "status": "ok", "wall_ms": 10.0, "params": {"n": 1}},
        {"bench": "t", "status": "ok", "wall_ms": 12.0, "params": {"n": 1}},
    ])
    df = summarize_runs(p)
    assert df.iloc[0]["time_source"] == "wall"
    assert df.iloc[0]["time_ms_median"] == 11.0
    assert df.iloc[0]["wall_ms_median"] == 11.0


def test_summarize_prefers_self_reported_time(tmp_path: Path):
    p = tmp_path / "runs.jsonl"
    _write_runs(p, [
        {"bench": "t", "status": "ok", "wall_ms": 90.0, "reported_ms": 1.0, "params": {"n": 1}},
        {"bench": "t", "status": "ok", "wall_ms": 95.0, "reported_ms": 3.0, "params": {"n": 1}},
    ])
    df = summarize_runs(p)
    assert df.iloc[0]["time_source"] == "reported"
    assert df.iloc[0]["time_ms_median"] == 2.0
    # Wall time stays available so the startup overhead is still inspectable.
    assert df.iloc[0]["wall_ms_median"] == 92.5


def test_summarize_does_not_mix_metrics_within_a_group(tmp_path: Path):
    """A group with a partial marker must use wall time for every one of its rows."""
    p = tmp_path / "runs.jsonl"
    _write_runs(p, [
        {"bench": "t", "status": "ok", "wall_ms": 90.0, "reported_ms": 1.0, "params": {"n": 1}},
        {"bench": "t", "status": "ok", "wall_ms": 94.0, "params": {"n": 1}},
        {"bench": "t", "status": "ok", "wall_ms": 50.0, "reported_ms": 5.0, "params": {"n": 2}},
        {"bench": "t", "status": "ok", "wall_ms": 60.0, "reported_ms": 7.0, "params": {"n": 2}},
    ])
    df = summarize_runs(p).sort_values("n").reset_index(drop=True)
    assert df.loc[0, "time_source"] == "wall"
    assert df.loc[0, "time_ms_median"] == 92.0
    assert df.loc[1, "time_source"] == "reported"
    assert df.loc[1, "time_ms_median"] == 6.0


def test_summarize_filters_outliers_on_the_canonical_metric(tmp_path: Path):
    """Wall time is noisy; a self-reported series must not inherit its outliers."""
    p = tmp_path / "runs.jsonl"
    records = [
        {"bench": "t", "status": "ok", "wall_ms": w, "reported_ms": r, "params": {"n": 1}}
        for w, r in [(90.0, 5.0), (91.0, 5.0), (900.0, 5.0), (92.0, 5.0), (93.0, 5.0)]
    ]
    _write_runs(p, records)
    df = summarize_runs(p)
    # The 900 ms wall outlier is not an outlier in the reported series, so all
    # five trials are kept and the reported median is unaffected.
    assert df.iloc[0]["time_ms_count"] == 5
    assert df.iloc[0]["time_ms_median"] == 5.0


def test_self_reported_records_without_wall_time_are_kept(tmp_path: Path):
    """runs.jsonl is an interchange format; a producer may omit wall time.

    The self-reported reading is the preferred metric, so its handling must not
    depend on the fallback metric also being present.
    """
    p = tmp_path / "runs.jsonl"
    _write_runs(p, [
        {"bench": "t", "status": "ok", "reported_ms": 2.0, "params": {"n": 1}},
        {"bench": "t", "status": "ok", "reported_ms": 4.0, "params": {"n": 1}},
    ])
    df = summarize_runs(p)
    assert len(df) == 1
    assert df.iloc[0]["time_source"] == "reported"
    assert df.iloc[0]["time_ms_median"] == 3.0


def test_records_with_no_duration_at_all_produce_nothing(tmp_path: Path):
    p = tmp_path / "runs.jsonl"
    _write_runs(p, [{"bench": "t", "status": "ok", "params": {"n": 1}}])
    assert summarize_runs(p).empty


def test_preferred_time_column_prefers_reported_then_falls_back():
    assert preferred_time_column(["n", "time_ms_median", "wall_ms_median"]) == "time_ms_median"
    assert preferred_time_column(["n", "wall_ms_median"]) == "wall_ms_median"
    assert preferred_time_column(["n", "wall_ms_mean"]) == "wall_ms_mean"
    assert preferred_time_column(["n", "impl"]) is None


def test_summarize_runs_empty(tmp_path: Path):
    p = tmp_path / "runs.jsonl"
    p.write_text("")
    df = summarize_runs(p)
    assert df.empty


def test_summarize_keeps_all_n_points_per_impl(tmp_path: Path):
    p = tmp_path / "runs.jsonl"
    records = []
    for impl, walls in {
        "hash_set": [36.0, 37.0, 38.0, 42.0],
        "sort_scan": [38.0, 39.0, 40.0, 44.0],
        "quadratic": [48.0, 56.0, 91.0, 210.0],
    }.items():
        for n, wall in zip([1000, 2000, 4000, 8000], walls):
            records.append({"bench": "u", "status": "ok", "wall_ms": wall, "peak_rss_mb": 1.0, "params": {"impl": impl, "n": n}})
            records.append({"bench": "u", "status": "ok", "wall_ms": wall + 0.2, "peak_rss_mb": 1.1, "params": {"impl": impl, "n": n}})
    _write_runs(p, records)

    df = summarize_runs(p)
    # 3 implementations × 4 n values
    assert len(df) == 12
    assert set(df["n"].tolist()) == {1000, 2000, 4000, 8000}


def test_summarize_derives_group_columns_from_params(tmp_path: Path):
    p = tmp_path / "runs.jsonl"
    records = [
        {
            "bench": "custom",
            "status": "ok",
            "wall_ms": 5.0,
            "peak_rss_mb": 1.0,
            "params": {"algo": "fast", "size": 10},
        },
        {
            "bench": "custom",
            "status": "ok",
            "wall_ms": 6.0,
            "peak_rss_mb": 1.1,
            "params": {"algo": "fast", "size": 10},
        },
        {
            "bench": "custom",
            "status": "ok",
            "wall_ms": 9.0,
            "peak_rss_mb": 1.4,
            "params": {"algo": "slow", "size": 10},
        },
    ]
    _write_runs(p, records)

    df = summarize_runs(p)
    assert set(["bench", "algo", "size"]).issubset(df.columns)
    assert len(df) == 2
    assert sorted(df["algo"].tolist()) == ["fast", "slow"]


def test_grid_columns_excludes_measurements_and_status_counts():
    from tembench.summarize import grid_columns

    columns = [
        "bench", "impl", "n",
        "time_ms_median", "time_ms_mean", "time_ms_count", "time_ms_p10", "time_ms_p90",
        "wall_ms_median", "peak_rss_mb_mean", "time_source", "ok", "timeout",
    ]
    assert grid_columns(columns) == ["bench", "impl", "n"]


def _frame(rows: list[dict]):
    import pandas as pd

    return pd.DataFrame(rows)


def test_axis_inference_prefers_the_documented_names():
    from tembench.summarize import infer_series_column, infer_x_column

    df = _frame([
        {"bench": "b", "impl": "a", "n": n, "time_ms_median": 1.0}
        for n in [10, 100, 1000]
    ])
    assert infer_x_column(df) == "n"
    assert infer_series_column(df, "n") == "impl"


def test_axis_inference_handles_a_grid_with_other_names():
    from tembench.summarize import infer_series_column, infer_x_column

    df = _frame([
        {"bench": "b", "algorithm": algo, "vertices": v, "time_ms_median": 1.0}
        for algo in ["dijkstra", "bellman"]
        for v in [100, 1000, 10000]
    ])
    assert infer_x_column(df) == "vertices"
    assert infer_series_column(df, "vertices") == "algorithm"


def test_the_widest_numeric_sweep_is_taken_as_the_input_size():
    from tembench.summarize import infer_series_column, infer_x_column

    df = _frame([
        {"bench": "b", "threads": t, "size": s, "time_ms_median": 1.0}
        for t in [1, 2]
        for s in [10, 100, 1000, 10000]
    ])
    assert infer_x_column(df) == "size"
    assert infer_series_column(df, "size") == "threads"


def test_axis_inference_reports_nothing_when_there_is_no_numeric_axis():
    from tembench.summarize import infer_x_column

    df = _frame([{"bench": "b", "impl": "a", "time_ms_median": 1.0}])
    assert infer_x_column(df) is None


def test_series_inference_returns_none_when_there_is_only_one_axis():
    from tembench.summarize import infer_series_column

    df = _frame([{"bench": "b", "n": n, "time_ms_median": 1.0} for n in [1, 2, 3]])
    assert infer_series_column(df, "n") is None


def test_metric_column_helpers_derive_their_siblings():
    from tembench.summarize import count_column_for, spread_columns_for

    assert count_column_for("time_ms_median") == "time_ms_count"
    assert count_column_for("wall_ms_mean") == "wall_ms_count"
    assert count_column_for("n") is None
    assert spread_columns_for("time_ms_median") == ("time_ms_p10", "time_ms_p90")
    assert spread_columns_for("n") is None
