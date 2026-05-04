from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tembench.plotting import create_dashboard, plot_memory, plot_runtime


def _write_summary(path: Path, records: list[dict]) -> Path:
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    return path


def test_plot_runtime_facets_multiple_benches(tmp_path: Path):
    summary = _write_summary(
        tmp_path / "summary.csv",
        [
            {"bench": "a", "impl": "x", "n": 100, "wall_ms_median": 10.0},
            {"bench": "a", "impl": "x", "n": 1000, "wall_ms_median": 20.0},
            {"bench": "b", "impl": "x", "n": 100, "wall_ms_median": 30.0},
            {"bench": "b", "impl": "x", "n": 1000, "wall_ms_median": 60.0},
        ],
    )
    chart = plot_runtime(summary, show_fit=False)
    spec = chart.to_dict()
    assert "facet" in spec


def test_plot_runtime_bench_filter_disables_facet(tmp_path: Path):
    summary = _write_summary(
        tmp_path / "summary.csv",
        [
            {"bench": "a", "impl": "x", "n": 100, "wall_ms_median": 10.0},
            {"bench": "a", "impl": "x", "n": 1000, "wall_ms_median": 20.0},
            {"bench": "b", "impl": "x", "n": 100, "wall_ms_median": 30.0},
            {"bench": "b", "impl": "x", "n": 1000, "wall_ms_median": 60.0},
        ],
    )
    chart = plot_runtime(summary, bench="a", show_fit=False)
    spec = chart.to_dict()
    assert "facet" not in spec


def test_plot_runtime_uses_zero_based_axes(tmp_path: Path):
    summary = _write_summary(
        tmp_path / "summary.csv",
        [
            {"bench": "a", "impl": "x", "n": 100, "wall_ms_median": 10.0},
            {"bench": "a", "impl": "x", "n": 1000, "wall_ms_median": 20.0},
        ],
    )
    chart = plot_runtime(summary, bench="a", show_fit=False)
    spec = chart.to_dict()
    base_enc = spec["layer"][0]["encoding"]
    assert base_enc["x"]["scale"]["zero"] is True
    assert base_enc["y"]["scale"]["zero"] is True


def test_plot_memory_uses_zero_based_axes(tmp_path: Path):
    summary = _write_summary(
        tmp_path / "summary.csv",
        [
            {"bench": "a", "impl": "x", "n": 100, "peak_rss_mb_median": 5.0},
            {"bench": "a", "impl": "x", "n": 1000, "peak_rss_mb_median": 8.0},
        ],
    )
    chart = plot_memory(summary)
    spec = chart.to_dict()
    assert spec["encoding"]["x"]["scale"]["zero"] is True
    assert spec["encoding"]["y"]["scale"]["zero"] is True


def test_plot_runtime_uses_log_x_scale(tmp_path: Path):
    summary = _write_summary(
        tmp_path / "summary.csv",
        [
            {"bench": "a", "impl": "x", "n": 100, "wall_ms_median": 10.0},
            {"bench": "a", "impl": "x", "n": 1000, "wall_ms_median": 20.0},
        ],
    )
    chart = plot_runtime(summary, bench="a", show_fit=False, log_x=True)
    spec = chart.to_dict()
    base_enc = spec["layer"][0]["encoding"]
    assert base_enc["x"]["scale"]["type"] == "log"
    assert base_enc["y"]["scale"]["zero"] is True


def test_plot_runtime_uses_log_y_scale(tmp_path: Path):
    summary = _write_summary(
        tmp_path / "summary.csv",
        [
            {"bench": "a", "impl": "x", "n": 100, "wall_ms_median": 10.0},
            {"bench": "a", "impl": "x", "n": 1000, "wall_ms_median": 20.0},
        ],
    )
    chart = plot_runtime(summary, bench="a", show_fit=False, log_y=True)
    spec = chart.to_dict()
    base_enc = spec["layer"][0]["encoding"]
    assert base_enc["x"]["scale"]["zero"] is True
    assert base_enc["y"]["scale"]["type"] == "log"


def test_plot_memory_uses_log_x_scale(tmp_path: Path):
    summary = _write_summary(
        tmp_path / "summary.csv",
        [
            {"bench": "a", "impl": "x", "n": 100, "peak_rss_mb_median": 5.0},
            {"bench": "a", "impl": "x", "n": 1000, "peak_rss_mb_median": 8.0},
        ],
    )
    chart = plot_memory(summary, log_x=True)
    spec = chart.to_dict()
    assert spec["encoding"]["x"]["scale"]["type"] == "log"
    assert spec["encoding"]["y"]["scale"]["zero"] is True


def test_plot_memory_uses_log_y_scale(tmp_path: Path):
    summary = _write_summary(
        tmp_path / "summary.csv",
        [
            {"bench": "a", "impl": "x", "n": 100, "peak_rss_mb_median": 5.0},
            {"bench": "a", "impl": "x", "n": 1000, "peak_rss_mb_median": 8.0},
        ],
    )
    chart = plot_memory(summary, log_y=True)
    spec = chart.to_dict()
    assert spec["encoding"]["x"]["scale"]["zero"] is True
    assert spec["encoding"]["y"]["scale"]["type"] == "log"


def test_create_dashboard_with_log_scales(tmp_path: Path):
    summary = _write_summary(
        tmp_path / "summary.csv",
        [
            {"impl": "x", "n": 100, "wall_ms_median": 10.0, "peak_rss_mb_median": 5.0},
            {"impl": "x", "n": 1000, "wall_ms_median": 20.0, "peak_rss_mb_median": 8.0},
        ],
    )
    dashboard = create_dashboard(summary, log_x=True, log_y=True)
    spec = dashboard.to_dict()
    # Dashboard returns a vconcat, check that both charts have log scales
    assert "vconcat" in spec
    charts = spec["vconcat"]
    # First chart is runtime
    runtime_spec = charts[0]
    runtime_base = runtime_spec["layer"][0]["encoding"]
    assert runtime_base["x"]["scale"]["type"] == "log"
    assert runtime_base["y"]["scale"]["type"] == "log"
    # Second chart is memory
    memory_spec = charts[1]
    assert memory_spec["encoding"]["x"]["scale"]["type"] == "log"
    assert memory_spec["encoding"]["y"]["scale"]["type"] == "log"


def test_plot_runtime_strict_strategy_uses_display_model_field(tmp_path: Path):
    summary = _write_summary(
        tmp_path / "summary.csv",
        [
            {"bench": "a", "impl": "x", "n": 100, "wall_ms_median": 10.0 * 100 * (2.0 ** 0.1)},
            {"bench": "a", "impl": "x", "n": 1000, "wall_ms_median": 10.0 * 1000 * (3.0 ** 0.5)},
            {"bench": "a", "impl": "x", "n": 10000, "wall_ms_median": 10.0 * 10000 * (4.0 ** 0.5)},
            {"bench": "a", "impl": "x", "n": 100000, "wall_ms_median": 10.0 * 100000 * (5.0 ** 0.5)},
        ],
    )
    chart = plot_runtime(summary, bench="a", complexity_strategy="strict")
    spec = chart.to_dict()
    assert "display_model" in json.dumps(spec)
