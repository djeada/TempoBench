from __future__ import annotations

from pathlib import Path

import pandas as pd

from tembench.plotting import plot_memory, plot_runtime


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
