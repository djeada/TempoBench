"""Regression tests for chart specs that used to fail to render or mislead."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import pytest
from typer.testing import CliRunner

from tembench.cli import app
from tembench.plotting import (
    create_dashboard,
    plot_boxplot,
    plot_heatmap,
    plot_memory,
    plot_runtime,
    save_chart,
)
from tembench.plotting._common import SERIES_LABEL

runner = CliRunner()
NAN = float("nan")


def _csv(path: Path, records: list[dict]) -> Path:
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def _two_impls(tmp_path: Path) -> Path:
    return _csv(
        tmp_path / "summary.csv",
        [
            {"bench": "b", "impl": impl, "n": n, "time_ms_median": k * n, "peak_rss_mb_median": 5.0 + n / 1000}
            for impl, k in (("a", 0.001), ("c", 0.002))
            for n in (1000, 10000, 100000, 1000000)
        ],
    )


def _multiparam(tmp_path: Path) -> Path:
    return _csv(
        tmp_path / "summary.csv",
        [
            {"bench": "x", "impl": impl, "dtype": dtype, "n": n, "time_ms_median": k * n}
            for impl, dtype, k in (("a", "int", 1e-4), ("a", "float", 5e-4), ("b", "int", 2e-4), ("b", "float", 1e-3))
            for n in (1000, 10000, 100000, 1000000)
        ],
    )


def _layer_data(spec: dict, layer: dict) -> list[dict]:
    """The rows a layer draws, after its filter on the row kind."""
    # Altair hoists data that every layer shares to the top level.
    data = layer.get("data") or spec["data"]
    rows = spec["datasets"][data["name"]]
    for transform in layer.get("transform", []):
        test = transform.get("filter")
        if isinstance(test, str) and test.startswith("datum._kind === "):
            kind = test.split("'")[1]
            rows = [row for row in rows if row.get("_kind") == kind]
    return rows


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_dashboard_attaches_each_selection_to_one_view(tmp_path: Path):
    """Vega rejected the dashboard: each selection was listed under two views."""
    spec = create_dashboard(_two_impls(tmp_path)).to_dict()
    names = [p["name"] for node in _walk(spec) for p in node.get("params", []) if "select" in p]
    assert len(names) == len(set(names))
    for node in _walk(spec):
        for param in node.get("params", []):
            assert len(param.get("views", [])) <= 1, param


def test_boxplot_has_no_selection_and_splits_by_input_size(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    runs.write_text("".join(
        json.dumps({"status": "ok", "wall_ms": 10.0 * n + i, "params": {"impl": impl, "n": n}}) + "\n"
        for impl in ("a", "b") for n in (10, 100) for i in range(3)
    ))
    spec = plot_boxplot(runs, x="impl", size="n").to_dict()
    assert not list(p for node in _walk(spec) for p in node.get("params", []))
    assert spec["encoding"]["x"]["field"] == "n"
    assert spec["encoding"]["xOffset"]["field"] == "impl"


def test_boxplot_picks_the_duration_per_grid_point(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    lines = [
        {"status": "ok", "wall_ms": 90.0, "reported_ms": 1.0, "params": {"impl": "a", "n": 1}},
        {"status": "ok", "wall_ms": 91.0, "reported_ms": 2.0, "params": {"impl": "a", "n": 1}},
        {"status": "ok", "wall_ms": 95.0, "params": {"impl": "b", "n": 1}},
    ]
    runs.write_text("".join(json.dumps(row) + "\n" for row in lines))
    spec = plot_boxplot(runs).to_dict()
    assert spec["encoding"]["y"]["field"] == "time_ms"
    values = sorted(row["time_ms"] for row in spec["datasets"][spec["data"]["name"]])
    assert values == [1.0, 2.0, 95.0]


def test_heatmap_facets_benches_and_names_extra_grid_axes(tmp_path: Path):
    summary = _csv(
        tmp_path / "summary.csv",
        [
            {"bench": bench, "impl": "a", "dtype": dtype, "n": n, "time_ms_median": 1.0}
            for bench in ("p", "q") for dtype in ("int", "float") for n in (1, 2)
        ],
    )
    spec = plot_heatmap(summary).to_dict()
    assert spec["facet"]["row"]["field"] == "bench"
    layer = spec["spec"]["layer"][0]
    assert layer["encoding"]["y"]["field"] == SERIES_LABEL
    cells = spec["datasets"][spec["data"]["name"]]
    assert len(cells) == 8  # one cell per grid point, nothing overplotted


def test_heatmap_labels_grid_points_without_a_measurement(tmp_path: Path):
    summary = _csv(
        tmp_path / "summary.csv",
        [
            {"impl": "a", "n": 1, "time_ms_median": 0.0019, "ok": 3, "timeout": 0},
            {"impl": "a", "n": 2, "time_ms_median": NAN, "ok": 0, "timeout": 3},
        ],
    )
    spec = plot_heatmap(summary).to_dict()
    texts = sorted(row["_text"] for row in spec["datasets"][spec["data"]["name"]])
    assert texts == ["0.00190", "timeout"]
    assert "NaN" not in json.dumps(spec)


def test_extra_grid_axes_get_their_own_series_and_fits(tmp_path: Path):
    spec = plot_runtime(_multiparam(tmp_path), color="impl").to_dict()
    fit_layer = next(layer for layer in spec["layer"] if layer["mark"]["type"] == "line")
    assert [d["field"] for d in fit_layer["encoding"]["detail"]] == ["bench", "impl", "dtype"]
    assert spec["layer"][0]["encoding"]["color"]["field"] == SERIES_LABEL
    # Formulas live in the tooltip and fits table, not a clipped subtitle.
    assert "subtitle" not in spec["title"]


def test_cli_fits_match_the_chart_series_and_bench_filter(tmp_path: Path):
    summary = _csv(
        tmp_path / "summary.csv",
        pd.read_csv(_multiparam(tmp_path)).to_dict("records")
        + [{"bench": "other", "impl": "a", "dtype": "int", "n": n, "time_ms_median": n} for n in (10, 100, 1000)],
    )
    fits = tmp_path / "fits.csv"
    result = runner.invoke(app, [
        "plot", "--summary", str(summary), "--x", "n", "--color", "impl", "--bench", "x",
        "--out-html", str(tmp_path / "rt.html"), "--export-fits", str(fits),
    ])
    assert result.exit_code == 0, result.output
    df = pd.read_csv(fits)
    assert set(df["bench"]) == {"x"}
    assert len(df) == 4 and "dtype" in df.columns


def test_single_series_chart_draws_the_fit_the_cli_prints(tmp_path: Path):
    summary = _csv(tmp_path / "summary.csv", [{"n": n, "time_ms_median": n / 10} for n in (10, 100, 1000, 10000)])
    spec = plot_runtime(summary, color=None).to_dict()
    assert any(layer["mark"]["type"] == "line" for layer in spec["layer"])


def test_log_axes_drop_nonpositive_points_and_say_so(tmp_path: Path):
    summary = _csv(
        tmp_path / "summary.csv",
        [{"impl": "a", "n": n, "time_ms_median": t} for n, t in ((0, 0.0), (10, 0.0), (100, 1.0), (1000, 10.0))],
    )
    spec = plot_runtime(summary, log_x=True, log_y=True).to_dict()
    base = spec["layer"][0]
    assert all(row["n"] > 0 and row["time_ms_median"] > 0 for row in _layer_data(spec, base))
    assert any("log scale" in line for line in spec["title"]["subtitle"])
    for layer in spec["layer"][1:]:
        if layer["mark"]["type"] == "line":
            assert all(row["n"] > 0 and row["yhat"] > 0 for row in _layer_data(spec, layer))


def test_failed_grid_points_are_not_drawn_as_zero(tmp_path: Path):
    summary = _csv(
        tmp_path / "summary.csv",
        [
            {"impl": "a", "n": 10, "time_ms_median": 1.0, "peak_rss_mb_median": 5.0, "ok": 3, "failed": 0},
            {"impl": "a", "n": 100, "time_ms_median": 10.0, "peak_rss_mb_median": 6.0, "ok": 3, "failed": 0},
            {"impl": "a", "n": 1000, "time_ms_median": NAN, "peak_rss_mb_median": NAN, "ok": 0, "failed": 3},
        ],
    )
    spec = plot_runtime(summary).to_dict()
    assert [row["n"] for row in _layer_data(spec, spec["layer"][0])] == [10, 100]
    assert any("no successful trial" in line for line in spec["title"]["subtitle"])
    mem = plot_memory(summary).to_dict()
    assert [row["n"] for row in mem["datasets"][mem["data"]["name"]]] == [10, 100]


def test_all_failed_summary_still_produces_a_chart(tmp_path: Path):
    summary = _csv(tmp_path / "summary.csv", [{"impl": "a", "n": n, "time_ms_median": NAN, "failed": 3} for n in (1, 2)])
    for chart in (plot_runtime(summary), plot_memory(summary), plot_heatmap(summary), create_dashboard(summary)):
        assert "NaN" not in json.dumps(chart.to_dict())


def test_every_layer_shares_one_legend_title(tmp_path: Path):
    spec = plot_runtime(_two_impls(tmp_path)).to_dict()
    titles = {layer["encoding"]["color"].get("title") for layer in spec["layer"] if "field" in layer["encoding"].get("color", {})}
    assert titles == {"Implementation  (click to toggle)"}


def test_labels_follow_a_non_time_y_column(tmp_path: Path):
    spec = plot_runtime(_two_impls(tmp_path), y="peak_rss_mb_median").to_dict()
    text = json.dumps(spec, ensure_ascii=False)
    assert spec["title"]["text"] == "Memory vs input size"
    assert "Upper bound (MB)" in text
    assert "Upper bound (ms)" not in text and "T(n)" not in text


def test_saved_chart_html_escapes_script_breakouts(tmp_path: Path):
    summary = _csv(
        tmp_path / "summary.csv",
        [{"bench": "</script><script>alert(1)</script>", "impl": "a", "n": n, "time_ms_median": n} for n in (1, 10)],
    )
    out = Path(save_chart(plot_runtime(summary, show_fit=False), tmp_path / "rt.html"))
    page = out.read_text(encoding="utf-8")
    assert "</script><script>alert" not in page
    assert f"vega-lite@{alt.VEGALITE_VERSION}" in page


@pytest.mark.parametrize("command", ["report", "dashboard", "memory", "heatmap"])
def test_out_html_is_accepted_everywhere(tmp_path: Path, command):
    summary = _two_impls(tmp_path)
    out = tmp_path / f"{command}.html"
    result = runner.invoke(app, [command, "--summary", str(summary), "--out-html", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()


def _two_benches(tmp_path: Path) -> Path:
    """Bench `lin` grows linearly, bench `quad` quadratically; both have impls a and b."""
    return _csv(
        tmp_path / "summary.csv",
        [
            {"bench": bench, "impl": impl, "n": n, "time_ms_median": k * (n if bench == "lin" else n * n)}
            for bench in ("lin", "quad")
            for impl, k in (("a", 1e-3), ("b", 2e-3))
            for n in (1000, 3000, 10000, 30000, 100000)
        ],
    )


def test_each_benchs_fits_are_drawn_only_in_its_own_panel(tmp_path: Path):
    """A layer with data of its own is drawn whole in every facet panel."""
    spec = plot_runtime(_two_benches(tmp_path)).to_dict()
    assert spec["facet"]["row"]["field"] == "bench"
    layers = spec["spec"]["layer"]
    assert all("data" not in layer for layer in layers), "every layer must be split by the facet"
    rows = spec["datasets"][spec["data"]["name"]]
    fits = {(r["bench"], r["display_model"]) for r in rows if r["_kind"] in ("fit", "label")}
    assert fits == {("lin", "O(n)"), ("quad", "O(n²)")}


def test_benchmarks_without_a_series_axis_share_one_panel(tmp_path: Path):
    """C++ vs Rust vs Python is the comparison; a panel each would hide it."""
    summary = _csv(
        tmp_path / "summary.csv",
        [{"bench": b, "n": n, "time_ms_median": k * n} for b, k in (("cpp", 1e-4), ("py", 1e-2)) for n in (10, 100, 1000)],
    )
    spec = plot_runtime(summary, color=None).to_dict()
    assert "facet" not in spec
    assert spec["layer"][0]["encoding"]["color"]["field"] == "bench"


def test_end_labels_that_would_overlap_are_spread_apart(tmp_path: Path):
    summary = _csv(
        tmp_path / "summary.csv",
        [{"impl": i, "n": n, "time_ms_median": k * n} for i, k in (("a", 1.0), ("b", 1.01)) for n in (10, 100, 1000, 10000)],
    )
    spec = plot_runtime(summary).to_dict()
    labels = [r for r in spec["datasets"][spec["data"]["name"]] if r["_kind"] == "label"]
    low, high = sorted(r["_label_y"] for r in labels)
    assert high / low > 1.2  # far enough apart on the log axis to read both


def test_wide_ranges_get_log_axes_unless_told_otherwise(tmp_path: Path):
    summary = _two_impls(tmp_path)  # sizes and timings both span 1000x
    enc = plot_runtime(summary).to_dict()["layer"][0]["encoding"]
    assert enc["x"]["scale"]["type"] == "log" and enc["y"]["scale"]["type"] == "log"
    assert all(f"{v:g}"[0] in "125" for v in enc["y"]["axis"]["values"]), "1-2-5 ticks, not every multiple"
    enc = plot_runtime(summary, log_x=False, log_y=False).to_dict()["layer"][0]["encoding"]
    assert enc["x"]["scale"]["zero"] is True and enc["y"]["scale"]["zero"] is True
    assert plot_memory(summary).to_dict()["encoding"]["x"]["scale"]["type"] == "log"


def test_heatmap_colour_scale_turns_logarithmic_over_a_wide_range(tmp_path: Path):
    spec = plot_heatmap(_two_impls(tmp_path)).to_dict()
    assert spec["layer"][0]["encoding"]["color"]["scale"]["type"] == "log"
    narrow = _csv(tmp_path / "narrow.csv", [{"impl": "a", "n": n, "time_ms_median": 1.0 + n} for n in (1, 2)])
    assert "type" not in plot_heatmap(narrow).to_dict()["layer"][0]["encoding"]["color"]["scale"]


def test_boxplot_respects_a_log_y_axis(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    runs.write_text("".join(
        json.dumps({"status": "ok", "wall_ms": 10.0 ** n, "params": {"impl": "a", "n": n}}) + "\n"
        for n in (1, 2, 3)
    ))
    assert plot_boxplot(runs, log_y=True).to_dict()["encoding"]["y"]["scale"]["type"] == "log"


def test_nothing_to_show_is_said_on_the_chart_not_left_blank(tmp_path: Path):
    """An empty-data text mark draws nothing: a blank page that looks like a bug."""
    no_memory = _csv(tmp_path / "s.csv", [{"impl": "a", "n": n, "time_ms_median": 1.0} for n in (1, 2)])
    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps({"status": "failed", "params": {"n": 1}}) + "\n")
    for chart in (plot_memory(no_memory), plot_heatmap(no_memory, y=None), plot_boxplot(runs)):
        spec = chart.to_dict()
        rows = spec["datasets"][spec["data"]["name"]]
        assert rows and rows[0]["_message"]


def test_series_keep_their_colour_across_charts_and_themes(tmp_path: Path):
    summary = _two_impls(tmp_path)
    runtime = plot_runtime(summary).to_dict()["layer"][0]["encoding"]["color"]["scale"]
    memory = plot_memory(summary).to_dict()["encoding"]["color"]["scale"]
    assert runtime == memory == {"domain": ["a", "c"]}  # colours come from the theme's range


def test_chart_pages_are_themed_and_responsive(tmp_path: Path):
    page = Path(save_chart(plot_runtime(_two_impls(tmp_path)), tmp_path / "rt.html", title="Runtime")).read_text()
    head = page[: page.index("</head>")]
    assert "<title>Runtime</title>" in head and 'name="viewport"' in head
    assert "localStorage" in head and "prefers-color-scheme" in head, "theme is set before first paint"
    assert '"width": "container"' in page
    assert 'id="tb-chart-dark"' in page and "tempobench:themechange" in page
