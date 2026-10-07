from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd

from ..reporting.formatting import format_number
from ..summarize import TIME_COLUMN_PREFERENCE
from ._common import (
    AXIS_FORMAT_EXPR,
    NUMBER_FORMAT,
    PALETTE,
    SEQUENTIAL_SCHEME,
    STATUS_COLUMNS,
    axis_scale,
    build_tooltips,
    categorical_color,
    chart_title,
    default_series,
    label,
    legend_opacity,
    legend_toggle,
    log_ticks,
    message_chart,
    multi_bench,
    number_axis,
    plottable_rows,
    read_summary,
    resolve_y,
    series_columns,
    shared_color_scale,
    wants_log,
    with_series_label,
    x_title,
)

#: Memory columns, best first.
MEMORY_COLUMNS = ("peak_rss_mb_median", "peak_rss_mb_mean", "peak_rss_mb")


def plot_memory(
    summary: Path | pd.DataFrame,
    x: str = "n",
    y: str = "peak_rss_mb_median",
    color: str | None = "impl",
    bench: str | None = None,
    log_x: bool | None = None,
    log_y: bool | None = None,
) -> alt.TopLevelMixin:
    """Create a memory usage line chart from the summary CSV."""
    everything = read_summary(summary)
    df = read_summary(everything, bench)
    y_col = resolve_y(df, y, MEMORY_COLUMNS)
    title = f"Memory usage vs {x_title(x)}"

    if y_col not in df.columns:
        return message_chart("No memory data in this summary", title)

    color = default_series(df, color)
    lines = series_columns(df, x, color)
    df, series, series_title = with_series_label(df, x, color)
    measured, _ = plottable_rows(df, x, y_col)
    if log_x is None:
        log_x = x in measured.columns and wants_log(measured[x])
    if log_y is None:
        log_y = wants_log(measured[y_col])
    shown, notes = plottable_rows(df, x, y_col, log_x=log_x, log_y=log_y)
    if shown.empty:
        return message_chart("; ".join(["No measured points to plot", *notes]), title)
    legend_sel = legend_toggle("mem_legend", series, series is not None)
    tooltips = build_tooltips([(x, label(x), ","), (y_col, label(y_col), NUMBER_FORMAT)])
    if series is not None and series != "bench":
        tooltips.append(alt.Tooltip(series, title=series_title))
    if "bench" in df.columns:
        tooltips.insert(0, alt.Tooltip("bench", title="Benchmark"))

    y_scale = axis_scale(log_y)
    encoding = dict(
        x=alt.X(f"{x}:Q", title=label(x), scale=axis_scale(log_x), axis=number_axis(log_x, shown[x])),
        y=alt.Y(f"{y_col}:Q", title=label(y_col), scale=y_scale, axis=number_axis(log_y, shown[y_col])),
        color=categorical_color(
            series,
            enabled=series is not None,
            title=f"{series_title}  (click to toggle)",
            fallback_color=PALETTE[0],
            scale=shared_color_scale(everything if series in everything else df, series),
        ),
        opacity=legend_opacity(legend_sel),
        tooltip=tooltips,
    )
    # One line per series, including one per benchmark: without `detail` two
    # benches sharing an implementation name would be joined into one line.
    if lines:
        encoding["detail"] = [f"{c}:N" for c in lines]
    chart = (
        alt.Chart(shown)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=48))
        .encode(**encoding)
        .properties(
            width=640,
            height=320,
            title=chart_title(title, notes),
        )
    )
    if legend_sel is not None:
        chart = chart.add_params(legend_sel)
    return chart


def _cell_status(row: pd.Series) -> str:
    """Why a heatmap cell has no value, from the summary's status counts."""
    for status in ("timeout", "error", "failed", "skipped"):
        if status in row and pd.notna(row[status]) and row[status] > 0:
            return status
    return "no data"


def plot_heatmap(
    summary: Path | pd.DataFrame,
    x: str = "n",
    y: str | None = "impl",
    value: str = "time_ms_median",
    bench: str | None = None,
    log_color: bool | None = None,
) -> alt.TopLevelMixin:
    """Create a performance heatmap, one panel per benchmark.

    A cell must stand for exactly one grid point: grid axes other than `x` and
    `y` become part of the row name, and rows that still collide are combined
    by their median and the subtitle says so.  Grid points where no trial
    succeeded are drawn grey and labelled with what went wrong.  Without a
    row axis, several benchmarks become the rows of one panel.  The colour
    scale is logarithmic when the values span more than `LOG_SPAN`, unless
    `log_color` says otherwise.
    """
    df = read_summary(summary, bench)
    y = default_series(df, y)
    if y is None or x not in df.columns or y not in df.columns:
        return message_chart("A heatmap needs an input-size column and a row column")

    value_col = resolve_y(df, value, TIME_COLUMN_PREFERENCE)
    if value_col not in df.columns:
        return message_chart(f"No '{value}' column for the heatmap")
    df, row_field, row_title = with_series_label(df, x, y)
    assert row_field is not None  # `y` is a column of df
    facet = multi_bench(df) and y != "bench"

    keys = (["bench"] if facet else []) + [x, row_field]
    counts = [c for c in STATUS_COLUMNS if c in df.columns]
    notes: list[str] = []
    sizes = df.groupby(keys, dropna=False).size()
    if (sizes > 1).any():
        notes.append(
            f"{int((sizes > 1).sum())} cell(s) combine several summary rows; "
            "the median is shown"
        )
    agg: dict[str, str] = {value_col: "median", **{c: "sum" for c in counts}}
    cells = df.groupby(keys, dropna=False, as_index=False).agg(agg)
    cells[value_col] = pd.to_numeric(cells[value_col], errors="coerce")

    measured = cells[value_col].notna()
    cells["_status"] = [
        "ok" if ok else _cell_status(row)
        for ok, (_, row) in zip(measured, cells.iterrows())
    ]
    cells["_text"] = [
        format_number(v) if ok else status
        for ok, v, status in zip(measured, cells[value_col], cells["_status"])
    ]
    values = cells.loc[measured, value_col]
    if log_color is None:
        log_color = wants_log(values)
    elif log_color and (values <= 0).any():
        log_color = False
        notes.append("values ≤ 0 present, so the colour scale is linear")
    # White text on the darker half of the colour scale.  Decided here: a
    # threshold in the spec would be NaN, and so invalid JSON, when no cell has
    # a value.
    position = values.map(math.log10) if log_color else values
    threshold = (position.min() + position.max()) / 2 if measured.any() else 0.0
    cells["_dark_cell"] = False
    cells.loc[measured, "_dark_cell"] = (position > threshold).to_numpy()

    x_enc = alt.X(
        f"{x}:O",
        title=label(x),
        axis=alt.Axis(labelAngle=0, labelExpr=f"isNumber(datum.value) ? {AXIS_FORMAT_EXPR} : datum.label"),
    )
    y_enc = alt.Y(f"{row_field}:N", title=row_title)
    tooltips = build_tooltips(
        [
            (x, label(x), None),
            (row_field, row_title, None),
            (value_col, label(value_col), NUMBER_FORMAT),
            ("_status", "Status", None),
        ]
    ) + [alt.Tooltip(f"{c}:Q", title=label(c)) for c in counts]
    if facet:
        tooltips.insert(0, alt.Tooltip("bench", title="Benchmark"))

    scale = (
        alt.Scale(type="log", scheme=SEQUENTIAL_SCHEME)
        if log_color
        else alt.Scale(scheme=SEQUENTIAL_SCHEME)
    )
    legend: dict[str, Any] = {
        "direction": "horizontal",
        "orient": "bottom",
        "gradientLength": 280,
        "labelExpr": AXIS_FORMAT_EXPR,
    }
    if log_color:
        # A gradient legend prints ticks even beyond its own ends.
        legend["values"] = [t for t in log_ticks(values) if values.min() <= t <= values.max()]
    base = alt.Chart(cells).encode(x=x_enc, y=y_enc, tooltip=tooltips)
    rect = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color(
            f"{value_col}:Q",
            title=label(value_col) + (" — log scale" if log_color else ""),
            scale=scale,
            legend=alt.Legend(**legend),
        )
    ).transform_filter(f"isValid(datum['{value_col}'])")
    failed = base.mark_rect(cornerRadius=3, color="#a9a8a2", opacity=0.35).transform_filter(
        f"!isValid(datum['{value_col}'])"
    )
    # Measured cells are always some shade of blue, so their text colour is
    # fixed; a failed cell is translucent and takes the theme's text ink.
    text = base.mark_text(fontSize=12, fontWeight=500).encode(
        text="_text:N",
        color=alt.condition("datum._dark_cell", alt.value("white"), alt.value("#1c1c1a")),
    ).transform_filter(f"isValid(datum['{value_col}'])")
    failed_text = base.mark_text(fontSize=12, fontStyle="italic").encode(
        text="_text:N"
    ).transform_filter(f"!isValid(datum['{value_col}'])")

    layers = (
        ([rect, text] if measured.any() else [])
        + ([] if measured.all() else [failed, failed_text])
    )
    rows = cells.groupby(keys[:-2], dropna=False)[row_field].nunique().max() if facet else (
        cells[row_field].nunique()
    )
    chart = alt.layer(*layers).properties(width=640, height=max(80, 36 * int(rows)))
    title = chart_title(f"Heatmap: {label(value_col)}", notes)
    if facet:
        return (
            chart.facet(
                row=alt.Row(
                    "bench:N",
                    title=None,
                    header=alt.Header(labelAngle=0, labelOrient="top", labelAnchor="start"),
                )
            )
            .resolve_scale(y="independent")
            .properties(title=title)
        )
    return chart.properties(title=title)
