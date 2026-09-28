from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd

from ..reporting.formatting import format_number
from ..summarize import TIME_COLUMN_PREFERENCE
from ._common import (
    NUMBER_FORMAT,
    PALETTE,
    STATUS_COLUMNS,
    axis_scale,
    build_tooltips,
    categorical_color,
    chart_title,
    label,
    legend_opacity,
    legend_toggle,
    message_chart,
    plottable_rows,
    resolve_y,
    series_columns,
    shared_color_scale,
    with_series_label,
    x_title,
)


def plot_memory(
    summary_csv: Path,
    x: str = "n",
    y: str = "peak_rss_mb_median",
    color: str | None = "impl",
    log_x: bool = False,
    log_y: bool = False,
) -> alt.TopLevelMixin:
    """Create a memory usage line chart from the summary CSV."""
    df = pd.read_csv(summary_csv)
    y_col = resolve_y(df, y, ["peak_rss_mb_median", "peak_rss_mb_mean", "peak_rss_mb"])

    if y_col not in df.columns:
        return (
            alt.Chart(pd.DataFrame())
            .mark_text()
            .encode(text=alt.value("No memory data available"))
        )

    lines = series_columns(df, x, color)
    df, series, series_title = with_series_label(df, x, color)
    shown, notes = plottable_rows(df, x, y_col, log_x=log_x, log_y=log_y)
    title = f"Memory Usage vs {x_title(x)}"
    if shown.empty:
        return message_chart("; ".join(["No measured points to plot", *notes]), title)
    legend_sel = legend_toggle("mem_legend", series, series is not None)
    tooltips = build_tooltips([(x, label(x), ","), (y_col, label(y_col), NUMBER_FORMAT)])
    if series is not None:
        tooltips.append(alt.Tooltip(series, title=series_title))
    if "bench" in df.columns:
        tooltips.insert(0, alt.Tooltip("bench", title="Benchmark"))

    encoding = dict(
        x=alt.X(x, title=label(x), scale=axis_scale(log_x), axis=alt.Axis(format="~s")),
        y=alt.Y(y_col, title=label(y_col), scale=axis_scale(log_y)),
        color=categorical_color(
            series,
            enabled=series is not None,
            title=f"{series_title}  (click to toggle)",
            fallback_color=PALETTE[2],
            scale=shared_color_scale(df, series),
            legend=alt.Legend(titleLimit=480) if series is not None else None,
        ),
        opacity=legend_opacity(legend_sel),
        tooltip=tooltips,
    )
    # One line per series, including one per benchmark: without `detail` two
    # benches sharing an implementation name would be joined into one line.
    if lines:
        encoding["detail"] = lines
    chart = (
        alt.Chart(shown)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=50))
        .encode(**encoding)
        .properties(
            width=640,
            height=360,
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
    summary_csv: Path,
    x: str = "n",
    y: str | None = "impl",
    value: str = "time_ms_median",
) -> alt.TopLevelMixin:
    """Create a performance heatmap, one panel per benchmark.

    A cell must stand for exactly one grid point: grid axes other than `x` and
    `y` become part of the row name, and rows that still collide are combined
    by their median and the subtitle says so.  Grid points where no trial
    succeeded are drawn grey and labelled with what went wrong.
    """
    df = pd.read_csv(summary_csv)

    if y is None or x not in df.columns or y not in df.columns:
        return (
            alt.Chart(pd.DataFrame())
            .mark_text()
            .encode(text=alt.value("Insufficient data for heatmap"))
        )

    value_col = resolve_y(df, value, list(TIME_COLUMN_PREFERENCE))
    if value_col not in df.columns:
        return (
            alt.Chart(pd.DataFrame())
            .mark_text()
            .encode(text=alt.value(f"No '{value}' column for heatmap"))
        )
    df, row_field, row_title = with_series_label(df, x, y)
    assert row_field is not None  # `y` is a column of df
    facet = "bench" in df.columns and df["bench"].nunique(dropna=False) > 1

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
    # White text on the darker half of the linear colour scale.  Decided here:
    # a threshold in the spec would be NaN, and so invalid JSON, when no cell
    # has a value.
    values = cells.loc[measured, value_col]
    threshold = (values.min() + values.max()) / 2 if measured.any() else 0.0
    cells["_dark_cell"] = [
        bool(ok and v > threshold) for ok, v in zip(measured, cells[value_col])
    ]

    x_enc = alt.X(f"{x}:O", title=label(x), axis=alt.Axis(labelAngle=0))
    y_enc = alt.Y(f"{row_field}:O", title=row_title)
    tooltips = build_tooltips(
        [
            (x, label(x), None),
            (row_field, row_title, None),
            (value_col, label(value_col), NUMBER_FORMAT),
            ("_status", "Status", None),
        ]
    ) + [alt.Tooltip(f"{c}:Q", title=c) for c in counts]
    if facet:
        tooltips.insert(0, alt.Tooltip("bench", title="Benchmark"))

    base = alt.Chart(cells).encode(x=x_enc, y=y_enc, tooltip=tooltips)
    rect = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color(
            f"{value_col}:Q",
            title=label(value_col),
            scale=alt.Scale(scheme="blues"),
            legend=alt.Legend(
                direction="horizontal",
                orient="bottom",
                gradientLength=300,
                titleLimit=300,
            ),
        )
    ).transform_filter(f"isValid(datum['{value_col}'])")
    failed = base.mark_rect(cornerRadius=4, color="#e2e8f0").transform_filter(
        f"!isValid(datum['{value_col}'])"
    )
    text = base.mark_text(fontSize=13, fontWeight=500).encode(
        text="_text:N",
        color=alt.condition(
            "datum._dark_cell", alt.value("white"), alt.value("#1e293b")
        ),
    )

    layers = ([rect] if measured.any() else []) + ([] if measured.all() else [failed])
    chart = alt.layer(*layers, text).properties(width=640, height=300)
    title = chart_title(f"Heatmap: {label(value_col)}", notes)
    if facet:
        return chart.facet(row=alt.Row("bench:N", title="Benchmark")).properties(
            title=title
        )
    return chart.properties(title=title)
