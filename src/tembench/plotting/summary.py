from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd

from ._common import (
    PALETTE,
    axis_scale,
    build_tooltips,
    categorical_color,
    label,
    legend_opacity,
    legend_toggle,
    resolve_y,
)


def plot_memory(
    summary_csv: Path,
    x: str = "n",
    y: str = "peak_rss_mb_median",
    color: str = "impl",
    log_x: bool = False,
    log_y: bool = False,
) -> alt.Chart:
    """Create a memory usage line chart from the summary CSV."""
    df = pd.read_csv(summary_csv)
    y_col = resolve_y(df, y, ["peak_rss_mb_median", "peak_rss_mb_mean", "peak_rss_mb"])

    if y_col not in df.columns:
        return (
            alt.Chart(pd.DataFrame())
            .mark_text()
            .encode(text=alt.value("No memory data available"))
        )

    legend_sel = legend_toggle("mem_legend", color, color in df.columns)
    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=50))
        .encode(
            x=alt.X(x, title=label(x), scale=axis_scale(log_x), axis=alt.Axis(format="~s")),
            y=alt.Y(y_col, title=label(y_col), scale=axis_scale(log_y)),
            color=categorical_color(
                color,
                enabled=color in df.columns,
                title=label(color),
                fallback_color=PALETTE[2],
                scale=alt.Scale(range=PALETTE) if color in df.columns else None,
                legend=alt.Legend(title="Implementation  (click to toggle)")
                if color in df.columns
                else None,
            ),
            opacity=legend_opacity(legend_sel),
            tooltip=build_tooltips(
                [
                    (x, label(x), ","),
                    (y_col, label(y_col), ".2f"),
                ]
            ),
        )
        .properties(width=640, height=360, title="Memory Usage vs Input Size")
    )
    if legend_sel is not None:
        chart = chart.add_params(legend_sel)
    return chart


def plot_heatmap(
    summary_csv: Path,
    x: str = "n",
    y: str = "impl",
    value: str = "wall_ms_median",
) -> alt.Chart:
    """Create a performance heatmap."""
    df = pd.read_csv(summary_csv)

    if x not in df.columns or y not in df.columns:
        return (
            alt.Chart(pd.DataFrame())
            .mark_text()
            .encode(text=alt.value("Insufficient data for heatmap"))
        )

    value_col = resolve_y(df, value, ["wall_ms_median", "wall_ms_mean"])

    rect = (
        alt.Chart(df)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X(f"{x}:O", title=label(x), axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y}:O", title=label(y)),
            color=alt.Color(
                f"{value_col}:Q",
                title=label(value_col),
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(
                    direction="horizontal", orient="bottom", gradientLength=300
                ),
            ),
            tooltip=build_tooltips(
                [
                    (x, label(x), None),
                    (y, label(y), None),
                    (value_col, label(value_col), ".1f"),
                ]
            ),
        )
        .properties(width=640, height=300, title=f"Heatmap: {label(value_col)}")
    )

    text = (
        alt.Chart(df)
        .mark_text(fontSize=13, fontWeight=500)
        .encode(
            x=alt.X(f"{x}:O"),
            y=alt.Y(f"{y}:O"),
            text=alt.Text(f"{value_col}:Q", format=".1f"),
            color=alt.condition(
                alt.datum[value_col] > df[value_col].median(),
                alt.value("white"),
                alt.value("#1e293b"),
            ),
        )
    )

    return rect + text
