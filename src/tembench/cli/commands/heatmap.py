"""`tembench heatmap` — performance heatmap."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...plotting import plot_heatmap, save_chart
from ...plotting._common import default_series
from ...summarize import TIME_COLUMN_PREFERENCE, preferred_time_column
from ..app import (
    BENCH_OPTION,
    app,
    fail,
    load_summary,
    output_option,
    print_artifact,
    print_axes,
    print_heading,
    require_column,
    resolve_axes,
    select_bench,
)


@app.command()
def heatmap(
    summary: Path = typer.Option(
        Path("artifacts/summary.csv"),
        exists=True,
        dir_okay=False,
        help="Path to summary CSV",
    ),
    x: Optional[str] = typer.Option(None, help="X axis parameter (default: inferred)"),
    y: Optional[str] = typer.Option(None, help="Row parameter (default: inferred)"),
    value: Optional[str] = typer.Option(
        None, help="Value shown in the cells (default: the preferred duration column)"
    ),
    bench: Optional[str] = BENCH_OPTION,
    output: Path = output_option("artifacts/heatmap.html", "heatmap"),
    log_color: Optional[bool] = typer.Option(
        None,
        "--log-color/--no-log-color",
        help="Log or linear colour scale (default: log when values span more than 30x)",
    ),
):
    """Generate a performance heatmap from summary data."""
    df = select_bench(load_summary(summary), bench)
    explicit_axes = x is not None and y is not None
    x, y = resolve_axes(df, x, y, series_flag="--y")
    if default_series(df, y) is None:
        raise fail(
            "A heatmap needs a row column, and none could be inferred.",
            "Pass --y to name the grid column to use for rows.",
        )
    if value is not None:
        require_column(df, value, "--value")
    else:
        value = preferred_time_column(df.columns)
        if value is None:
            raise fail(
                f"{summary} has no duration column to show.",
                f"Pass --value, or provide one of: {', '.join(TIME_COLUMN_PREFERENCE)}.",
            )
    print_heading("Heatmap", summary=summary, value=value)
    print_axes(x, y, explicit_axes, series_flag="--y")
    chart = plot_heatmap(df, x=x, y=y, value=value, log_color=log_color)
    save_chart(chart, output, title=f"Heatmap: {bench}" if bench else "Heatmap", meta=str(summary))
    print_artifact("Heatmap", output)
