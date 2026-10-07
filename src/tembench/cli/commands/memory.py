"""`tembench memory` — memory usage chart."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...plotting import plot_memory, save_chart
from ...plotting.summary import MEMORY_COLUMNS
from ..app import (
    BENCH_OPTION,
    LOG_X_OPTION,
    LOG_Y_OPTION,
    app,
    fail,
    load_summary,
    output_option,
    print_artifact,
    print_axes,
    print_heading,
    resolve_axes,
    select_bench,
)


@app.command()
def memory(
    summary: Path = typer.Option(
        Path("artifacts/summary.csv"),
        exists=True,
        dir_okay=False,
        help="Path to summary CSV",
    ),
    x: Optional[str] = typer.Option(None, help="X axis parameter (default: inferred)"),
    color: Optional[str] = typer.Option(
        None, help="Series grouping column (default: inferred)"
    ),
    bench: Optional[str] = BENCH_OPTION,
    output: Path = output_option("artifacts/memory.html", "memory chart"),
    log_x: Optional[bool] = LOG_X_OPTION,
    log_y: Optional[bool] = LOG_Y_OPTION,
):
    """Generate a memory usage chart from summary data."""
    df = select_bench(load_summary(summary), bench)
    if not any(c in df.columns for c in MEMORY_COLUMNS):
        raise fail(
            f"{summary} has no memory columns to chart.",
            f"Expected one of: {', '.join(MEMORY_COLUMNS)} — "
            "peak memory is only recorded where the platform reports it.",
        )
    explicit_axes = x is not None and color is not None
    x, color = resolve_axes(df, x, color)
    print_heading("Memory Chart", summary=summary)
    print_axes(x, color, explicit_axes)
    chart = plot_memory(df, x=x, color=color, log_x=log_x, log_y=log_y)
    save_chart(chart, output, title=f"Memory: {bench}" if bench else "Memory", meta=str(summary))
    print_artifact("Memory chart", output)
