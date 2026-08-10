"""`tembench memory` — memory usage chart."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...plotting import plot_memory
from ..app import app, console, load_summary, print_axes, resolve_axes


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
    output: Path = typer.Option(
        Path("artifacts/memory.html"), help="Output path for memory chart"
    ),
    log_x: bool = typer.Option(False, help="Use log scale for X axis"),
    log_y: bool = typer.Option(False, help="Use log scale for Y axis"),
):
    """Generate a memory usage chart from summary data."""
    df = load_summary(summary)
    explicit_axes = x is not None and color is not None
    x, color = resolve_axes(df, x, color)
    print_axes(x, color, explicit_axes)
    chart = plot_memory(summary, x=x, color=color, log_x=log_x, log_y=log_y)

    output.parent.mkdir(parents=True, exist_ok=True)
    chart.save(output)

    console.print(f"[green]✓[/green] Memory chart saved to [bold]{output}[/bold]")
