"""`tembench heatmap` — performance heatmap."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...plotting import plot_heatmap
from ..app import app, console, load_summary, print_axes, resolve_axes


@app.command()
def heatmap(
    summary: Path = typer.Option(
        Path("artifacts/summary.csv"),
        exists=True,
        dir_okay=False,
        help="Path to summary CSV",
    ),
    x: Optional[str] = typer.Option(None, help="X axis parameter (default: inferred)"),
    y: Optional[str] = typer.Option(None, help="Y axis parameter (default: inferred)"),
    value: str = typer.Option("time_ms_median", help="Value to display in cells"),
    output: Path = typer.Option(
        Path("artifacts/heatmap.html"), help="Output path for heatmap"
    ),
):
    """Generate a performance heatmap from summary data."""
    df = load_summary(summary)
    explicit_axes = x is not None and y is not None
    x, y = resolve_axes(df, x, y)
    print_axes(x, y, explicit_axes)
    chart = plot_heatmap(summary, x=x, y=y, value=value)

    output.parent.mkdir(parents=True, exist_ok=True)
    chart.save(output)

    console.print(f"[green]✓[/green] Heatmap saved to [bold]{output}[/bold]")
