"""`tembench heatmap` — performance heatmap."""

from __future__ import annotations

from pathlib import Path

import typer

from ...plotting import plot_heatmap
from ..app import app, console


@app.command()
def heatmap(
    summary: Path = typer.Option(
        Path("artifacts/summary.csv"),
        exists=True,
        dir_okay=False,
        help="Path to summary CSV",
    ),
    x: str = typer.Option("n", help="X axis parameter"),
    y: str = typer.Option("impl", help="Y axis parameter"),
    value: str = typer.Option("wall_ms_median", help="Value to display in cells"),
    output: Path = typer.Option(
        Path("artifacts/heatmap.html"), help="Output path for heatmap"
    ),
):
    """Generate a performance heatmap from summary data."""
    chart = plot_heatmap(summary, x=x, y=y, value=value)

    output.parent.mkdir(parents=True, exist_ok=True)
    chart.save(output)

    console.print(f"[green]✓[/green] Heatmap saved to [bold]{output}[/bold]")
