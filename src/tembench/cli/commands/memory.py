"""`tembench memory` — memory usage chart."""

from __future__ import annotations

from pathlib import Path

import typer

from ...plotting import plot_memory
from ..app import app, console


@app.command()
def memory(
    summary: Path = typer.Option(
        Path("artifacts/summary.csv"),
        exists=True,
        dir_okay=False,
        help="Path to summary CSV",
    ),
    x: str = typer.Option("n", help="X axis parameter"),
    color: str = typer.Option("impl", help="Series grouping column"),
    output: Path = typer.Option(
        Path("artifacts/memory.html"), help="Output path for memory chart"
    ),
    log_x: bool = typer.Option(False, help="Use log scale for X axis"),
    log_y: bool = typer.Option(False, help="Use log scale for Y axis"),
):
    """Generate a memory usage chart from summary data."""
    chart = plot_memory(summary, x=x, color=color, log_x=log_x, log_y=log_y)

    output.parent.mkdir(parents=True, exist_ok=True)
    chart.save(output)

    console.print(f"[green]✓[/green] Memory chart saved to [bold]{output}[/bold]")
