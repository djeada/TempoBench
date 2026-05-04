"""`tembench dashboard` — multi-chart interactive dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...plotting import create_dashboard
from ..app import app, console


@app.command()
def dashboard(
    summary: Path = typer.Option(
        Path("artifacts/summary.csv"),
        exists=True,
        dir_okay=False,
        help="Path to summary CSV",
    ),
    runs: Optional[Path] = typer.Option(
        None, help="Path to raw JSONL runs (optional, for boxplots)"
    ),
    x: str = typer.Option("n", help="X axis parameter"),
    color: str = typer.Option("impl", help="Series grouping column"),
    output: Path = typer.Option(
        Path("artifacts/dashboard.html"), help="Output path for dashboard"
    ),
    title: str = typer.Option("TempoBench Dashboard", help="Dashboard title"),
    log_x: bool = typer.Option(False, help="Use log scale for X axis"),
    log_y: bool = typer.Option(False, help="Use log scale for Y axis"),
):
    """Generate an interactive dashboard with multiple charts.

    [bold]Example:[/bold]
        tembench dashboard --summary artifacts/summary.csv --output artifacts/dashboard.html
    """
    console.print("[bold blue]Generating TempoBench Dashboard...[/bold blue]")

    if runs is None:
        default_runs = summary.parent / "runs.jsonl"
        if default_runs.exists():
            runs = default_runs
            console.print(f"[dim]Auto-detected runs:[/dim] {runs}")

    dashboard_chart = create_dashboard(
        summary_csv=summary,
        runs_jsonl=runs,
        x=x,
        color=color,
        title=title,
        log_x=log_x,
        log_y=log_y,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    dashboard_chart.save(output)

    console.print()
    console.print(f"[green]✓[/green] Dashboard saved to [bold]{output}[/bold]")
    console.print(f"[dim]Open in browser: file://{output.absolute()}[/dim]")
