"""`tembench dashboard` — multi-chart interactive dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...plotting import dashboard_charts, save_chart
from ..app import (
    BENCH_OPTION,
    LOG_X_OPTION,
    LOG_Y_OPTION,
    STRATEGY_OPTION,
    ComplexityStrategy,
    app,
    console,
    load_summary,
    output_option,
    print_artifact,
    print_axes,
    print_heading,
    resolve_axes,
    select_bench,
)


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
    x: Optional[str] = typer.Option(None, help="X axis parameter (default: inferred)"),
    color: Optional[str] = typer.Option(
        None, help="Series grouping column (default: inferred)"
    ),
    bench: Optional[str] = BENCH_OPTION,
    output: Path = output_option("artifacts/dashboard.html", "dashboard"),
    title: str = typer.Option("TempoBench Dashboard", help="Dashboard title"),
    complexity_strategy: ComplexityStrategy = STRATEGY_OPTION,
    log_x: Optional[bool] = LOG_X_OPTION,
    log_y: Optional[bool] = LOG_Y_OPTION,
):
    """Generate an interactive dashboard with multiple charts.

    [bold]Example:[/bold]
        tembench dashboard --summary artifacts/summary.csv --output artifacts/dashboard.html
    """
    df = select_bench(load_summary(summary), bench)
    explicit_axes = x is not None and color is not None
    x, color = resolve_axes(df, x, color)
    print_heading("Dashboard", summary=summary, title=title)
    print_axes(x, color, explicit_axes)

    if runs is None:
        default_runs = summary.parent / "runs.jsonl"
        if default_runs.exists():
            runs = default_runs
            console.print(f"[dim]Auto-detected runs:[/dim] {runs}")

    charts = dashboard_charts(
        df,
        runs_jsonl=runs,
        x=x,
        color=color,
        bench=bench,
        complexity_strategy=complexity_strategy.value,
        log_x=log_x,
        log_y=log_y,
    )
    save_chart(charts, output, title=title, kind="Dashboard", meta=str(summary))

    console.print()
    print_artifact("Dashboard", output)
