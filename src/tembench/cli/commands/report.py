"""`tembench report` — full HTML report with charts, tables, system info."""

from __future__ import annotations

import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from ...complexity import fit_models
from ...reporting import generate_report
from ..app import app, console


class ComplexityStrategy(str, Enum):
    heuristic = "heuristic"
    strict = "strict"


@app.command()
def report(
    summary: Path = typer.Option(
        Path("artifacts/summary.csv"),
        exists=True,
        dir_okay=False,
        help="Path to summary CSV",
    ),
    runs: Optional[Path] = typer.Option(None, help="Path to raw JSONL runs (optional)"),
    fits: Optional[Path] = typer.Option(
        None, help="Path to complexity fits CSV (optional)"
    ),
    chart: Optional[Path] = typer.Option(
        None, help="Path to pre-generated chart HTML (optional)"
    ),
    output: Path = typer.Option(
        Path("artifacts/report.html"), help="Output path for HTML report"
    ),
    title: str = typer.Option("TempoBench Report", help="Report title"),
    complexity_strategy: ComplexityStrategy = typer.Option(
        ComplexityStrategy.heuristic,
        "--complexity-strategy",
        help="How aggressively to collapse uncertain exponent bands to canonical Big-O classes",
    ),
):
    """Generate a comprehensive HTML report with charts, tables, and system info.

    [bold]Example:[/bold]
        tembench report --summary artifacts/summary.csv --output artifacts/report.html

    The report includes:
    - Performance overview with key statistics
    - Run success/failure counts (if runs.jsonl provided)
    - Embedded charts (if chart.html provided)
    - Detailed results table
    - Complexity analysis (if fits.csv provided)
    - System information for reproducibility
    """
    console.print("[bold blue]Generating TempoBench Report...[/bold blue]")

    # Auto-detect optional files if not provided
    if runs is None:
        default_runs = summary.parent / "runs.jsonl"
        if default_runs.exists():
            runs = default_runs
            console.print(f"[dim]Auto-detected runs:[/dim] {runs}")

    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    if fits is None:
        default_fits = summary.parent / "fits.csv"
        if default_fits.exists():
            fits = default_fits
            console.print(f"[dim]Auto-detected fits:[/dim] {fits}")
        else:
            summary_df = pd.read_csv(summary)
            if "n" in summary_df.columns:
                by = [c for c in ["bench", "impl"] if c in summary_df.columns]
                if not by and "bench" in summary_df.columns:
                    by = ["bench"]
                y_fit = (
                    "wall_ms_median"
                    if "wall_ms_median" in summary_df.columns
                    else "wall_ms_mean"
                )
                if by and y_fit in summary_df.columns:
                    temp_dir = tempfile.TemporaryDirectory()
                    fits = Path(temp_dir.name) / "fits.csv"
                    fit_models(
                        summary_df,
                        x_col="n",
                        y_col=y_fit,
                        by=by,
                        strategy=complexity_strategy.value,
                    ).to_csv(fits, index=False)
                    console.print(
                        "[dim]Auto-generated fits:[/dim] "
                        f"{fits.name} ({complexity_strategy.value})"
                    )

    if chart is None:
        default_chart = summary.parent / "runtime.html"
        if default_chart.exists():
            chart = default_chart
            console.print(f"[dim]Auto-detected chart:[/dim] {chart}")

    try:
        generate_report(
            summary_csv=summary,
            runs_jsonl=runs,
            fits_csv=fits,
            chart_html=chart,
            title=title,
            output_path=output,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    console.print()
    console.print(f"[green]✓[/green] Report saved to [bold]{output}[/bold]")
    console.print(f"[dim]Open in browser: file://{output.absolute()}[/dim]")
