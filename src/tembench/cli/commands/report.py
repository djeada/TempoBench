"""`tembench report` — full HTML report with charts, tables, system info."""

from __future__ import annotations

import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from ...complexity import fit_models
from ...reporting import generate_report
from ...runner.provenance import PROVENANCE_FILENAME
from ...summarize import (
    count_column_for,
    infer_series_column,
    infer_x_column,
    preferred_time_column,
    spread_columns_for,
)
from ..app import app, console, load_summary


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
    provenance: Optional[Path] = typer.Option(
        None,
        help="Path to provenance.json describing the machine that ran the benchmark",
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
    summary_df = load_summary(summary)
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
            # The sweep axes come from the summary rather than fixed names, so a
            # grid that is not called `n`/`impl` still gets a complexity section.
            x_col = infer_x_column(summary_df)
            series = infer_series_column(summary_df, x_col)
            by = [c for c in ["bench", series] if c and c in summary_df.columns]
            y_fit = preferred_time_column(summary_df.columns)
            if x_col and by and y_fit:
                fitted = fit_models(
                    summary_df,
                    x_col=x_col,
                    y_col=y_fit,
                    by=by,
                    strategy=complexity_strategy.value,
                    count_col=count_column_for(y_fit),
                    spread_cols=spread_columns_for(y_fit),
                )
                # A series needs at least two input sizes to fit anything; with
                # fewer, writing the empty frame would leave a headerless CSV
                # that the report cannot read back.
                if fitted.empty:
                    console.print(
                        "[yellow]Not enough input sizes to fit a complexity class[/yellow]"
                        " — the report will omit that section."
                    )
                else:
                    temp_dir = tempfile.TemporaryDirectory()
                    fits = Path(temp_dir.name) / "fits.csv"
                    fitted.to_csv(fits, index=False)
                    console.print(
                        "[dim]Auto-generated fits:[/dim] "
                        f"{fits.name} ({complexity_strategy.value})"
                    )

    if chart is None:
        default_chart = summary.parent / "runtime.html"
        if default_chart.exists():
            chart = default_chart
            console.print(f"[dim]Auto-detected chart:[/dim] {chart}")

    if provenance is None:
        default_provenance = summary.parent / PROVENANCE_FILENAME
        if default_provenance.exists():
            provenance = default_provenance
            console.print(f"[dim]Auto-detected provenance:[/dim] {provenance}")
        else:
            console.print(
                "[yellow]No provenance snapshot found[/yellow] — the System Information "
                "section will describe this machine, not the one that ran the benchmark."
            )

    try:
        generate_report(
            summary_csv=summary,
            runs_jsonl=runs,
            fits_csv=fits,
            chart_html=chart,
            title=title,
            output_path=output,
            provenance_json=provenance,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    console.print()
    console.print(f"[green]✓[/green] Report saved to [bold]{output}[/bold]")
    console.print(f"[dim]Open in browser: file://{output.absolute()}[/dim]")
