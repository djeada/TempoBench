"""`tembench report` — full HTML report with charts, tables, system info."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from ...plotting import fit_runtime
from ...reporting import generate_report
from ...runner.provenance import PROVENANCE_FILENAME
from ...summarize import infer_x_column, preferred_time_column
from ..app import (
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
)


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
        None, help="Path to complexity fits CSV (optional; fitted when absent)"
    ),
    chart: Optional[Path] = typer.Option(
        None,
        hidden=True,
        help="Ignored: the runtime chart is now drawn from the summary",
    ),
    provenance: Optional[Path] = typer.Option(
        None,
        help="Path to provenance.json describing the machine that ran the benchmark",
    ),
    x: Optional[str] = typer.Option(None, help="X axis parameter (default: inferred)"),
    color: Optional[str] = typer.Option(
        None, help="Series grouping column (default: inferred)"
    ),
    output: Path = output_option("artifacts/report.html", "HTML report"),
    title: str = typer.Option("TempoBench Report", help="Report title"),
    complexity_strategy: ComplexityStrategy = STRATEGY_OPTION,
):
    """Generate a comprehensive HTML report with charts, tables, and system info.

    [bold]Example:[/bold]
        tembench report --summary artifacts/summary.csv --output artifacts/report.html

    The report includes:
    - The fitted complexity class of every series, with its confidence
    - The runtime chart with each series' upper bound
    - Run success/failure counts (if runs.jsonl is provided or found)
    - The evidence behind each class, and the detailed results table
    - System information for reproducibility
    """
    summary_df = load_summary(summary)
    print_heading("Report", summary=summary, title=title)
    if x is None and infer_x_column(summary_df) is None:
        # A summary without a size sweep still gets a report, just no chart.
        console.print(
            "[yellow]No input-size column found[/yellow] — the report will have no "
            "runtime chart or complexity fits."
        )
    else:
        explicit_axes = x is not None and color is not None
        x, color = resolve_axes(summary_df, x, color)
        print_axes(x, color, explicit_axes)
    if chart is not None:
        console.print(
            "[yellow]--chart is ignored[/yellow] — the runtime chart is drawn from the summary."
        )

    # Auto-detect optional files if not provided
    if runs is None:
        default_runs = summary.parent / "runs.jsonl"
        if default_runs.exists():
            runs = default_runs
            console.print(f"[dim]Auto-detected runs:[/dim] {runs}")

    fitted: Path | pd.DataFrame | None = fits
    if fits is None:
        default_fits = summary.parent / "fits.csv"
        if default_fits.exists():
            fitted = default_fits
            console.print(f"[dim]Auto-detected fits:[/dim] {default_fits}")
        elif x is not None and (y := preferred_time_column(summary_df.columns)) is not None:
            # Grouped exactly as `plot` groups, so both show the same fits.
            fitted, _ = fit_runtime(
                summary_df, x=x, y=y, color=color,
                complexity_strategy=complexity_strategy.value,
            )
            if fitted.empty:
                console.print(
                    "[yellow]Not enough input sizes to fit a complexity class[/yellow]"
                    " — the report will omit that section."
                )

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

    generate_report(
        summary_csv=summary,
        runs_jsonl=runs,
        fits_csv=fitted,
        title=title,
        output_path=output,
        provenance_json=provenance,
        x=x,
        series=color,
        complexity_strategy=complexity_strategy.value,
    )

    console.print()
    print_artifact("Report", output)
