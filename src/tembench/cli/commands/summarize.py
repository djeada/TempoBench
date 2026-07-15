"""`tembench summarize` — aggregate JSONL runs into a summary CSV."""

from __future__ import annotations

from pathlib import Path

import typer

from ...summarize import summarize_runs
from ..app import app, console, print_artifact, print_heading


@app.command()
def summarize(
    runs: Path = typer.Option(
        Path("artifacts/runs.jsonl"), exists=True, dir_okay=False
    ),
    out_csv: Path = typer.Option(Path("artifacts/summary.csv"), dir_okay=False),
    include_outliers: bool = typer.Option(
        False, help="Include outliers in medians/means"
    ),
):
    """Summarize JSONL runs into CSV with medians and percentiles."""
    print_heading(
        "Summarizing Benchmark Runs",
        runs=runs,
        outliers="included" if include_outliers else "filtered (Tukey fences)",
    )
    df = summarize_runs(runs, include_outliers=include_outliers)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if df.empty:
        console.print("[yellow bold]! No valid benchmark rows found[/yellow bold]")
        console.print("  An empty CSV was created. Check the input for valid JSONL records.")
    else:
        console.print(f"[dim]Result[/dim]  {len(df):,} configuration(s), {len(df.columns):,} columns")
    console.print()
    print_artifact("Summary", out_csv)
