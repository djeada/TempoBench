"""`tembench summarize` — aggregate JSONL runs into a summary CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from ...summarize import (
    TIME_SOURCE_COL,
    grid_columns,
    preferred_time_column,
    read_jsonl,
    summarize_runs,
)
from ..app import app, console, fail, print_artifact, print_heading


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
    time_col = preferred_time_column(df.columns)
    if df.empty or time_col is None or df[time_col].notna().sum() == 0:
        successes = sum(1 for rec in read_jsonl(runs) if rec.get("status") == "ok")
        if successes:
            raise fail(
                f"{runs} has {successes} successful trial(s) but none carry a usable duration.",
                f"Inspect what was recorded: tembench inspect --runs {runs}",
            )
        raise fail(
            f"{runs} contains no successful trials to summarize.",
            "Only trials with status 'ok' are aggregated.",
            f"Inspect what was recorded: tembench inspect --runs {runs}",
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    console.print(f"[dim]Result[/dim]  {len(df):,} configuration(s), {len(df.columns):,} columns")

    unmeasured = df[df[time_col].isna()]
    if not unmeasured.empty:
        keys = grid_columns(df.columns)
        console.print(
            f"[yellow]![/yellow] {len(unmeasured)} configuration(s) had no successful "
            "trial and are kept with empty timings:"
        )
        for _, row in unmeasured.head(10).iterrows():
            point = ", ".join(f"{k}={row[k]}" for k in keys)
            statuses = ", ".join(
                f"{int(row[s])} {s}"
                for s in ("failed", "error", "timeout", "skipped")
                if s in row and pd.notna(row[s]) and row[s]
            )
            console.print(f"    {point}  [dim]{statuses}[/dim]")
        if len(unmeasured) > 10:
            console.print(f"    [dim]… and {len(unmeasured) - 10} more[/dim]")

    if TIME_SOURCE_COL in df.columns:
        sources = sorted(df[TIME_SOURCE_COL].dropna().unique())
        if sources == ["wall"]:
            console.print(
                "[dim]Timing[/dim]  wall clock, including process startup "
                "[yellow](startup can dominate small inputs and flatten the curve)[/yellow]"
            )
        elif "wall" in sources:
            console.print(
                "[dim]Timing[/dim]  mixed: some series self-reported, others wall clock"
            )
        else:
            console.print("[dim]Timing[/dim]  self-reported by the command, startup excluded")

    console.print()
    print_artifact("Summary", out_csv)
