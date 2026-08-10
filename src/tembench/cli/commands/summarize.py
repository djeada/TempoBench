"""`tembench summarize` — aggregate JSONL runs into a summary CSV."""

from __future__ import annotations

from pathlib import Path

import typer

from ...summarize import TIME_SOURCE_COL, read_jsonl, summarize_runs
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
    if df.empty:
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
