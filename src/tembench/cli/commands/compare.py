"""`tembench compare` — detect regressions vs a baseline summary."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel

from ...reporting import compare_summaries, generate_comparison_report
from ...reporting.comparison import comparison_keys, comparison_tally
from ..app import (
    app,
    console,
    fail,
    load_summary,
    output_option,
    print_artifact,
    print_heading,
)


@app.command()
def compare(
    current: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="Path to current summary CSV"
    ),
    baseline: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="Path to baseline summary CSV"
    ),
    threshold: float = typer.Option(5.0, help="Regression threshold percentage"),
    output: Path = output_option("artifacts/comparison.html", "comparison report"),
    output_csv: Optional[Path] = typer.Option(
        None, help="Optional path to save comparison CSV"
    ),
):
    """Compare current benchmark results against a baseline to detect regressions.

    [bold]Example:[/bold]
        tembench compare --current artifacts/summary.csv --baseline baseline/summary.csv

    A configuration fails when it is slower than the baseline by more than the
    threshold percentage, or when the baseline measured it and the current run
    could not.
    """
    print_heading(
        "Comparing Benchmark Results",
        current=current,
        baseline=baseline,
        threshold=f"{threshold:g}%",
    )

    comparison_df = compare_summaries(
        load_summary(current), load_summary(baseline), threshold_pct=threshold
    )
    if comparison_df.empty:
        raise fail(
            "No comparable data found between current and baseline.",
            "Both summaries need the same grid columns and a shared duration column.",
        )

    tally = comparison_tally(comparison_df, threshold)
    unmeasured = tally["unmeasured"]
    slower = tally["regressions"] - unmeasured
    console.print(
        f"[dim]Compared[/dim] {tally['compared']} configuration(s)  "
        f"[dim]Faster[/dim] {tally['improvements']}  "
        f"[dim]Slower[/dim] {slower}  "
        f"[dim]Could not be checked[/dim] {unmeasured}"
    )
    if unmeasured:
        keys = comparison_keys(comparison_df)
        console.print(
            f"[red]✗[/red] {unmeasured} configuration(s) the baseline measured "
            "could not be checked (each one fails the comparison):"
        )
        for _, row in comparison_df[comparison_df["problem"] != ""].head(10).iterrows():
            point = ", ".join(f"{k}={row[k]}" for k in keys)
            console.print(f"    {point}  [dim]{row['problem']}[/dim]")

    generate_comparison_report(
        comparison_df=comparison_df,
        title="TempoBench Comparison Report",
        threshold_pct=threshold,
        output_path=output,
        current_name=current.name if current.name != baseline.name else str(current),
        baseline_name=baseline.name if current.name != baseline.name else str(baseline),
    )

    console.print()
    print_artifact("Comparison report", output)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_csv, index=False)
        print_artifact("Comparison CSV", output_csv)

    console.print()
    if tally["regressions"] > 0:
        reasons = []
        if slower:
            reasons.append(f"{slower} slower than the baseline by more than {threshold:g}%")
        if unmeasured:
            reasons.append(f"{unmeasured} measured by the baseline but missing now")
        console.print(
            Panel(
                f"[red bold]⚠ {tally['regressions']} regression(s) detected[/red bold]\n\n"
                + " + ".join(reasons)
                + ".\nReview the comparison report for details.",
                title="Regression Alert",
                border_style="red",
            )
        )
        raise typer.Exit(1)
    console.print(
        Panel(
            "[green bold]✓ No regressions detected[/green bold]\n\n"
            f"All configurations are within {threshold:g}% of the baseline.",
            title="Comparison Passed",
            border_style="green",
        )
    )
