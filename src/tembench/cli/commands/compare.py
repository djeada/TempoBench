"""`tembench compare` — detect regressions vs a baseline summary."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel

from ...reporting import compare_summaries, generate_comparison_report
from ..app import app, console


@app.command()
def compare(
    current: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="Path to current summary CSV"
    ),
    baseline: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="Path to baseline summary CSV"
    ),
    threshold: float = typer.Option(5.0, help="Regression threshold percentage"),
    output: Path = typer.Option(
        Path("artifacts/comparison.html"), help="Output path for comparison report"
    ),
    output_csv: Optional[Path] = typer.Option(
        None, help="Optional path to save comparison CSV"
    ),
):
    """Compare current benchmark results against a baseline to detect regressions.

    [bold]Example:[/bold]
        tembench compare --current artifacts/summary.csv --baseline baseline/summary.csv

    Regressions are flagged when performance degrades by more than the threshold percentage.
    """
    console.print("[bold blue]Comparing Benchmark Results...[/bold blue]")
    console.print(f"[dim]Current:[/dim] {current}")
    console.print(f"[dim]Baseline:[/dim] {baseline}")
    console.print(f"[dim]Threshold:[/dim] {threshold}%")
    console.print()

    comparison_df = compare_summaries(current, baseline, threshold_pct=threshold)

    if comparison_df.empty:
        console.print(
            "[yellow]⚠[/yellow] No comparable data found between current and baseline."
        )
        raise typer.Exit(1)

    # Check for regressions
    regression_cols = [c for c in comparison_df.columns if c.endswith("_regression")]
    total_regressions = 0
    for col in regression_cols:
        total_regressions += comparison_df[col].sum()

    generate_comparison_report(
        comparison_df=comparison_df,
        title="TempoBench Comparison Report",
        threshold_pct=threshold,
        output_path=output,
    )

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_csv, index=False)
        console.print(
            f"[green]✓[/green] Comparison CSV saved to [bold]{output_csv}[/bold]"
        )

    console.print(f"[green]✓[/green] Comparison report saved to [bold]{output}[/bold]")

    if total_regressions > 0:
        console.print()
        console.print(
            Panel(
                f"[red bold]⚠ {int(total_regressions)} regression(s) detected![/red bold]\n\n"
                f"Performance degraded by more than {threshold}% in {int(total_regressions)} configuration(s).\n"
                "Review the comparison report for details.",
                title="Regression Alert",
                border_style="red",
            )
        )
        raise typer.Exit(1)
    else:
        console.print()
        console.print(
            Panel(
                "[green bold]✓ No regressions detected[/green bold]\n\n"
                "All configurations are within the acceptable threshold.",
                title="Comparison Passed",
                border_style="green",
            )
        )
