from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .runner import run_benchmarks
from .summarize import summarize_runs
from .plotting import plot_runtime
from .complexity import fit_models

app = typer.Typer(help="TempoBench CLI: run benchmarks, summarize, and plot.")
console = Console()


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to YAML config"),
    out_dir: Path = typer.Option(Path("artifacts"), help="Directory for artifacts"),
    seed: int = typer.Option(42, help="Random seed for sweep order"),
    retries: int = typer.Option(0, help="Retries per failed repetition"),
):
    """Execute configured benchmarks and write JSONL results."""
    cfg = load_config(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "runs.jsonl"
    console.rule("Running benchmarks")
    run_benchmarks(cfg, results_path, seed=seed, retries=retries)
    console.print(f"Wrote runs to {results_path}")


@app.command()
def summarize(
    runs: Path = typer.Option(Path("artifacts/runs.jsonl"), exists=True, dir_okay=False),
    out_csv: Path = typer.Option(Path("artifacts/summary.csv"), dir_okay=False),
    include_outliers: bool = typer.Option(False, help="Include outliers in medians/means"),
):
    """Summarize JSONL runs into CSV with medians and percentiles."""
    df = summarize_runs(runs, include_outliers=include_outliers)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    console.print(f"Wrote summary to {out_csv}")


@app.command()
def plot(
    summary: Path = typer.Option(Path("artifacts/summary.csv"), exists=True, dir_okay=False),
    x: str = typer.Option("n", help="X axis parameter"),
    y: str = typer.Option("wall_ms_median", help="Y axis metric"),
    color: str = typer.Option("impl", help="Series grouping column"),
    out_html: Optional[Path] = typer.Option(Path("artifacts/runtime.html")),
    no_fit: bool = typer.Option(False, help="Disable Big-O fit overlay"),
    export_fits: Optional[Path] = typer.Option(None, help="Optional path to save fitted models CSV"),
    log_x: bool = typer.Option(False, help="Use log scale for X axis"),
    log_y: bool = typer.Option(False, help="Use log scale for Y axis"),
):
    """Create a simple runtime plot from the summary CSV."""
    chart = plot_runtime(summary, x=x, y=y, color=color, show_fit=not no_fit, log_x=log_x, log_y=log_y)
    if out_html:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        chart.save(out_html)
        console.print(f"Wrote plot to {out_html}")
    else:
        # Print Vega-Lite JSON to stdout for piping
        json.dump(chart.to_dict(), sys.stdout)
    if export_fits:
        import pandas as pd
        df = pd.read_csv(summary)
        by = [c for c in [color] if c in df.columns]
        fits = fit_models(df, x_col=x, y_col=y if y in df.columns else ("wall_ms_median" if "wall_ms_median" in df.columns else "wall_ms_mean"), by=by or [color])
        fits.to_csv(export_fits, index=False)
        console.print(f"Wrote fits to {export_fits}")


@app.command()
def inspect(runs: Path = typer.Option(Path("artifacts/runs.jsonl"), exists=True, dir_okay=False)):
    """Quickly preview the last few runs."""
    table = Table(title="Recent Runs")
    columns = ["ts", "status", "cmd", "params", "wall_ms", "peak_rss_mb"]
    for c in columns:
        table.add_column(c)
    last = []
    with runs.open() as f:
        for line in f:
            last.append(json.loads(line))
    for rec in last[-10:]:
        table.add_row(
            str(rec.get("ts")),
            rec.get("status", ""),
            rec.get("cmd", "")[:80],
            json.dumps(rec.get("params", {})),
            f"{rec.get('wall_ms', None)}",
            f"{rec.get('peak_rss_mb', None)}",
        )
    console.print(table)


@app.command()
def report():
    """Stub: assemble HTML report (coming soon)."""
    console.print("Report generation is not implemented yet.")


@app.command()
def compare():
    """Stub: compare summary against a baseline (coming soon)."""
    console.print("Compare is not implemented yet.")


if __name__ == "__main__":
    app()
