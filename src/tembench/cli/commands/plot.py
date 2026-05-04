"""`tembench plot` — runtime chart with optional Big-O fit overlay."""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from ...complexity import fit_models
from ...plotting import plot_runtime
from ..app import app, console


class ComplexityStrategy(str, Enum):
    heuristic = "heuristic"
    strict = "strict"


@app.command()
def plot(
    summary: Path = typer.Option(
        Path("artifacts/summary.csv"), exists=True, dir_okay=False
    ),
    x: str = typer.Option("n", help="X axis parameter"),
    y: str = typer.Option("wall_ms_median", help="Y axis metric"),
    color: str = typer.Option("impl", help="Series grouping column"),
    bench: Optional[str] = typer.Option(
        None, help="Optional benchmark name filter (column: bench)"
    ),
    out_html: Optional[Path] = typer.Option(Path("artifacts/runtime.html")),
    no_fit: bool = typer.Option(False, help="Disable Big-O fit overlay"),
    export_fits: Optional[Path] = typer.Option(
        None, help="Optional path to save fitted models CSV"
    ),
    complexity_strategy: ComplexityStrategy = typer.Option(
        ComplexityStrategy.heuristic,
        "--complexity-strategy",
        help="How aggressively to collapse uncertain exponent bands to canonical Big-O classes",
    ),
    log_x: bool = typer.Option(False, help="Use log scale for X axis"),
    log_y: bool = typer.Option(False, help="Use log scale for Y axis"),
):
    """Create a simple runtime plot from the summary CSV."""
    chart = plot_runtime(
        summary,
        x=x,
        y=y,
        color=color,
        bench=bench,
        show_fit=not no_fit,
        complexity_strategy=complexity_strategy.value,
        log_x=log_x,
        log_y=log_y,
    )
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
        by = [c for c in ["bench", color] if c in df.columns]
        y_fit = (
            y
            if y in df.columns
            else (
                "wall_ms_median" if "wall_ms_median" in df.columns else "wall_ms_mean"
            )
        )
        fits = fit_models(
            df,
            x_col=x,
            y_col=y_fit,
            by=by or [color],
            strategy=complexity_strategy.value,
        )
        fits.to_csv(export_fits, index=False)
        console.print(f"Wrote fits to {export_fits}")
