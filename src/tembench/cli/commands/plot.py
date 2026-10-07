"""`tembench plot` — runtime chart with optional Big-O fit overlay."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.table import Table

from ...plotting import fit_runtime, plot_runtime, save_chart
from ...plotting._common import SINGLE_SERIES, default_series
from ...summarize import TIME_COLUMN_PREFERENCE, preferred_time_column
from ..app import (
    BENCH_OPTION,
    LOG_X_OPTION,
    LOG_Y_OPTION,
    STRATEGY_OPTION,
    ComplexityStrategy,
    app,
    console,
    fail,
    load_summary,
    print_artifact,
    print_axes,
    print_heading,
    require_column,
    resolve_axes,
    select_bench,
)

_CONFIDENCE_STYLE = {"high": "green", "medium": "yellow", "low": "red"}


def print_fits(fits: pd.DataFrame, by: list[str]) -> None:
    """Show each fitted class next to how much the data supports it.

    The complexity class is the headline output of TempoBench, so it belongs in
    the terminal rather than only inside the generated HTML — and it is never
    shown without its confidence, because a class fitted to four noisy points
    looks exactly like one fitted to a clean decade of input sizes.
    """
    table = Table(title="Complexity fits", box=None, title_style="bold")
    keys = [c for c in by if c != SINGLE_SERIES]
    for column in keys:
        table.add_column(str(column).title())
    table.add_column("Class")
    table.add_column("Upper bound", overflow="fold")
    table.add_column("Confidence")
    table.add_column("Caveats", overflow="fold")

    for _, row in fits.iterrows():
        confidence = str(row.get("confidence", "")) or "-"
        style = _CONFIDENCE_STYLE.get(confidence, "dim")
        klass = str(row.get("display_model", row["model"]))
        # When a rival class explains the data about as well, naming it is more
        # useful than the winner alone — it says which way the answer might go.
        rival = row.get("runner_up")
        if rival and "ambiguous-class" in str(row.get("caveats", "")).split(","):
            klass = f"{klass} [dim]≈[/dim] {rival}"
        table.add_row(
            *[str(row[c]) for c in keys],
            klass,
            str(row["formula"]),
            f"[{style}]{confidence}[/{style}]",
            str(row.get("confidence_notes", "") or "—"),
        )
    console.print(table)

    if "confidence" not in fits.columns:
        return
    ratings = fits["confidence"].tolist()
    if "low" in ratings:
        console.print(
            "\n[red]The measurements do not establish these classes.[/red] "
            "Widen the input-size range, add repeats, or have the command report "
            "its own timing so process startup is excluded."
        )
    elif "medium" in ratings:
        console.print(
            "\n[yellow]Some classes rest on weak evidence[/yellow] — see the caveats above."
        )


@app.command()
def plot(
    summary: Path = typer.Option(
        Path("artifacts/summary.csv"), exists=True, dir_okay=False
    ),
    x: Optional[str] = typer.Option(None, help="X axis parameter (default: inferred)"),
    y: Optional[str] = typer.Option(
        None, help="Y axis metric (default: the summary's preferred duration column)"
    ),
    color: Optional[str] = typer.Option(
        None, help="Series grouping column (default: inferred)"
    ),
    bench: Optional[str] = BENCH_OPTION,
    out_html: Optional[Path] = typer.Option(
        Path("artifacts/runtime.html"),
        "--output",
        "--out-html",
        help="Output HTML path, or '-' to write the Vega-Lite JSON to stdout",
    ),
    no_fit: bool = typer.Option(False, help="Disable Big-O fit overlay"),
    export_fits: Optional[Path] = typer.Option(
        None, help="Optional path to save fitted models CSV"
    ),
    complexity_strategy: ComplexityStrategy = STRATEGY_OPTION,
    log_x: Optional[bool] = LOG_X_OPTION,
    log_y: Optional[bool] = LOG_Y_OPTION,
):
    """Create a runtime plot, with each series' fitted complexity class."""
    # Without a file to write, the chart JSON owns stdout, so nothing
    # human-readable may be written there or the pipe is corrupted.
    piping = out_html is None or str(out_html) == "-"
    if piping:
        out_html = None

    df = select_bench(load_summary(summary), bench)
    explicit_axes = x is not None and color is not None
    x, color = resolve_axes(df, x, color)
    if y is not None:
        require_column(df, y, "--y")
    else:
        y = preferred_time_column(df.columns)
        if y is None:
            raise fail(
                f"{summary} has no duration column to plot.",
                f"Pass --y to name a column, or provide one of: "
                f"{', '.join(TIME_COLUMN_PREFERENCE)}.",
                f"Columns present: {', '.join(map(str, df.columns))}",
            )

    # Fitted once: the chart draws, the table shows and the CSV holds the
    # same fits, with the same bench filter and series grouping.
    fits, by = (
        (pd.DataFrame(), [])
        if no_fit
        else fit_runtime(df, x=x, y=y, color=color, complexity_strategy=complexity_strategy.value)
    )
    if out_html:
        series = default_series(df, color)
        print_heading(
            "Runtime Plot",
            summary=summary,
            axes=f"{x} \u2192 {y}",
            series=series or "(single series)",
            complexity_fit="off"
            if no_fit
            else f"{complexity_strategy.value}, one per {' × '.join(c for c in by if c != SINGLE_SERIES) or 'summary'}",
        )
        print_axes(x, color, explicit_axes)
    chart = plot_runtime(
        df,
        x=x,
        y=y,
        color=color,
        show_fit=not no_fit,
        complexity_strategy=complexity_strategy.value,
        log_x=log_x,
        log_y=log_y,
        fits=fits,
    )
    if out_html:
        save_chart(chart, out_html, title=f"Runtime: {bench}" if bench else "Runtime", meta=str(summary))
        print_artifact("Runtime plot", out_html)
    else:
        json.dump(chart.to_dict(), sys.stdout)
    if no_fit:
        return

    if not piping and not fits.empty:
        console.print()
        print_fits(fits, by)

    if export_fits:
        export_fits.parent.mkdir(parents=True, exist_ok=True)
        fits.to_csv(export_fits, index=False)
        if not piping:
            console.print()
            print_artifact("Complexity fits", export_fits)
