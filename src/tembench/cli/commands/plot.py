"""`tembench plot` — runtime chart with optional Big-O fit overlay."""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from ...complexity import fit_models
from ...plotting import plot_runtime
from ...summarize import (
    TIME_COLUMN_PREFERENCE,
    count_column_for,
    preferred_time_column,
    spread_columns_for,
)
from ..app import (
    app,
    console,
    fail,
    load_summary,
    print_artifact,
    print_axes,
    print_heading,
    resolve_axes,
)


class ComplexityStrategy(str, Enum):
    heuristic = "heuristic"
    strict = "strict"


_CONFIDENCE_STYLE = {"high": "green", "medium": "yellow", "low": "red"}


def print_fits(fits, by: list[str]) -> None:
    """Show each fitted class next to how much the data supports it.

    The complexity class is the headline output of TempoBench, so it belongs in
    the terminal rather than only inside the generated HTML — and it is never
    shown without its confidence, because a class fitted to four noisy points
    looks exactly like one fitted to a clean decade of input sizes.
    """
    table = Table(title="Complexity fits", box=None, title_style="bold")
    for column in by:
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
        if rival and "another class" in str(row.get("confidence_notes", "")):
            klass = f"{klass} [dim]≈[/dim] {rival}"
        table.add_row(
            *[str(row[c]) for c in by],
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
    y: str = typer.Option("time_ms_median", help="Y axis metric"),
    color: Optional[str] = typer.Option(
        None, help="Series grouping column (default: inferred)"
    ),
    bench: Optional[str] = typer.Option(
        None, help="Optional benchmark name filter (column: bench)"
    ),
    out_html: Optional[Path] = typer.Option(
        Path("artifacts/runtime.html"),
        help="Output HTML path, or '-' to write the Vega-Lite JSON to stdout",
    ),
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
    # Without a file to write, the chart JSON owns stdout, so nothing
    # human-readable may be written there or the pipe is corrupted.
    piping = out_html is None or str(out_html) == "-"
    if piping:
        out_html = None

    df = load_summary(summary)
    explicit_axes = x is not None and color is not None
    x, color = resolve_axes(df, x, color)
    if y not in df.columns and preferred_time_column(df.columns) is None:
        raise fail(
            f"{summary} has no duration column to plot.",
            f"Expected --y to name a column, or one of: {', '.join(TIME_COLUMN_PREFERENCE)}.",
            f"Columns present: {', '.join(map(str, df.columns))}",
        )
    if out_html:
        print_heading(
            "Generating Runtime Plot",
            summary=summary,
            axes=f"{x} \u2192 {y}",
            series=color or "(none)",
            complexity_fit="off" if no_fit else complexity_strategy.value,
        )
        print_axes(x, color, explicit_axes)
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
        print_artifact("Runtime plot", out_html)
    else:
        # Print Vega-Lite JSON to stdout for piping
        json.dump(chart.to_dict(), sys.stdout)
    if no_fit:
        return

    by = [c for c in ["bench", color] if c and c in df.columns]
    y_fit = y if y in df.columns else preferred_time_column(df.columns)
    assert y_fit is not None  # guaranteed by the duration-column check above
    if not by:
        # A summary with a single unnamed series still deserves a fit.
        df = df.assign(_series="all")
        by = ["_series"]
    fits = fit_models(
        df,
        x_col=x,
        y_col=y_fit,
        by=by,
        strategy=complexity_strategy.value,
        count_col=count_column_for(y_fit),
        spread_cols=spread_columns_for(y_fit),
    )

    if not piping and not fits.empty:
        console.print()
        print_fits(fits, by)

    if export_fits:
        export_fits.parent.mkdir(parents=True, exist_ok=True)
        fits.to_csv(export_fits, index=False)
        if not piping:
            console.print()
            print_artifact("Complexity fits", export_fits)
