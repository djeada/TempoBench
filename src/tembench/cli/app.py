"""Shared Typer app and Rich console for TempoBench CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ..summarize import grid_columns, infer_series_column, infer_x_column

app = typer.Typer(
    help=(
        "TempoBench CLI: Language-agnostic benchmarking orchestrator for running commands "
        "with parameter sweeps, recording metrics, and generating reports."
    ),
    rich_markup_mode="rich",
)
console = Console()


def print_heading(title: str, **details: object) -> None:
    """Print a consistent command heading and its most useful inputs."""
    console.rule(f"[bold blue]{title}[/bold blue]")
    if details:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim", no_wrap=True)
        table.add_column(overflow="fold")
        for label, value in details.items():
            table.add_row(label.replace("_", " ").title(), str(value))
        console.print(table)
    console.print()


def fail(message: str, *hints: str) -> typer.Exit:
    """Print an error and return the exception to raise for a non-zero exit.

    Commands must not report success for empty or broken input: TempoBench is
    meant to run in CI, where a green exit code on a benchmark that never ran is
    worse than no benchmark at all.
    """
    console.print(f"[red bold]✗ {message}[/red bold]")
    for hint in hints:
        console.print(f"  [dim]{hint}[/dim]")
    return typer.Exit(code=1)


def load_summary(path: Path) -> pd.DataFrame:
    """Read a summary CSV, refusing to build an artifact out of nothing."""
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if df.empty:
        raise fail(
            f"{path} has no rows.",
            "Produce it from a run with successful trials: tembench summarize --runs <runs.jsonl>",
        )
    return df


def resolve_axes(
    df: pd.DataFrame, x: str | None, series: str | None
) -> tuple[str, str | None]:
    """Settle on the input-size axis and the series axis for a chart.

    Explicit flags always win; otherwise the axes are read off the summary.  The
    grid is user-defined, so defaulting to `n`/`impl` would leave every command
    but `run` and `summarize` working only for sweeps that happen to use the
    bundled examples' names.
    """
    resolved_x = x or infer_x_column(df)
    if resolved_x is None:
        raise fail(
            "Could not tell which column is the input size.",
            f"Pass --x explicitly. Grid columns present: "
            f"{', '.join(grid_columns(df.columns)) or '(none)'}",
        )
    if resolved_x not in df.columns:
        raise fail(
            f"Column {resolved_x!r} is not in the summary.",
            f"Columns present: {', '.join(map(str, df.columns))}",
        )

    resolved_series = series if series is not None else infer_series_column(df, resolved_x)
    if resolved_series is not None and resolved_series not in df.columns:
        resolved_series = None
    return resolved_x, resolved_series


def print_axes(x: str, series: str | None, explicit: bool) -> None:
    """Tell the user which axes were used when they did not choose them."""
    if explicit:
        return
    console.print(
        f"[dim]Axes[/dim]  x = {x}"
        + (f", series = {series}" if series else ", no series column")
        + " [dim](inferred; override with --x / --color)[/dim]"
    )


def print_artifact(kind: str, path: Path) -> None:
    """Print a consistent success message for a generated artifact."""
    resolved = path.resolve()
    console.print(f"[green bold]\u2713 {kind} ready[/green bold]")
    console.print(f"  [dim]Path[/dim]  [bold]{resolved}[/bold]")
    if path.suffix.lower() == ".html":
        console.print(f"  [dim]Open[/dim]  file://{resolved}")
