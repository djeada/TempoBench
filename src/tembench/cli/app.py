"""Shared Typer app and Rich console for TempoBench CLI."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table

from ..config import Config, load_config
from ..runner.core import pinning_problem
from ..summarize import grid_columns, infer_series_column, infer_x_column

app = typer.Typer(
    help=(
        "TempoBench CLI: Language-agnostic benchmarking orchestrator for running commands "
        "with parameter sweeps, recording metrics, and generating reports."
    ),
    rich_markup_mode="rich",
)
console = Console()


class ComplexityStrategy(str, Enum):
    heuristic = "heuristic"
    strict = "strict"


# Options shared by every chart and report command, so each spells them alike.
STRATEGY_OPTION = typer.Option(
    ComplexityStrategy.heuristic,
    "--complexity-strategy",
    help="How aggressively to collapse uncertain exponent bands to canonical Big-O classes",
)
BENCH_OPTION = typer.Option(None, help="Only chart this benchmark (column: bench)")
LOG_X_OPTION = typer.Option(
    None,
    "--log-x/--no-log-x",
    help="Log or linear x axis (default: log when the sizes span more than 30x)",
)
LOG_Y_OPTION = typer.Option(
    None,
    "--log-y/--no-log-y",
    help="Log or linear y axis (default: log when the values span more than 30x)",
)


def output_option(default: str, what: str) -> Any:
    """The output path option, spelled ``--output`` or ``--out-html`` everywhere."""
    return typer.Option(Path(default), "--output", "--out-html", help=f"Output path for the {what}")


def print_heading(heading: str, /, **details: object) -> None:
    """Print a consistent command heading and its most useful inputs."""
    console.rule(f"[bold blue]{heading}[/bold blue]")
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


def select_bench(df: pd.DataFrame, bench: str | None) -> pd.DataFrame:
    """The rows of one benchmark, or every row when `bench` is None."""
    if bench is None:
        return df
    if "bench" not in df.columns:
        raise fail("The summary has no 'bench' column, so --bench cannot filter it.")
    rows = df[df["bench"].astype(str) == bench]
    if rows.empty:
        names = ", ".join(sorted(df["bench"].dropna().astype(str).unique()))
        raise fail(f"No rows for bench {bench!r}.", f"Benchmarks present: {names}")
    return rows.copy()


def require_column(df: pd.DataFrame, column: str, flag: str) -> None:
    """Refuse a column the user named that the summary does not have.

    Quietly drawing something else instead would hand back a chart of a metric
    nobody asked for, under the name of the one they did.
    """
    if column not in df.columns:
        raise fail(
            f"Column {column!r} (from {flag}) is not in the summary.",
            f"Columns present: {', '.join(map(str, df.columns))}",
        )


def resolve_axes(
    df: pd.DataFrame, x: str | None, series: str | None, series_flag: str = "--color"
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

    if series is not None:
        require_column(df, series, series_flag)
        return resolved_x, series
    resolved_series = infer_series_column(df, resolved_x)
    if resolved_series is not None and resolved_series not in df.columns:
        resolved_series = None
    return resolved_x, resolved_series


def print_axes(
    x: str, series: str | None, explicit: bool, series_flag: str = "--color"
) -> None:
    """Tell the user which axes were used when they did not choose them."""
    if explicit:
        return
    console.print(
        f"[dim]Axes[/dim]  x = {x}"
        + (f", series = {series}" if series else ", no series column")
        + f" [dim](inferred; override with --x / {series_flag})[/dim]"
    )


def print_artifact(kind: str, path: Path) -> None:
    """Print a consistent success message for a generated artifact."""
    resolved = path.resolve()
    console.print(f"[green bold]\u2713 {kind} ready[/green bold]")
    console.print(f"  [dim]Path[/dim]  [bold]{resolved}[/bold]")
    if path.suffix.lower() == ".html":
        console.print(f"  [dim]Open[/dim]  file://{resolved}")


def load_config_or_fail(path: Path, workers: int | None = None) -> Config:
    """Load a benchmark config, reporting a bad one as a message, not a traceback.

    `workers` overrides the configured worker count before anything that
    depends on it is checked.
    """
    try:
        cfg = load_config(path)
        if workers is not None:
            cfg.limits.workers = workers
        problem = pinning_problem(cfg)
    except (ValueError, yaml.YAMLError) as e:
        raise fail(f"Invalid config: {e}") from None
    if problem:
        console.print(f"[yellow]![/yellow] {problem}.")
    if cfg.limits.prune_on_timeout and cfg.limits.workers > 1:
        console.print(
            "[yellow]![/yellow] prune_on_timeout only skips the remaining repeats of a "
            "point that timed out when running with several workers; larger inputs "
            "still run."
        )
    return cfg
