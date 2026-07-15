"""Shared Typer app and Rich console for TempoBench CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

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


def print_artifact(kind: str, path: Path) -> None:
    """Print a consistent success message for a generated artifact."""
    resolved = path.resolve()
    console.print(f"[green bold]\u2713 {kind} ready[/green bold]")
    console.print(f"  [dim]Path[/dim]  [bold]{resolved}[/bold]")
    if path.suffix.lower() == ".html":
        console.print(f"  [dim]Open[/dim]  file://{resolved}")
