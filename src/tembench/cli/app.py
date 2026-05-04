"""Shared Typer app and Rich console for TempoBench CLI."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    help=(
        "TempoBench CLI: Language-agnostic benchmarking orchestrator for running commands "
        "with parameter sweeps, recording metrics, and generating reports."
    ),
    rich_markup_mode="rich",
)
console = Console()
