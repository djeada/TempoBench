"""`tembench inspect` — preview recent runs in a table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table

from ...summarize import read_jsonl
from ..app import app, console


@app.command()
def inspect(
    runs: Path = typer.Option(
        Path("artifacts/runs.jsonl"),
        exists=True,
        dir_okay=False,
        help="Path to JSONL runs",
    ),
    count: int = typer.Option(
        10, "--count", "-n", min=1, help="Number of runs to show"
    ),
    status: Optional[str] = typer.Option(
        None, help="Filter by status (ok, failed, error, timeout, skipped)"
    ),
):
    """Quickly preview recent runs with detailed statistics.

    [bold]Example:[/bold]
        tembench inspect --runs artifacts/runs.jsonl --count 5
        tembench inspect --status failed
    """
    all_runs = read_jsonl(runs)
    if not all_runs:
        console.print("[yellow]No runs found in the file.[/yellow]")
        return

    filtered_runs = [r for r in all_runs if r.get("status") == status] if status else all_runs

    counts: dict[str, int] = {}
    for rec in all_runs:
        counts[str(rec.get("status"))] = counts.get(str(rec.get("status")), 0) + 1

    console.print()
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("", style="dim")
    stats_table.add_column("", style="bold")
    stats_table.add_row("Total Runs", str(len(all_runs)))
    stats_table.add_row("Successful", f"[green]{counts.get('ok', 0)}[/green]")
    for label, key, style in (
        ("Failed", "failed", "red"),
        ("Errors", "error", "red"),
        ("Timeouts", "timeout", "yellow"),
        ("Skipped", "skipped", "yellow"),
    ):
        value = counts.get(key, 0)
        stats_table.add_row(label, f"[{style}]{value}[/{style}]" if value else "0")

    console.print(Panel(stats_table, title="Run Statistics", border_style="blue"))
    console.print()

    # Show recent runs table
    shown = filtered_runs[-count:]

    table = Table(title=f"Recent Runs (last {len(shown)})")
    table.add_column("Status", justify="center")
    table.add_column("Wall (ms)", justify="right")
    table.add_column("Reported (ms)", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Command", max_width=40)
    table.add_column("Params")

    for rec in shown:
        status_val = rec.get("status", "")
        status_display = {
            "ok": "[green]✓ ok[/green]",
            "failed": "[red]✗ failed[/red]",
            "error": "[red]✗ error[/red]",
            "timeout": "[yellow]⏱ timeout[/yellow]",
            "skipped": "[yellow]⊘ skipped[/yellow]",
        }.get(status_val, status_val)

        wall_ms = rec.get("wall_ms")
        reported_ms = rec.get("reported_ms")
        peak_rss = rec.get("peak_rss_mb")

        table.add_row(
            status_display,
            f"{wall_ms:.2f}" if wall_ms is not None else "-",
            f"{reported_ms:.3f}" if reported_ms is not None else "-",
            f"{peak_rss:.2f}" if peak_rss is not None else "-",
            rec.get("cmd", "")[:40],
            json.dumps(rec.get("params", {})),
        )

    console.print(table)

    # Diagnosing a failure is the main reason to reach for `inspect`, so the
    # command's own message gets the full width of its own section rather than
    # being squeezed into a table cell and wrapped mid-word.
    messages: dict[str, int] = {}
    for rec in shown:
        if rec.get("status") == "ok":
            continue
        text = str(rec.get("stderr") or rec.get("stdout") or "").strip()
        if text:
            line = text.splitlines()[-1].strip()
            messages[line] = messages.get(line, 0) + 1

    if messages:
        console.print()
        console.print("[red bold]Messages from unsuccessful trials[/red bold]")
        for line, occurrences in sorted(messages.items(), key=lambda item: -item[1]):
            suffix = f" [dim](x{occurrences})[/dim]" if occurrences > 1 else ""
            console.print(f"  {line}{suffix}")
