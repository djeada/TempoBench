"""`tembench inspect` — preview recent runs in a table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table

from ..app import app, console


@app.command()
def inspect(
    runs: Path = typer.Option(
        Path("artifacts/runs.jsonl"),
        exists=True,
        dir_okay=False,
        help="Path to JSONL runs",
    ),
    count: int = typer.Option(10, "--count", "-n", help="Number of runs to show"),
    status: Optional[str] = typer.Option(
        None, help="Filter by status (ok, failed, timeout)"
    ),
):
    """Quickly preview recent runs with detailed statistics.

    [bold]Example:[/bold]
        tembench inspect --runs artifacts/runs.jsonl --count 5
        tembench inspect --status failed
    """
    all_runs = []
    with runs.open() as f:
        for line in f:
            try:
                all_runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not all_runs:
        console.print("[yellow]No runs found in the file.[/yellow]")
        return

    # Filter by status if specified
    filtered_runs = all_runs
    if status:
        filtered_runs = [r for r in all_runs if r.get("status") == status]

    # Show statistics first
    total = len(all_runs)
    ok_count = sum(1 for r in all_runs if r.get("status") == "ok")
    failed_count = sum(1 for r in all_runs if r.get("status") == "failed")
    timeout_count = sum(1 for r in all_runs if r.get("status") == "timeout")

    console.print()
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("", style="dim")
    stats_table.add_column("", style="bold")
    stats_table.add_row("Total Runs", str(total))
    stats_table.add_row("Successful", f"[green]{ok_count}[/green]")
    stats_table.add_row(
        "Failed", f"[red]{failed_count}[/red]" if failed_count > 0 else "0"
    )
    stats_table.add_row(
        "Timeouts", f"[yellow]{timeout_count}[/yellow]" if timeout_count > 0 else "0"
    )

    console.print(Panel(stats_table, title="Run Statistics", border_style="blue"))
    console.print()

    # Show recent runs table
    table = Table(title=f"Recent Runs (last {min(count, len(filtered_runs))})")
    table.add_column("Status", justify="center")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Command", max_width=60)
    table.add_column("Params")

    for rec in filtered_runs[-count:]:
        status_val = rec.get("status", "")
        if status_val == "ok":
            status_display = "[green]✓ ok[/green]"
        elif status_val == "failed":
            status_display = "[red]✗ failed[/red]"
        elif status_val == "timeout":
            status_display = "[yellow]⏱ timeout[/yellow]"
        else:
            status_display = status_val

        wall_ms = rec.get("wall_ms")
        wall_display = f"{wall_ms:.2f}" if wall_ms is not None else "-"

        peak_rss = rec.get("peak_rss_mb")
        mem_display = f"{peak_rss:.2f}" if peak_rss is not None else "-"

        table.add_row(
            status_display,
            wall_display,
            mem_display,
            rec.get("cmd", "")[:60],
            json.dumps(rec.get("params", {})),
        )

    console.print(table)
