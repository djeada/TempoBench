"""`tembench sysinfo` — display system information for reproducibility."""

from __future__ import annotations

from rich.table import Table

from ...reporting import get_system_info
from ..app import app, console


@app.command()
def sysinfo():
    """Display system information for reproducibility."""
    info = get_system_info()

    console.print()
    console.rule("[bold blue]TempoBench System Information[/bold blue]")
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Platform", info["platform"])
    table.add_row("Python", info["python_version"])
    table.add_row("Processor", info["processor"] or "N/A")
    table.add_row("Architecture", info["architecture"])
    table.add_row(
        "CPU Cores",
        f"{info['cpu_count_physical']} physical / {info['cpu_count_logical']} logical",
    )
    if "cpu_freq_mhz" in info:
        table.add_row("CPU Frequency", f"{info['cpu_freq_mhz']} MHz")
    table.add_row("Memory", f"{info['memory_total_gb']} GB")
    table.add_row("Hostname", info["hostname"])
    table.add_row("Timestamp", info["timestamp"])

    console.print(table)
    console.print()
