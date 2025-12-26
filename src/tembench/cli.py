from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text

from .config import load_config
from .runner import run_benchmarks
from .summarize import summarize_runs
from .plotting import (
    plot_runtime,
    plot_memory,
    plot_heatmap,
    plot_boxplot,
    create_dashboard,
    save_chart,
)
from .complexity import fit_models
from .reporting import (
    generate_report,
    compare_summaries,
    generate_comparison_report,
    get_system_info,
)

app = typer.Typer(
    help="TempoBench CLI: Language-agnostic benchmarking orchestrator for running commands with parameter sweeps, recording metrics, and generating reports.",
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to YAML config"),
    out_dir: Path = typer.Option(Path("artifacts"), help="Directory for artifacts"),
    seed: int = typer.Option(42, help="Random seed for sweep order"),
    retries: int = typer.Option(0, help="Retries per failed repetition"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
):
    """Execute configured benchmarks and write JSONL results.
    
    [bold]Example:[/bold]
        tembench run --config examples/sort_bench.yaml --out-dir artifacts
    """
    cfg = load_config(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "runs.jsonl"
    
    if not quiet:
        console.rule("[bold blue]TempoBench - Running Benchmarks[/bold blue]")
        console.print(f"[dim]Config:[/dim] {config}")
        console.print(f"[dim]Output:[/dim] {out_dir}")
        console.print()
    
    run_benchmarks(cfg, results_path, seed=seed, retries=retries)
    
    if not quiet:
        console.print()
        console.print(f"[green]✓[/green] Wrote runs to [bold]{results_path}[/bold]")
        console.print()
        console.print("[dim]Next steps:[/dim]")
        console.print(f"  tembench summarize --runs {results_path}")
        console.print(f"  tembench report --summary {out_dir / 'summary.csv'}")


@app.command()
def summarize(
    runs: Path = typer.Option(Path("artifacts/runs.jsonl"), exists=True, dir_okay=False),
    out_csv: Path = typer.Option(Path("artifacts/summary.csv"), dir_okay=False),
    include_outliers: bool = typer.Option(False, help="Include outliers in medians/means"),
):
    """Summarize JSONL runs into CSV with medians and percentiles."""
    df = summarize_runs(runs, include_outliers=include_outliers)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    console.print(f"Wrote summary to {out_csv}")


@app.command()
def plot(
    summary: Path = typer.Option(Path("artifacts/summary.csv"), exists=True, dir_okay=False),
    x: str = typer.Option("n", help="X axis parameter"),
    y: str = typer.Option("wall_ms_median", help="Y axis metric"),
    color: str = typer.Option("impl", help="Series grouping column"),
    out_html: Optional[Path] = typer.Option(Path("artifacts/runtime.html")),
    no_fit: bool = typer.Option(False, help="Disable Big-O fit overlay"),
    export_fits: Optional[Path] = typer.Option(None, help="Optional path to save fitted models CSV"),
    log_x: bool = typer.Option(False, help="Use log scale for X axis"),
    log_y: bool = typer.Option(False, help="Use log scale for Y axis"),
):
    """Create a simple runtime plot from the summary CSV."""
    chart = plot_runtime(summary, x=x, y=y, color=color, show_fit=not no_fit, log_x=log_x, log_y=log_y)
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
        by = [c for c in [color] if c in df.columns]
        fits = fit_models(df, x_col=x, y_col=y if y in df.columns else ("wall_ms_median" if "wall_ms_median" in df.columns else "wall_ms_mean"), by=by or [color])
        fits.to_csv(export_fits, index=False)
        console.print(f"Wrote fits to {export_fits}")


@app.command()
def inspect(
    runs: Path = typer.Option(Path("artifacts/runs.jsonl"), exists=True, dir_okay=False, help="Path to JSONL runs"),
    count: int = typer.Option(10, "--count", "-n", help="Number of runs to show"),
    status: Optional[str] = typer.Option(None, help="Filter by status (ok, failed, timeout)"),
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
    stats_table.add_row("Failed", f"[red]{failed_count}[/red]" if failed_count > 0 else "0")
    stats_table.add_row("Timeouts", f"[yellow]{timeout_count}[/yellow]" if timeout_count > 0 else "0")
    
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


@app.command()
def report(
    summary: Path = typer.Option(Path("artifacts/summary.csv"), exists=True, dir_okay=False, help="Path to summary CSV"),
    runs: Optional[Path] = typer.Option(None, help="Path to raw JSONL runs (optional)"),
    fits: Optional[Path] = typer.Option(None, help="Path to complexity fits CSV (optional)"),
    chart: Optional[Path] = typer.Option(None, help="Path to pre-generated chart HTML (optional)"),
    output: Path = typer.Option(Path("artifacts/report.html"), help="Output path for HTML report"),
    title: str = typer.Option("TempoBench Report", help="Report title"),
):
    """Generate a comprehensive HTML report with charts, tables, and system info.
    
    [bold]Example:[/bold]
        tembench report --summary artifacts/summary.csv --output artifacts/report.html
        
    The report includes:
    - Performance overview with key statistics
    - Run success/failure counts (if runs.jsonl provided)
    - Embedded charts (if chart.html provided)
    - Detailed results table
    - Complexity analysis (if fits.csv provided)
    - System information for reproducibility
    """
    console.print("[bold blue]Generating TempoBench Report...[/bold blue]")
    
    # Auto-detect optional files if not provided
    if runs is None:
        default_runs = summary.parent / "runs.jsonl"
        if default_runs.exists():
            runs = default_runs
            console.print(f"[dim]Auto-detected runs:[/dim] {runs}")
    
    if fits is None:
        default_fits = summary.parent / "fits.csv"
        if default_fits.exists():
            fits = default_fits
            console.print(f"[dim]Auto-detected fits:[/dim] {fits}")
    
    if chart is None:
        default_chart = summary.parent / "runtime.html"
        if default_chart.exists():
            chart = default_chart
            console.print(f"[dim]Auto-detected chart:[/dim] {chart}")
    
    generate_report(
        summary_csv=summary,
        runs_jsonl=runs,
        fits_csv=fits,
        chart_html=chart,
        title=title,
        output_path=output,
    )
    
    console.print()
    console.print(f"[green]✓[/green] Report saved to [bold]{output}[/bold]")
    console.print(f"[dim]Open in browser: file://{output.absolute()}[/dim]")


@app.command()
def compare(
    current: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to current summary CSV"),
    baseline: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to baseline summary CSV"),
    threshold: float = typer.Option(5.0, help="Regression threshold percentage"),
    output: Path = typer.Option(Path("artifacts/comparison.html"), help="Output path for comparison report"),
    output_csv: Optional[Path] = typer.Option(None, help="Optional path to save comparison CSV"),
):
    """Compare current benchmark results against a baseline to detect regressions.
    
    [bold]Example:[/bold]
        tembench compare --current artifacts/summary.csv --baseline baseline/summary.csv
        
    Regressions are flagged when performance degrades by more than the threshold percentage.
    The comparison report shows:
    - Overall regression status
    - Side-by-side current vs baseline values
    - Delta percentages with color coding
    - Regression flags for quick identification
    """
    console.print("[bold blue]Comparing Benchmark Results...[/bold blue]")
    console.print(f"[dim]Current:[/dim] {current}")
    console.print(f"[dim]Baseline:[/dim] {baseline}")
    console.print(f"[dim]Threshold:[/dim] {threshold}%")
    console.print()
    
    comparison_df = compare_summaries(current, baseline, threshold_pct=threshold)
    
    if comparison_df.empty:
        console.print("[yellow]⚠[/yellow] No comparable data found between current and baseline.")
        raise typer.Exit(1)
    
    # Check for regressions
    regression_cols = [c for c in comparison_df.columns if c.endswith('_regression')]
    total_regressions = 0
    for col in regression_cols:
        total_regressions += comparison_df[col].sum()
    
    # Generate report
    generate_comparison_report(
        comparison_df=comparison_df,
        title="TempoBench Comparison Report",
        threshold_pct=threshold,
        output_path=output,
    )
    
    # Optionally save CSV
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_csv, index=False)
        console.print(f"[green]✓[/green] Comparison CSV saved to [bold]{output_csv}[/bold]")
    
    console.print(f"[green]✓[/green] Comparison report saved to [bold]{output}[/bold]")
    
    if total_regressions > 0:
        console.print()
        console.print(Panel(
            f"[red bold]⚠ {int(total_regressions)} regression(s) detected![/red bold]\n\n"
            f"Performance degraded by more than {threshold}% in {int(total_regressions)} configuration(s).\n"
            "Review the comparison report for details.",
            title="Regression Alert",
            border_style="red",
        ))
        raise typer.Exit(1)
    else:
        console.print()
        console.print(Panel(
            "[green bold]✓ No regressions detected[/green bold]\n\n"
            "All configurations are within the acceptable threshold.",
            title="Comparison Passed",
            border_style="green",
        ))


@app.command()
def dashboard(
    summary: Path = typer.Option(Path("artifacts/summary.csv"), exists=True, dir_okay=False, help="Path to summary CSV"),
    runs: Optional[Path] = typer.Option(None, help="Path to raw JSONL runs (optional, for boxplots)"),
    x: str = typer.Option("n", help="X axis parameter"),
    color: str = typer.Option("impl", help="Series grouping column"),
    output: Path = typer.Option(Path("artifacts/dashboard.html"), help="Output path for dashboard"),
    title: str = typer.Option("TempoBench Dashboard", help="Dashboard title"),
):
    """Generate an interactive dashboard with multiple charts.
    
    [bold]Example:[/bold]
        tembench dashboard --summary artifacts/summary.csv --output artifacts/dashboard.html
        
    The dashboard includes:
    - Runtime vs input size chart with complexity fits
    - Memory usage chart (if memory data available)
    - Performance heatmap
    - Distribution boxplot (if raw runs provided)
    """
    console.print("[bold blue]Generating TempoBench Dashboard...[/bold blue]")
    
    # Auto-detect runs if not provided
    if runs is None:
        default_runs = summary.parent / "runs.jsonl"
        if default_runs.exists():
            runs = default_runs
            console.print(f"[dim]Auto-detected runs:[/dim] {runs}")
    
    dashboard_chart = create_dashboard(
        summary_csv=summary,
        runs_jsonl=runs,
        x=x,
        color=color,
        title=title,
    )
    
    output.parent.mkdir(parents=True, exist_ok=True)
    dashboard_chart.save(output)
    
    console.print()
    console.print(f"[green]✓[/green] Dashboard saved to [bold]{output}[/bold]")
    console.print(f"[dim]Open in browser: file://{output.absolute()}[/dim]")


@app.command()
def sysinfo():
    """Display system information for reproducibility.
    
    Shows details about the current system that affect benchmark results:
    - Platform and OS version
    - CPU model, cores, and frequency
    - Available memory
    - Python version
    """
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
    table.add_row("CPU Cores", f"{info['cpu_count_physical']} physical / {info['cpu_count_logical']} logical")
    if "cpu_freq_mhz" in info:
        table.add_row("CPU Frequency", f"{info['cpu_freq_mhz']} MHz")
    table.add_row("Memory", f"{info['memory_total_gb']} GB")
    table.add_row("Hostname", info["hostname"])
    table.add_row("Timestamp", info["timestamp"])
    
    console.print(table)
    console.print()


@app.command()
def memory(
    summary: Path = typer.Option(Path("artifacts/summary.csv"), exists=True, dir_okay=False, help="Path to summary CSV"),
    x: str = typer.Option("n", help="X axis parameter"),
    color: str = typer.Option("impl", help="Series grouping column"),
    output: Path = typer.Option(Path("artifacts/memory.html"), help="Output path for memory chart"),
    log_x: bool = typer.Option(False, help="Use log scale for X axis"),
    log_y: bool = typer.Option(False, help="Use log scale for Y axis"),
):
    """Generate a memory usage chart from summary data.
    
    [bold]Example:[/bold]
        tembench memory --summary artifacts/summary.csv --output artifacts/memory.html
    """
    chart = plot_memory(summary, x=x, color=color, log_x=log_x, log_y=log_y)
    
    output.parent.mkdir(parents=True, exist_ok=True)
    chart.save(output)
    
    console.print(f"[green]✓[/green] Memory chart saved to [bold]{output}[/bold]")


@app.command()
def heatmap(
    summary: Path = typer.Option(Path("artifacts/summary.csv"), exists=True, dir_okay=False, help="Path to summary CSV"),
    x: str = typer.Option("n", help="X axis parameter"),
    y: str = typer.Option("impl", help="Y axis parameter"),
    value: str = typer.Option("wall_ms_median", help="Value to display in cells"),
    output: Path = typer.Option(Path("artifacts/heatmap.html"), help="Output path for heatmap"),
):
    """Generate a performance heatmap from summary data.
    
    [bold]Example:[/bold]
        tembench heatmap --summary artifacts/summary.csv --output artifacts/heatmap.html
        
    Creates a heatmap showing performance values across two dimensions.
    """
    chart = plot_heatmap(summary, x=x, y=y, value=value)
    
    output.parent.mkdir(parents=True, exist_ok=True)
    chart.save(output)
    
    console.print(f"[green]✓[/green] Heatmap saved to [bold]{output}[/bold]")


if __name__ == "__main__":
    app()
