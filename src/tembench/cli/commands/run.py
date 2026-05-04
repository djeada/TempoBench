"""`tembench run` — execute configured benchmarks and write JSONL results."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from ...config import load_config
from ...runner import expand_grid, run_benchmarks
from ..app import app, console


@app.command()
def run(
    config: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="Path to YAML config"
    ),
    out_dir: Path = typer.Option(Path("artifacts"), help="Directory for artifacts"),
    seed: int = typer.Option(42, help="Random seed for sweep order"),
    retries: int = typer.Option(0, help="Retries per failed repetition"),
    append: bool = typer.Option(
        False, help="Append to existing runs.jsonl instead of overwriting it"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
    workers: int = typer.Option(
        0, "--workers", "-j", help="Parallel workers (0 = use config value, default 1)"
    ),
):
    """Execute configured benchmarks and write JSONL results.

    [bold]Example:[/bold]
        tembench run --config examples/unique_bench.yaml --out-dir artifacts
    """
    cfg = load_config(config)
    # CLI --workers overrides config; 0 means use config value
    if workers > 0:
        cfg.limits.workers = workers
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "runs.jsonl"

    if not quiet:
        console.rule("[bold blue]TempoBench - Running Benchmarks[/bold blue]")
        console.print(f"[dim]Config:[/dim] {config}")
        console.print(f"[dim]Output:[/dim] {out_dir}")
        if cfg.limits.workers > 1:
            console.print(
                f"[dim]Workers:[/dim] {cfg.limits.workers}  [yellow]⚠ parallel mode — timings may have cross-talk[/yellow]"
            )
        console.print()

    total_trials = (
        len(cfg.benchmarks) * len(expand_grid(cfg.grid)) * max(1, cfg.limits.repeats)
    )

    if not quiet:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )
        task_id = progress.add_task("Running benchmarks", total=total_trials)

        def on_trial(bench_name, params, rep, total_reps, result):
            label = f"[dim]{bench_name}[/dim] {params}"
            progress.update(task_id, advance=1, description=label)

        with progress:
            run_benchmarks(
                cfg,
                results_path,
                seed=seed,
                retries=retries,
                on_trial=on_trial,
                append=append,
            )
    else:
        run_benchmarks(cfg, results_path, seed=seed, retries=retries, append=append)

    if not quiet:
        console.print()
        console.print(f"[green]✓[/green] Wrote runs to [bold]{results_path}[/bold]")
        console.print()
        console.print("[dim]Next steps:[/dim]")
        console.print(f"  tembench summarize --runs {results_path}")
        console.print(f"  tembench report --summary {out_dir / 'summary.csv'}")
