"""`tembench run` — execute configured benchmarks and write JSONL results."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from ...config import load_config
from ...runner import expand_grid, run_benchmarks
from ..app import app, console, print_artifact, print_heading


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
    initial_size = results_path.stat().st_size if append and results_path.exists() else 0

    started = time.perf_counter()
    if not quiet:
        points = len(expand_grid(cfg.grid))
        print_heading(
            "Running Benchmarks",
            config=config,
            output=out_dir,
            plan=f"{len(cfg.benchmarks)} benchmark(s) x {points} grid point(s) x {max(1, cfg.limits.repeats)} repeat(s)",
            workers=cfg.limits.workers,
        )
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
        elapsed = time.perf_counter() - started
        statuses: dict[str, int] = {}
        measured_ms = 0.0
        with results_path.open() as handle:
            handle.seek(initial_size)
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                status = str(record.get("status", "unknown"))
                statuses[status] = statuses.get(status, 0) + 1
                if record.get("wall_ms") is not None:
                    measured_ms += float(record["wall_ms"])

        console.print()
        summary = Table(title="Run complete", box=None)
        summary.add_column("Successful", justify="right", style="green bold")
        summary.add_column("Failed", justify="right", style="red bold")
        summary.add_column("Errors", justify="right", style="red bold")
        summary.add_column("Timed out", justify="right", style="yellow bold")
        summary.add_column("Skipped", justify="right", style="yellow")
        summary.add_column("Elapsed", justify="right")
        summary.add_row(
            str(statuses.get("ok", 0)),
            str(statuses.get("failed", 0)),
            str(statuses.get("error", 0)),
            str(statuses.get("timeout", 0)),
            str(statuses.get("skipped", 0)),
            f"{elapsed:.2f} s",
        )
        console.print(summary)
        if measured_ms:
            console.print(f"[dim]Total measured command time: {measured_ms / 1000:.2f} s[/dim]")
        console.print()
        print_artifact("Raw benchmark data", results_path)
        console.print()
        console.print("[bold]Next step[/bold]")
        console.print(f"  tembench summarize --runs {results_path} --out-csv {out_dir / 'summary.csv'}")
