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
from ..app import app, console, fail, print_artifact, print_heading

#: Statuses that always indicate a broken setup or a crashing command.
_BROKEN_STATUSES = ("error", "failed")
#: Statuses that a sweep may produce on purpose, e.g. when probing for the
#: input size at which an implementation stops being viable.
_TOLERATED_STATUSES = ("timeout", "skipped")


def _failure_reason(record: dict) -> str:
    """Summarize why a trial did not succeed, preferring the command's own words."""
    for stream in ("stderr", "stdout"):
        text = str(record.get(stream) or "").strip()
        if text:
            return text.splitlines()[-1].strip()[:160]
    rc = record.get("rc")
    return f"exit code {rc}" if rc is not None else "no output captured"


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
    allow_failures: bool = typer.Option(
        False,
        "--allow-failures",
        help="Exit 0 even when trials fail (default: a failed or errored trial exits 1)",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Also exit non-zero on timed-out or skipped trials",
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

    elapsed = time.perf_counter() - started
    statuses: dict[str, int] = {}
    reasons: dict[tuple[str, str], int] = {}
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
            if status in _BROKEN_STATUSES:
                key = (status, _failure_reason(record))
                reasons[key] = reasons.get(key, 0) + 1
            if record.get("wall_ms") is not None:
                measured_ms += float(record["wall_ms"])

    broken = sum(statuses.get(s, 0) for s in _BROKEN_STATUSES)
    tolerated = sum(statuses.get(s, 0) for s in _TOLERATED_STATUSES)

    if not quiet:
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

        if reasons:
            console.print()
            why = Table(title="Why trials did not succeed", box=None, title_style="red bold")
            why.add_column("Status", style="red")
            why.add_column("Count", justify="right")
            why.add_column("Last message from the command", overflow="fold")
            for (status, reason), count in sorted(
                reasons.items(), key=lambda item: -item[1]
            ):
                why.add_row(status, str(count), reason)
            console.print(why)

        console.print()
        print_artifact("Raw benchmark data", results_path)

    if statuses.get("ok", 0) == 0:
        raise fail(
            "No trial succeeded — there is nothing to summarize.",
            f"Inspect the captured output: tembench inspect --runs {results_path}",
        )
    if broken and not allow_failures:
        raise fail(
            f"{broken} trial(s) failed or errored.",
            "Re-run with --allow-failures to keep the partial results and exit 0.",
        )
    if tolerated and strict and not allow_failures:
        raise fail(f"{tolerated} trial(s) timed out or were skipped (--strict).")

    if not quiet:
        console.print()
        console.print("[bold]Next step[/bold]")
        console.print(f"  tembench summarize --runs {results_path} --out-csv {out_dir / 'summary.csv'}")
