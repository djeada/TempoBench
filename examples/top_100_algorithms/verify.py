"""Verify operation-count and wall-clock scaling for all 100 demos."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from tembench.complexity import fit_models

from .catalog import ALGORITHMS
from .probes import count_python_steps, median_runtime_ns

SIZES = {
    "O(1)": [64, 256, 1024, 4096, 16384],
    "O(log n)": [16, 64, 256, 1024, 4096],
    "O(√n)": [100, 400, 1600, 6400, 25600],
    "O(n)": [100, 500, 2500, 12500],
    "O(n log n)": [50, 250, 1250, 6250, 31250],
    "O(n²)": [8, 16, 32, 64, 128],
    "O(n³)": [3, 5, 8, 13, 21],
    "O(n² 2^n)": [4, 5, 6, 7, 8],
}

TIME_SIZES = {**SIZES, "O(n³)": [5, 10, 20, 40, 80]}


def collect(repeats: int = 5) -> pd.DataFrame:
    rows = []
    for spec in ALGORITHMS.values():
        step_sizes = SIZES[spec.expected]
        time_sizes = TIME_SIZES[spec.expected]
        started = time.perf_counter_ns()
        spec.run(time_sizes[-1])
        calibration_ns = max(1, time.perf_counter_ns() - started)
        batch = min(10_000, max(1, math.ceil(1_000_000 / calibration_ns)))
        metadata = {
            "algorithm": spec.name,
            "category": spec.category,
            "expected": spec.expected,
            "assumption": spec.assumption,
        }
        for n in step_sizes:
            _, steps = count_python_steps(spec.run, n)
            rows.append({**metadata, "n": n, "python_steps": steps, "runtime_ns": float("nan")})
        for n in time_sizes:
            runtime = median_runtime_ns(spec.run, n, repeats=repeats, batch=batch)
            rows.append({**metadata, "n": n, "python_steps": float("nan"), "runtime_ns": runtime})
    return pd.DataFrame(rows)


def add_fits(measurements: pd.DataFrame) -> pd.DataFrame:
    keys = ["algorithm"]
    step_data = measurements.dropna(subset=["python_steps"])
    time_data = measurements.dropna(subset=["runtime_ns"])
    step_fits = fit_models(step_data, "n", "python_steps", keys)[["algorithm", "model"]].rename(columns={"model": "measured_steps"})
    time_fits = fit_models(time_data, "n", "runtime_ns", keys)[["algorithm", "model"]].rename(columns={"model": "measured_time"})
    report = measurements[["algorithm", "category", "expected", "assumption"]].drop_duplicates()
    report = report.merge(step_fits, on="algorithm").merge(time_fits, on="algorithm")
    report["steps_match"] = report["expected"] == report["measured_steps"]
    report["time_matches"] = report["expected"] == report["measured_time"]
    return report.sort_values(["category", "algorithm"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/top-100-complexity.csv"))
    parser.add_argument("--measurements", type=Path, default=Path("artifacts/top-100-measurements.csv"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    console = Console()
    console.rule("[bold blue]Top 100 Algorithm Complexity Verification[/bold blue]")
    console.print("[dim]Deterministic Python steps are authoritative; timings are supporting evidence.[/dim]\n")
    measurements = collect(max(1, args.repeats))
    report = add_fits(measurements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.measurements.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    measurements.to_csv(args.measurements, index=False)

    step_ok = int(report["steps_match"].sum())
    time_ok = int(report["time_matches"].sum())
    table = Table(box=None)
    table.add_column("Check")
    table.add_column("Matched", justify="right")
    table.add_column("Meaning")
    table.add_row("Python steps", f"{step_ok}/100", "deterministic pass/fail")
    table.add_row("Wall clock", f"{time_ok}/100", "noise-sensitive supporting signal")
    console.print(table)
    console.print(f"\n[green]Report[/green]       {args.output.resolve()}")
    console.print(f"[green]Measurements[/green] {args.measurements.resolve()}")

    failures = report.loc[~report["steps_match"], ["algorithm", "expected", "measured_steps"]]
    if not failures.empty:
        console.print("\n[red bold]Deterministic complexity mismatches[/red bold]")
        console.print(failures.to_string(index=False))
        sys.exit(1)

    timing_mismatches = report.loc[~report["time_matches"], ["algorithm", "expected", "measured_time"]]
    if not timing_mismatches.empty:
        console.print("\n[yellow bold]Wall-clock classifications that differ[/yellow bold]")
        console.print(timing_mismatches.to_string(index=False))
        console.print(
            "[dim]These do not invalidate deterministic complexity, but remain visible "
            "because cache, allocation, scheduling, and interpreter effects can change "
            "finite-range wall-clock fits.[/dim]"
        )


if __name__ == "__main__":
    main()
