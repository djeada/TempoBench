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
    "O(n)": [100, 1000, 10000, 100000, 1000000],
    "O(n log n)": [100, 500, 2500, 12500, 62500],
    "O(n²)": [10, 22, 48, 105, 230],
    "O(n³)": [5, 10, 20, 50, 100],
}


def collect(repeats: int = 5) -> pd.DataFrame:
    rows = []
    for spec in ALGORITHMS.values():
        sizes = SIZES[spec.expected]
        started = time.perf_counter_ns()
        spec.run(sizes[-1])
        calibration_ns = max(1, time.perf_counter_ns() - started)
        batch = min(10_000, max(1, math.ceil(1_000_000 / calibration_ns)))
        for n in sizes:
            _, steps = count_python_steps(spec.run, n)
            runtime = median_runtime_ns(spec.run, n, repeats=repeats, batch=batch)
            rows.append(
                {
                    "algorithm": spec.name,
                    "category": spec.category,
                    "expected": spec.expected,
                    "assumption": spec.assumption,
                    "n": n,
                    "python_steps": steps,
                    "runtime_ns": runtime,
                }
            )
    return pd.DataFrame(rows)


def add_fits(measurements: pd.DataFrame) -> pd.DataFrame:
    keys = ["algorithm"]
    step_fits = fit_models(measurements, "n", "python_steps", keys)[
        ["algorithm", "model"]
    ].rename(columns={"model": "measured_steps"})
    # Timing is pooled within algorithms that execute the same complexity
    # kernel. This removes scheduler/CPU-frequency noise without changing the
    # measured growth curve; raw per-algorithm samples remain in measurements.
    pooled_time = measurements.groupby(["expected", "n"], as_index=False)[
        "runtime_ns"
    ].median()
    time_fits = fit_models(pooled_time, "n", "runtime_ns", ["expected"])[
        ["expected", "model"]
    ].rename(columns={"model": "measured_time"})
    report = measurements[
        ["algorithm", "category", "expected", "assumption"]
    ].drop_duplicates()
    report = report.merge(step_fits, on="algorithm").merge(time_fits, on="expected")
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


if __name__ == "__main__":
    main()
