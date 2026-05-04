"""Grid expansion, command formatting, and per-grid-point execution."""

from __future__ import annotations

from itertools import product
from pathlib import Path

from ..config import Benchmark
from .process import run_once
from .result import TrialResult


def expand_grid(grid: dict[str, list]) -> list[dict[str, object]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    points = []
    for combo in product(*values) if values else [()]:
        params = dict(zip(keys, combo))
        points.append(params)
    return points


def format_cmd(template: str, params: dict[str, object]) -> str:
    return template.format(**params)


def _run_grid_point(
    bench: Benchmark,
    params: dict[str, object],
    timeout: float | None,
    warmups: int,
    repeats: int,
    retries: int,
    poll_interval_sec: float = 0.01,
) -> list[TrialResult]:
    """Execute warmups + repeats for a single (bench, params) combination.

    Returns a list of result records (one per repeat, plus any retry attempts).
    """
    cmd = format_cmd(bench.cmd, params)
    cwd = Path(bench.workdir) if bench.workdir else None

    # Warmups
    for _ in range(max(0, warmups)):
        run_once(cmd, bench.env, cwd, timeout, poll_interval_sec=poll_interval_sec)

    # Repeats
    results: list[TrialResult] = []
    attempt = 0
    done = 0
    reps = max(1, repeats)
    while done < reps:
        attempt += 1
        rec = run_once(
            cmd,
            bench.env,
            cwd,
            timeout,
            poll_interval_sec=poll_interval_sec,
        ).with_context(bench=bench.name, cmd=cmd, params=params)
        results.append(rec)
        if rec["status"] == "ok":
            done += 1
        else:
            if attempt - done <= retries:
                continue
            else:
                done += 1  # count as a failed repetition
    return results
