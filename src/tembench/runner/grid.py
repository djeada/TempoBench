"""Grid expansion, command formatting, and per-grid-point execution."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from ..config import Benchmark
from ..placeholders import format_cmd
from .process import run_once
from .reported import MARKER_NAME
from .result import TrialResult


def expand_grid(grid: dict[str, list]) -> list[dict[str, object]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    points = []
    for combo in product(*values) if values else [()]:
        params = dict(zip(keys, combo))
        points.append(params)
    return points


def _require_reported(rec: TrialResult) -> TrialResult:
    """Fail an otherwise-successful trial that did not self-report its duration."""
    if rec.status != "ok" or rec.reported_ms is not None:
        return rec
    note = (
        f"limits.metric is 'reported' but the command printed no {MARKER_NAME} marker "
        f"on stdout. Print a line like '{MARKER_NAME}: 12.345' from the command, or "
        "set limits.metric to 'auto' or 'wall'."
    )
    stderr = f"{rec.stderr}\n{note}" if rec.stderr else note
    return dataclasses.replace(rec, status="failed", stderr=stderr)


def _run_grid_point(
    bench: Benchmark,
    params: dict[str, object],
    timeout: float | None,
    warmups: int,
    repeats: int,
    retries: int,
    poll_interval_sec: float = 0.01,
    metric: str = "auto",
    prune_on_timeout: bool = False,
    cpu: int | None = None,
) -> list[TrialResult]:
    """Execute warmups + repeats for a single (bench, params) combination.

    Returns exactly one record per repetition.  A repetition that fails is
    re-run up to `retries` times and only its final attempt is kept, so a flaky
    trial that eventually succeeds is not reported as a failure.  With
    `prune_on_timeout`, a timeout ends the point: its remaining repetitions
    would only time out too, so they are recorded as skipped.
    """
    cmd = format_cmd(bench.cmd, params)
    cwd = Path(bench.workdir) if bench.workdir else None
    reps = repeats

    def launch() -> TrialResult:
        return run_once(cmd, bench.env, cwd, timeout, poll_interval_sec=poll_interval_sec, cpu=cpu)

    def context(rec: TrialResult) -> TrialResult:
        return rec.with_context(bench=bench.name, cmd=cmd, params=params, metric=metric)

    def skipped(count: int) -> list[TrialResult]:
        return [skipped_record(bench, params, metric) for _ in range(count)]

    for _ in range(warmups):
        warm = launch()
        if prune_on_timeout and warm.status == "timeout":
            # The warm-up is the observation that this point is too slow.
            return [context(warm), *skipped(reps - 1)]

    results: list[TrialResult] = []
    for _ in range(reps):
        for attempt in range(1, max(0, retries) + 2):
            rec = context(launch())
            if metric == "reported":
                rec = _require_reported(rec)
            if rec.status == "ok":
                break
        results.append(dataclasses.replace(rec, attempts=attempt))
        if prune_on_timeout and rec.status == "timeout":
            return results + skipped(reps - len(results))
    return results


def skipped_record(
    bench: Benchmark, params: dict[str, object], metric: str | None = None
) -> TrialResult:
    """One repetition that was not run because a smaller input timed out."""
    return TrialResult(
        ts=datetime.now(timezone.utc).isoformat(),
        status="skipped",
        bench=bench.name,
        cmd=format_cmd(bench.cmd, params),
        params=dict(params),
        metric=metric,
    )


def error_records(
    bench: Benchmark,
    params: dict[str, object],
    reps: int,
    message: str,
    metric: str | None = None,
) -> list[TrialResult]:
    """Stand-in records for repetitions that could not run at all.

    One per repetition, so every column of the run summary counts trials.
    """
    ts = datetime.now(timezone.utc).isoformat()
    rec = TrialResult(
        ts=ts,
        status="error",
        rc=None,
        stderr=message,
        bench=bench.name,
        cmd=format_cmd(bench.cmd, params),
        params=dict(params),
        metric=metric,
    )
    return [rec] * max(1, reps)
