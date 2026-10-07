"""Top-level orchestration: run all benchmarks per the supplied Config."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..config import Config
from .grid import expand_grid
from .parallel import _run_parallel
from .provenance import write_provenance
from .result import ResultWriter, TrialCallback
from .serial import _run_serial


def pinning_problem(cfg: Config) -> str | None:
    """Say why `pin_cpu` will not be applied to this run, or None if it will.

    Raises ValueError for a CPU this process may not run on at all.
    """
    if cfg.pin_cpu is None:
        return None
    if not hasattr(os, "sched_setaffinity"):
        return "pin_cpu is ignored: CPU affinity is only supported on Linux"
    if cfg.limits.workers > 1:
        return "pin_cpu is ignored with more than one worker"
    available = sorted(os.sched_getaffinity(0))
    if cfg.pin_cpu not in available:
        raise ValueError(
            f"pin_cpu: CPU {cfg.pin_cpu} is not available to this process "
            f"(available: {', '.join(map(str, available))})"
        )
    return None


@contextmanager
def _runner_off_cpu(cpu: int | None) -> Iterator[None]:
    """Keep the runner's own threads off the core the benchmark is pinned to."""
    if cpu is None:
        yield
        return
    previous = os.sched_getaffinity(0)
    others = previous - {cpu}
    if others:
        os.sched_setaffinity(0, others)
    try:
        yield
    finally:
        os.sched_setaffinity(0, previous)


def run_benchmarks(
    cfg: Config,
    out_path: Path,
    seed: int = 42,
    retries: int = 0,
    on_trial: TrialCallback | None = None,
    append: bool = False,
) -> None:
    """Run all benchmarks and write one JSON record per trial to `out_path`.

    Args:
        on_trial: Optional callback(bench_name, params, rep, total_reps, result)
                  called after each trial for progress reporting.
    """
    workers = cfg.limits.workers
    points = expand_grid(cfg.grid)
    if cfg.limits.shuffle:
        random.Random(seed).shuffle(points)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cpu = cfg.pin_cpu if pinning_problem(cfg) is None else None

    write_provenance(out_path.parent, seed=seed, workers=workers, append=append)

    with out_path.open("a" if append else "w", encoding="utf-8") as handle, _runner_off_cpu(cpu):
        writer = ResultWriter(handle, cfg.limits.repeats, on_trial)
        if workers == 1:
            _run_serial(cfg, writer, points, retries, cpu)
        else:
            _run_parallel(cfg, writer, points, retries, workers)
