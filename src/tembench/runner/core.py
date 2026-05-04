"""Top-level orchestration: run all benchmarks per the supplied Config."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import psutil

from ..config import Config
from .grid import expand_grid
from .parallel import _run_parallel
from .provenance import write_provenance
from .result import TrialResult
from .serial import _run_serial

TrialCallback = Callable[[str, dict[str, object], int, int, TrialResult], None]


def run_benchmarks(
    cfg: Config,
    out_path: Path,
    seed: int = 42,
    retries: int = 0,
    on_trial: TrialCallback | None = None,
    append: bool = False,
):
    """Run all benchmarks and write results.

    Args:
        on_trial: Optional callback(bench_name, params, rep, total_reps, result)
                  called after each trial for progress reporting.
    """
    workers = max(1, cfg.limits.workers)
    points = expand_grid(cfg.grid)
    if cfg.limits.shuffle:
        random.Random(seed).shuffle(points)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # optional CPU pinning (Linux, serial mode only)
    if cfg.pin_cpu is not None and workers == 1:
        try:
            p = psutil.Process()
            p.cpu_affinity([cfg.pin_cpu])
        except Exception:
            pass

    write_provenance(out_path.parent, seed=seed, workers=workers)

    if workers == 1:
        _run_serial(
            cfg,
            out_path,
            points,
            retries,
            on_trial,
            append,
            cfg.limits.rss_poll_interval_sec,
        )
    else:
        _run_parallel(
            cfg,
            out_path,
            points,
            retries,
            on_trial,
            append,
            workers,
            cfg.limits.rss_poll_interval_sec,
        )
