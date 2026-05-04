"""Parallel benchmark execution via ProcessPoolExecutor."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from .grid import _run_grid_point, format_cmd
from .process import build_once
from .result import TrialResult

TrialCallback = Callable[[str, dict[str, object], int, int, TrialResult], None]


def _run_parallel(
    cfg: Config,
    out_path: Path,
    points: list[dict[str, object]],
    retries: int,
    on_trial: TrialCallback | None,
    append: bool,
    workers: int,
    poll_interval_sec: float,
):
    """Parallel execution using ProcessPoolExecutor."""
    mode = "a" if append else "w"
    reps = max(1, cfg.limits.repeats)
    lock = threading.Lock()

    with out_path.open(mode) as f, ProcessPoolExecutor(max_workers=workers) as pool:
        for bench in cfg.benchmarks:
            build_once(bench)

            # Submit all grid points to the shared pool.
            future_to_params = {}
            for params in points:
                fut = pool.submit(
                    _run_grid_point,
                    bench,
                    params,
                    cfg.limits.timeout_sec,
                    cfg.limits.warmups,
                    cfg.limits.repeats,
                    retries,
                    poll_interval_sec,
                )
                future_to_params[fut] = params

            for fut in as_completed(future_to_params):
                params = future_to_params[fut]
                try:
                    results = fut.result()
                except Exception as exc:
                    # If a worker crashes, record an error
                    results = [
                        TrialResult(
                            ts=datetime.now(timezone.utc).isoformat(),
                            status="error",
                            rc=None,
                            wall_ms=0.0,
                            peak_rss_mb=0.0,
                            stderr=str(exc),
                            bench=bench.name,
                            cmd=format_cmd(bench.cmd, params),
                            params=dict(params),
                        )
                    ]

                with lock:
                    for i, rec in enumerate(results):
                        f.write(json.dumps(rec.to_dict()) + "\n")
                        f.flush()
                        if on_trial:
                            on_trial(bench.name, params, i + 1, reps, rec)
