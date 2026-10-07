"""Parallel benchmark execution via ProcessPoolExecutor."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

from ..config import Config
from .grid import _run_grid_point, error_records
from .process import build_once
from .result import ResultWriter


def _run_parallel(
    cfg: Config,
    writer: ResultWriter,
    points: list[dict[str, object]],
    retries: int,
    workers: int,
) -> None:
    """Run grid points concurrently.

    Results are written from this process only, as each point completes.
    Points are dispatched independently, so a timeout cannot prune the larger
    inputs of its series; it only ends that point's own repetitions.
    """
    limits = cfg.limits
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for bench in cfg.benchmarks:
            build_error = build_once(bench)
            if build_error:
                for params in points:
                    writer.emit(bench.name, params, error_records(bench, params, limits.repeats, build_error, limits.metric))
                continue

            future_to_params = {
                pool.submit(
                    _run_grid_point,
                    bench,
                    params,
                    limits.timeout_sec,
                    limits.warmups,
                    limits.repeats,
                    retries,
                    limits.rss_poll_interval_sec,
                    limits.metric,
                    limits.prune_on_timeout,
                ): params
                for params in points
            }
            for fut in as_completed(future_to_params):
                params = future_to_params[fut]
                try:
                    results = fut.result()
                except Exception as exc:  # a worker process crashed
                    results = error_records(bench, params, limits.repeats, str(exc), limits.metric)
                writer.emit(bench.name, params, results)
