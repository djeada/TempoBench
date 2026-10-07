"""Sequential benchmark execution path (preserves prune-on-timeout semantics)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ..config import Config
from .grid import _run_grid_point, error_records, skipped_record
from .process import build_once
from .result import ResultWriter


def _should_prune_key(timed_out_keys: set[object], key_val: object) -> bool:
    """Return True when a key should be skipped after a smaller/equal timeout."""
    for timed_out_key in timed_out_keys:
        try:
            if cast(Any, key_val) >= cast(Any, timed_out_key):
                return True
        except TypeError:
            if key_val == timed_out_key:
                return True
    return False


def _series_key(params: Mapping[str, object], growth_key: str) -> tuple:
    """Identify the sweep a grid point belongs to, ignoring the input size.

    Timing out says something about one implementation at one size, not about
    every implementation at that size.  Pruning is therefore scoped to the other
    grid axes, so a slow quadratic implementation cannot truncate the sweep of a
    fast one and starve its complexity fit of data points.
    """
    return tuple(sorted((k, repr(v)) for k, v in params.items() if k != growth_key))


def _run_serial(
    cfg: Config,
    writer: ResultWriter,
    points: list[dict[str, object]],
    retries: int,
    cpu: int | None,
) -> None:
    """Run every grid point in turn, pruning larger inputs after a timeout."""
    limits = cfg.limits
    gk = limits.growth_key
    for bench in cfg.benchmarks:
        build_error = build_once(bench)
        if build_error:
            for params in points:
                writer.emit(bench.name, params, error_records(bench, params, limits.repeats, build_error, limits.metric))
            continue

        timed_out_by_series: dict[tuple, set] = {}
        for params in points:
            key_val = params.get(gk) if gk else None
            timed_out_keys = timed_out_by_series.setdefault(_series_key(params, gk) if gk else (), set())
            prunable = limits.prune_on_timeout and key_val is not None
            if prunable and _should_prune_key(timed_out_keys, key_val):
                writer.emit(
                    bench.name,
                    params,
                    [skipped_record(bench, params, limits.metric) for _ in range(limits.repeats)],
                )
                continue

            results = _run_grid_point(
                bench,
                params,
                limits.timeout_sec,
                limits.warmups,
                limits.repeats,
                retries,
                limits.rss_poll_interval_sec,
                limits.metric,
                prune_on_timeout=limits.prune_on_timeout,
                cpu=cpu,
            )
            writer.emit(bench.name, params, results)
            if prunable and any(rec.status == "timeout" for rec in results):
                timed_out_keys.add(key_val)
