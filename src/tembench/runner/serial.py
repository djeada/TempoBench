"""Sequential benchmark execution path (preserves prune-on-timeout semantics)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from ..config import Config
from .grid import _run_grid_point, format_cmd
from .process import build_once
from .result import TrialResult

TrialCallback = Callable[[str, dict[str, object], int, int, TrialResult], None]


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


def _run_serial(
    cfg: Config,
    out_path: Path,
    points: list[dict[str, object]],
    retries: int,
    on_trial: TrialCallback | None,
    append: bool,
    poll_interval_sec: float,
):
    """Original sequential execution path — preserves pruning behaviour."""
    mode = "a" if append else "w"
    reps = max(1, cfg.limits.repeats)
    with out_path.open(mode) as f:
        for bench in cfg.benchmarks:
            build_once(bench)
            timed_out_keys: set = set()
            for params in points:
                gk = cfg.limits.growth_key
                key_val = params.get(gk) if gk else None
                if (
                    cfg.limits.prune_on_timeout
                    and gk
                    and key_val is not None
                    and _should_prune_key(timed_out_keys, key_val)
                ):
                    skip_rec = TrialResult(
                        ts=datetime.now(timezone.utc).isoformat(),
                        status="skipped",
                        bench=bench.name,
                        cmd=format_cmd(bench.cmd, params),
                        params=dict(params),
                    )
                    f.write(json.dumps(skip_rec.to_dict()) + "\n")
                    f.flush()
                    if on_trial:
                        for rep in range(reps):
                            on_trial(bench.name, params, rep + 1, reps, skip_rec)
                    continue

                results = _run_grid_point(
                    bench,
                    params,
                    cfg.limits.timeout_sec,
                    cfg.limits.warmups,
                    cfg.limits.repeats,
                    retries,
                    poll_interval_sec,
                )
                for i, rec in enumerate(results):
                    f.write(json.dumps(rec.to_dict()) + "\n")
                    f.flush()
                    if on_trial:
                        on_trial(bench.name, params, i + 1, reps, rec)
                    if (
                        rec["status"] == "timeout"
                        and cfg.limits.prune_on_timeout
                        and gk
                        and key_val is not None
                    ):
                        timed_out_keys.add(key_val)
