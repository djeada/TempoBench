from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .placeholders import BUILTIN_PLACEHOLDER_NAMES

#: How the per-trial duration used for summaries and complexity fitting is chosen.
#:
#: ``wall``      always use the wall-clock time of the whole process;
#: ``reported``  require the command to print a ``TEMPOBENCH_MS`` marker, and
#:               fail the trial when it does not;
#: ``auto``      use the marker when present, otherwise fall back to wall time.
METRICS = ("auto", "wall", "reported")


@dataclass
class Benchmark:
    name: str
    cmd: str
    build: Optional[str] = None
    workdir: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class Limits:
    timeout_sec: Optional[float] = None
    warmups: int = 1
    repeats: int = 3
    rss_poll_interval_sec: float = 0.01
    prune_on_timeout: bool = False
    shuffle: bool = True
    growth_key: Optional[str] = "n"
    workers: int = 1
    metric: str = "auto"


@dataclass
class Config:
    benchmarks: List[Benchmark]
    grid: Dict[str, List[Any]]
    limits: Limits = field(default_factory=Limits)
    pin_cpu: Optional[int] = None


def _template_fields(template: str) -> set[str]:
    """Extract named format placeholders from a command template."""
    fields: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if not field_name:
            continue
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        if root and not root.isdigit():
            fields.add(root)
    return fields


def _validate_cmd_templates(benches: List[Benchmark], grid: Dict[str, List[Any]]) -> None:
    """Ensure every placeholder referenced by a benchmark command can be expanded."""
    known = set(grid) | BUILTIN_PLACEHOLDER_NAMES
    for bench in benches:
        missing = sorted(_template_fields(bench.cmd) - known)
        if missing:
            grid_keys = ", ".join(sorted(grid)) or "(none)"
            builtins = ", ".join(sorted(BUILTIN_PLACEHOLDER_NAMES))
            raise ValueError(
                f"Benchmark '{bench.name}' cmd references unknown placeholder(s): "
                f"{', '.join(missing)}. Available grid keys: {grid_keys}. "
                f"Built-in placeholders: {builtins}"
            )


def _validate_limits(limits: Limits) -> None:
    """Reject limit values that cannot produce a usable measurement."""
    if limits.metric not in METRICS:
        raise ValueError(
            f"limits.metric must be one of: {', '.join(METRICS)} (got {limits.metric!r})"
        )
    if limits.repeats < 1:
        raise ValueError(f"limits.repeats must be at least 1 (got {limits.repeats})")
    if limits.warmups < 0:
        raise ValueError(f"limits.warmups must not be negative (got {limits.warmups})")
    if limits.workers < 1:
        raise ValueError(f"limits.workers must be at least 1 (got {limits.workers})")
    if limits.timeout_sec is not None and limits.timeout_sec <= 0:
        raise ValueError(
            f"limits.timeout_sec must be positive when set (got {limits.timeout_sec})"
        )


def load_config(path: Path) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    benches = [Benchmark(**b) for b in data.get("benchmarks", [])]
    if not benches:
        raise ValueError(f"{path}: no benchmarks defined")

    grid = data.get("grid", {})
    empty_axes = sorted(key for key, values in grid.items() if not values)
    if empty_axes:
        raise ValueError(
            f"{path}: grid key(s) with no values would produce an empty sweep: "
            f"{', '.join(empty_axes)}"
        )

    limits = Limits(**data.get("limits", {}))
    pin_cpu = data.get("pin_cpu", None)
    _validate_cmd_templates(benches, grid)
    _validate_limits(limits)
    return Config(benchmarks=benches, grid=grid, limits=limits, pin_cpu=pin_cpu)
