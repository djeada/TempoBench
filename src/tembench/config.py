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


#: Grid axis names a summary already uses for something else.  An axis with one
#: of these names would be read back as a status count, dropping it from the
#: grid point's identity and joining unrelated rows in `compare`.  Mirrors
#: `summarize.NON_KEY_COLUMNS` and `METRIC_SUFFIXES`, kept separate so loading a
#: config does not import pandas.
RESERVED_AXIS_NAMES = frozenset(
    {"bench", "time_source", "ok", "failed", "timeout", "error", "skipped"}
)
_METRIC_SUFFIXES = ("_median", "_mean", "_count", "_p10", "_p90")


def _validate_grid(path: Path, grid: Any) -> Dict[str, List[Any]]:
    """Require a mapping of axis name to a non-empty list of values."""
    if grid is None:
        return {}
    if not isinstance(grid, dict):
        raise ValueError(f"{path}: grid must be a mapping of axis name to a list of values")
    for key, values in grid.items():
        # A bare string is iterable, so it would be swept one character at a time.
        if not isinstance(values, list):
            raise ValueError(
                f"{path}: grid.{key} must be a list of values (got {values!r}); "
                f"write it as [{values!r}] for a single value"
            )
        if str(key) in RESERVED_AXIS_NAMES or str(key).endswith(_METRIC_SUFFIXES):
            raise ValueError(
                f"{path}: grid axis name {key!r} is reserved for summary columns; "
                "rename it"
            )
    empty_axes = sorted(str(key) for key, values in grid.items() if not values)
    if empty_axes:
        raise ValueError(
            f"{path}: grid key(s) with no values would produce an empty sweep: "
            f"{', '.join(empty_axes)}"
        )
    return grid


def _check_type(name: str, value: Any, kinds: tuple[type, ...], label: str) -> None:
    # bool is an int subclass, so `repeats: true` would otherwise pass as 1.
    if isinstance(value, bool) and bool not in kinds:
        raise ValueError(f"limits.{name} must be {label} (got {value!r})")
    if not isinstance(value, kinds):
        raise ValueError(f"limits.{name} must be {label} (got {value!r})")


def _validate_limits(limits: Limits) -> None:
    """Reject limit values that cannot produce a usable measurement."""
    for name in ("warmups", "repeats", "workers"):
        _check_type(name, getattr(limits, name), (int,), "a whole number")
    for name in ("prune_on_timeout", "shuffle"):
        # YAML reads `"false"` as a string, and any non-empty string is truthy.
        _check_type(name, getattr(limits, name), (bool,), "true or false")
    _check_type("rss_poll_interval_sec", limits.rss_poll_interval_sec, (int, float), "a number")
    if limits.timeout_sec is not None:
        _check_type("timeout_sec", limits.timeout_sec, (int, float), "a number")
    if limits.rss_poll_interval_sec <= 0:
        raise ValueError(
            "limits.rss_poll_interval_sec must be positive "
            f"(got {limits.rss_poll_interval_sec})"
        )
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


def _load_benchmark(path: Path, entry: Any) -> Benchmark:
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: each benchmark must be a mapping with name and cmd")
    unknown = sorted(set(entry) - set(Benchmark.__dataclass_fields__))
    if unknown:
        raise ValueError(
            f"{path}: benchmark {entry.get('name', '?')!r} has unknown key(s): "
            f"{', '.join(unknown)}"
        )
    for required in ("name", "cmd"):
        if not isinstance(entry.get(required), str) or not entry[required].strip():
            raise ValueError(f"{path}: every benchmark needs a non-empty {required!r}")
    env = entry.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError(f"{path}: benchmark {entry['name']!r} env must be a mapping")
    # A subprocess environment holds only strings; YAML reads `1` as an int.
    return Benchmark(**{**entry, "env": {str(k): str(v) for k, v in env.items()}})


def load_config(path: Path) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    benches = [_load_benchmark(path, b) for b in data.get("benchmarks") or []]
    if not benches:
        raise ValueError(f"{path}: no benchmarks defined")

    grid = _validate_grid(path, data.get("grid"))

    limits_data = data.get("limits") or {}
    if not isinstance(limits_data, dict):
        raise ValueError(f"{path}: limits must be a mapping")
    unknown = sorted(set(limits_data) - set(Limits.__dataclass_fields__))
    if unknown:
        raise ValueError(
            f"{path}: unknown limits key(s): {', '.join(unknown)}. "
            f"Known keys: {', '.join(Limits.__dataclass_fields__)}"
        )
    limits = Limits(**limits_data)
    pin_cpu = data.get("pin_cpu", None)
    if pin_cpu is not None and (isinstance(pin_cpu, bool) or not isinstance(pin_cpu, int)):
        raise ValueError(f"{path}: pin_cpu must be a CPU index (got {pin_cpu!r})")
    _validate_cmd_templates(benches, grid)
    _validate_limits(limits)
    return Config(benchmarks=benches, grid=grid, limits=limits, pin_cpu=pin_cpu)
