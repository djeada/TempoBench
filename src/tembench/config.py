from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


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
    """Ensure every placeholder referenced by a benchmark command is in the grid."""
    grid_keys = set(grid)
    for bench in benches:
        missing = sorted(_template_fields(bench.cmd) - grid_keys)
        if missing:
            available = ", ".join(sorted(grid_keys)) or "(none)"
            raise ValueError(
                f"Benchmark '{bench.name}' cmd references missing grid key(s): "
                f"{', '.join(missing)}. Available grid keys: {available}"
            )


def load_config(path: Path) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    benches = [Benchmark(**b) for b in data.get("benchmarks", [])]
    grid = data.get("grid", {})
    limits = Limits(**data.get("limits", {}))
    pin_cpu = data.get("pin_cpu", None)
    _validate_cmd_templates(benches, grid)
    return Config(benchmarks=benches, grid=grid, limits=limits, pin_cpu=pin_cpu)
