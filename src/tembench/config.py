from __future__ import annotations

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
    prune_on_timeout: bool = False
    shuffle: bool = True
    growth_key: Optional[str] = "n"


@dataclass
class Config:
    benchmarks: List[Benchmark]
    grid: Dict[str, List[Any]]
    limits: Limits = field(default_factory=Limits)
    pin_cpu: Optional[int] = None


def load_config(path: Path) -> Config:
    data = yaml.safe_load(Path(path).read_text())
    benches = [Benchmark(**b) for b in data.get("benchmarks", [])]
    grid = data.get("grid", {})
    limits = Limits(**data.get("limits", {}))
    pin_cpu = data.get("pin_cpu", None)
    return Config(benchmarks=benches, grid=grid, limits=limits, pin_cpu=pin_cpu)
