"""Benchmark runner.

Public API:
    run_benchmarks, run_once, build_once, expand_grid, format_cmd, TrialResult

Back-compat re-exports used by tests:
    _run_grid_point, _run_serial, _run_parallel
"""

from .core import run_benchmarks
from .grid import _run_grid_point, expand_grid, format_cmd
from .parallel import _run_parallel
from .process import build_once, run_once
from .result import TrialResult
from .serial import _run_serial

__all__ = [
    "run_benchmarks",
    "run_once",
    "build_once",
    "TrialResult",
    "expand_grid",
    "format_cmd",
    "_run_grid_point",
    "_run_serial",
    "_run_parallel",
]
