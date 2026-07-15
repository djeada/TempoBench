"""Deterministic and wall-clock probes for algorithm demonstrations."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from statistics import median
from types import FrameType
from typing import Any


def count_python_steps(function: Callable[[int], Any], n: int) -> tuple[Any, int]:
    """Count executed Python line events inside the implementation module.

    Line events are deterministic for deterministic inputs and measure the code
    that actually ran, unlike a complexity formula supplied by the demo.
    """
    steps = 0
    module_name = function.__module__

    def trace(frame: FrameType, event: str, arg: object):
        nonlocal steps
        if event == "line" and frame.f_globals.get("__name__") == module_name:
            steps += 1
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        result = function(n)
    finally:
        sys.settrace(previous)
    return result, steps


def median_runtime_ns(
    function: Callable[[int], Any], n: int, repeats: int = 5, batch: int = 1
) -> int:
    """Return the median untraced runtime, keeping tracing out of timings."""
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(batch):
            function(n)
        samples.append((time.perf_counter_ns() - started) / batch)
    return int(median(samples))
