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

    The global trace decides once per call whether a frame belongs to the tree
    rooted at `function`, and only those frames get a line tracer.  Testing
    ancestry on every line event instead would cost a walk up the whole stack
    each time, which is quadratic in depth for the recursive demos.
    """
    steps = 0
    root_code = function.__code__
    depth = 0

    def trace_lines(frame: FrameType, event: str, arg: object):
        nonlocal steps, depth
        if event == "line":
            steps += 1
        elif event == "return":
            # Also fires when a frame unwinds on an exception, and when a
            # generator yields — each resume re-enters through `trace_calls`.
            depth -= 1
        return trace_lines

    def trace_calls(frame: FrameType, event: str, arg: object):
        nonlocal depth
        if event != "call":
            return None
        # Inside the root already, or this is the root itself: everything
        # reachable from here is part of the implementation under test.
        if depth > 0 or frame.f_code is root_code:
            depth += 1
            return trace_lines
        return None

    previous = sys.gettrace()
    try:
        sys.settrace(trace_calls)
        result = function(n)
    finally:
        sys.settrace(previous)
    return result, steps


def median_runtime_ns(function: Callable[[int], Any], n: int, repeats: int = 5, batch: int = 1) -> int:
    """Return the median untraced runtime, keeping tracing out of timings."""
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(batch):
            function(n)
        samples.append((time.perf_counter_ns() - started) / batch)
    return int(median(samples))
