"""What every Python implementation in examples/cross_language shares.

The input generator, the timing loop and the output.  harness.hpp and
harness.rs are the same code in C++ and Rust, so all three languages do the
same work.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, TypeVar

SEED = 42
HASH_MOD = 1_000_000_007
MASK64 = (1 << 64) - 1

T = TypeVar("T")


class SplitMix64:
    """The input generator, identical in every language."""

    def __init__(self, seed: int = SEED) -> None:
        self.state = seed

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, bound: int) -> int:
        return (self.next() >> 11) % bound


def hash_sequence(values: list[int]) -> int:
    h = 0
    for v in values:
        h = (h * 31 + v) % HASH_MOD
    return h


def random_values(rng: SplitMix64, n: int, bound: int) -> list[int]:
    return [rng.below(bound) for _ in range(n)]


class RotatedCopy:
    """Refill `work` with `source` rotated by a step that changes every call.

    Sorting the very same list over and over is not what a sort sees in
    practice; a rotation gives every call a different arrangement while the
    sorted result stays the same.
    """

    def __init__(self, source: list[int], work: list[int]) -> None:
        self.source, self.work, self.shift = source, work, 0

    def __call__(self) -> None:
        self.shift = (self.shift + 7919) % len(self.source)
        self.work[:] = self.source[self.shift:] + self.source[: self.shift]


def measure(min_ms: float, kernel: Callable[[], T], prepare: Callable[[], None] | None = None) -> tuple[float, T]:
    """Repeat `kernel` until at least `min_ms` of it has been timed.

    Returns the mean duration of one call and the last call's result.
    `prepare` runs untimed before each call.
    """
    total_ns = 0
    calls = 0
    while True:
        if prepare is not None:
            prepare()
        start = time.perf_counter_ns()
        result = kernel()
        total_ns += time.perf_counter_ns() - start
        calls += 1
        if total_ns >= min_ms * 1e6:
            return total_ns / calls / 1e6, result


def parse_args(description: str, max_n: int | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--min-ms", type=float, default=20.0)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    if max_n is not None and args.n > max_n:
        parser.error(f"--n must be at most {max_n}")
    return args


def report(checksum: int, ms: float) -> None:
    """The two lines TempoBench reads: what was computed, and how long it took."""
    print(f"CHECKSUM: {checksum}")
    print(f"TEMPOBENCH_MS: {ms:.6f}")
