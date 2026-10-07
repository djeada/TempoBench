"""Seven textbook algorithms, one per complexity class TempoBench fits.

A port of algorithms.cpp: the same algorithms on the same SplitMix64 inputs, so
all three languages print the same CHECKSUM for a given --algo and --n.  Only
the hot section is timed; interpreter startup and input generation are excluded
through the TEMPOBENCH_MS marker.

    python algorithms.py --algo merge_sort --n 100000
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, TypeVar

SEED = 42
HASH_MOD = 1_000_000_007
QUERIES = 1 << 16
MAX_CITIES = 24  # Held-Karp needs n * 2^n table entries
MASK64 = (1 << 64) - 1
INF = 1 << 61

T = TypeVar("T")


class SplitMix64:
    def __init__(self, seed: int) -> None:
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


# --- O(log n): binary search, a fixed batch of lookups ----------------------


def binary_search_count(sorted_values: list[int], queries: list[int]) -> int:
    found = 0
    size = len(sorted_values)
    for q in queries:
        lo, hi = 0, size
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_values[mid] < q:
                lo = mid + 1
            else:
                hi = mid
        if lo < size and sorted_values[lo] == q:
            found += 1
    return found


# --- O(sqrt n): count divisors by trial division -----------------------------


def divisor_count(n: int) -> int:
    count = 0
    d = 1
    while d * d < n:
        if n % d == 0:
            count += 2
        d += 1
    if d * d == n:
        count += 1
    return count


# --- O(n): Kadane's maximum subarray sum ------------------------------------


def max_subarray(values: list[int]) -> int:
    best = current = values[0]
    for v in values[1:]:
        current = v if v > current + v else current + v
        if current > best:
            best = current
    return best


# --- O(n log n): top-down merge sort ----------------------------------------


def _merge_sort_range(a: list[int], tmp: list[int], lo: int, hi: int) -> None:
    if hi - lo < 2:
        return
    mid = (lo + hi) // 2
    _merge_sort_range(a, tmp, lo, mid)
    _merge_sort_range(a, tmp, mid, hi)
    i, j, k = lo, mid, lo
    while i < mid and j < hi:
        if a[j] < a[i]:
            tmp[k] = a[j]
            j += 1
        else:
            tmp[k] = a[i]
            i += 1
        k += 1
    while i < mid:
        tmp[k] = a[i]
        i += 1
        k += 1
    while j < hi:
        tmp[k] = a[j]
        j += 1
        k += 1
    a[lo:hi] = tmp[lo:hi]


def merge_sort(a: list[int]) -> None:
    _merge_sort_range(a, [0] * len(a), 0, len(a))


# --- O(n^2): insertion sort --------------------------------------------------


def insertion_sort(a: list[int]) -> None:
    for i in range(1, len(a)):
        key = a[i]
        j = i
        while j > 0 and a[j - 1] > key:
            a[j] = a[j - 1]
            j -= 1
        a[j] = key


# --- O(n^3): naive matrix multiplication -------------------------------------


def matrix_multiply(a: list[int], b: list[int], c: list[int], n: int) -> None:
    for idx in range(n * n):
        c[idx] = 0
    for i in range(n):
        row = i * n
        for k in range(n):
            aik = a[row + k]
            col = k * n
            for j in range(n):
                c[row + j] += aik * b[col + j]


# --- O(n^2 2^n): Held-Karp travelling salesman -------------------------------


def held_karp(dist: list[int], n: int) -> int:
    if n == 1:
        return 0
    subsets = 1 << n
    dp = [INF] * (subsets * n)
    dp[n] = 0  # subset {0}, ending at city 0
    for mask in range(1, subsets, 2):
        for last in range(n):
            cost = dp[mask * n + last]
            if cost >= INF or not (mask >> last) & 1:
                continue
            for nxt in range(n):
                if (mask >> nxt) & 1:
                    continue
                to = (mask | (1 << nxt)) * n + nxt
                candidate = cost + dist[last * n + nxt]
                if candidate < dp[to]:
                    dp[to] = candidate
    full = (subsets - 1) * n
    return min(dp[full + last] + dist[last * n] for last in range(1, n))


def relabel_cities(dist: list[int], out: list[int], n: int, rng: SplitMix64) -> None:
    """Copy the distance matrix with the cities shuffled, as in algorithms.cpp.

    The shortest tour is the same whatever the cities are called.
    """
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.below(i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    for i in range(n):
        row = perm[i] * n
        for j in range(n):
            out[i * n + j] = dist[row + perm[j]]


# --- Harness -----------------------------------------------------------------


def measure(min_ms: float, kernel: Callable[[], T], prepare: Callable[[], None] | None = None) -> tuple[float, T]:
    """Repeat `kernel` until at least `min_ms` of it has been timed.

    Returns the mean duration of one call and the last call's result.
    `prepare` runs untimed before each call, e.g. to restore an input that the
    kernel sorts in place.
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


def run(algo: str, n: int, min_ms: float) -> tuple[float, int]:
    """Time one algorithm on its size-`n` input; return (ms per call, checksum)."""
    rng = SplitMix64(SEED)
    if algo == "binary_search":
        sorted_values = [2 * i for i in range(n)]
        queries = random_values(rng, QUERIES, 2 * n)
        return measure(min_ms, lambda: binary_search_count(sorted_values, queries))
    if algo == "divisor_count":
        return measure(min_ms, lambda: divisor_count(n))
    if algo == "max_subarray":
        values = [v - 1000 for v in random_values(rng, n, 2001)]
        return measure(min_ms, lambda: max_subarray(values))
    if algo in ("merge_sort", "insertion_sort"):
        source = random_values(rng, n, 1 << 30)
        sort = merge_sort if algo == "merge_sort" else insertion_sort
        work: list[int] = []
        shift = 0

        def restore() -> None:
            # Rotated by a step that changes every call, as in algorithms.cpp:
            # a fresh arrangement each time, the same sorted result.
            nonlocal shift
            shift = (shift + 7919) % n
            work[:] = source[shift:] + source[:shift]

        ms, _ = measure(min_ms, lambda: sort(work), restore)
        return ms, hash_sequence(work)
    if algo == "matrix_multiply":
        a = random_values(rng, n * n, 10)
        b = random_values(rng, n * n, 10)
        c = [0] * (n * n)
        ms, _ = measure(min_ms, lambda: matrix_multiply(a, b, c, n))
        return ms, hash_sequence(c)
    if algo == "held_karp":
        dist = [d + 1 for d in random_values(rng, n * n, 100)]
        relabelled = [0] * (n * n)
        return measure(min_ms, lambda: held_karp(relabelled, n), lambda: relabel_cities(dist, relabelled, n, rng))
    raise ValueError(f"unknown --algo {algo!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Time one algorithm for TempoBench.")
    parser.add_argument("--algo", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--min-ms", type=float, default=20.0)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    if args.algo == "held_karp" and args.n > MAX_CITIES:
        parser.error(f"held_karp needs --n <= {MAX_CITIES}")
    try:
        ms, checksum = run(args.algo, args.n, args.min_ms)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"CHECKSUM: {checksum}")
    print(f"TEMPOBENCH_MS: {ms:.6f}")


if __name__ == "__main__":
    main()
