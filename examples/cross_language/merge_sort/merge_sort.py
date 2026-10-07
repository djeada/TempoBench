"""Merge sort: O(n log n).  Top-down merge sort of n random values.

The same algorithm, on the same input, as merge_sort.cpp and merge_sort.rs: all three
print the same CHECKSUM.  Only the algorithm is timed.

    python merge_sort.py --n 1000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from harness import RotatedCopy, SplitMix64, hash_sequence, measure, parse_args, random_values, report  # noqa: E402


def _merge_sort_range(a: list[int], tmp: list[int], lo: int, hi: int) -> None:
    """Sort a[lo:hi]: sort each half, then merge the halves through `tmp`."""
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


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    rng = SplitMix64()
    source = random_values(rng, args.n, 1 << 30)
    work: list[int] = []
    ms, _ = measure(args.min_ms, lambda: merge_sort(work), RotatedCopy(source, work))
    report(hash_sequence(work), ms)


if __name__ == "__main__":
    main()
