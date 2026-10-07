"""Binary search: O(log n).  65,536 lookups in a sorted array of n elements.

The same algorithm, on the same input, as binary_search.cpp and binary_search.rs: all three
print the same CHECKSUM.  Only the algorithm is timed.

    python binary_search.py --n 1000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from harness import SplitMix64, measure, parse_args, random_values, report  # noqa: E402

QUERIES = 1 << 16


def binary_search_count(sorted_values: list[int], queries: list[int]) -> int:
    """How many of `queries` are in `sorted_values`, each found by halving the range."""
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


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    rng = SplitMix64()
    sorted_values = [2 * i for i in range(args.n)]
    queries = random_values(rng, QUERIES, 2 * args.n)
    ms, found = measure(args.min_ms, lambda: binary_search_count(sorted_values, queries))
    report(found, ms)


if __name__ == "__main__":
    main()
