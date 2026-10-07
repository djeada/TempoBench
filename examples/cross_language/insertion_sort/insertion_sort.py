"""Insertion sort: O(n²).  Insertion sort of n random values.

The same algorithm, on the same input, as insertion_sort.cpp and insertion_sort.rs: all three
print the same CHECKSUM.  Only the algorithm is timed.

    python insertion_sort.py --n 1000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from harness import RotatedCopy, SplitMix64, hash_sequence, measure, parse_args, random_values, report  # noqa: E402


def insertion_sort(a: list[int]) -> None:
    """Grow a sorted prefix one element at a time, shifting larger ones right."""
    for i in range(1, len(a)):
        key = a[i]
        j = i
        while j > 0 and a[j - 1] > key:
            a[j] = a[j - 1]
            j -= 1
        a[j] = key


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    rng = SplitMix64()
    source = random_values(rng, args.n, 1 << 30)
    work: list[int] = []
    ms, _ = measure(args.min_ms, lambda: insertion_sort(work), RotatedCopy(source, work))
    report(hash_sequence(work), ms)


if __name__ == "__main__":
    main()
