"""Maximum subarray (Kadane): O(n).  The largest sum of a contiguous run of n values.

The same algorithm, on the same input, as max_subarray.cpp and max_subarray.rs: all three
print the same CHECKSUM.  Only the algorithm is timed.

    python max_subarray.py --n 1000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from harness import SplitMix64, measure, parse_args, random_values, report  # noqa: E402


def max_subarray(values: list[int]) -> int:
    """Kadane's algorithm: the best sum ending here either extends the best sum
    ending one step earlier or starts afresh."""
    best = current = values[0]
    for v in values[1:]:
        current = v if v > current + v else current + v
        if current > best:
            best = current
    return best


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    rng = SplitMix64()
    values = [v - 1000 for v in random_values(rng, args.n, 2001)]
    ms, best = measure(args.min_ms, lambda: max_subarray(values))
    report(best, ms)


if __name__ == "__main__":
    main()
