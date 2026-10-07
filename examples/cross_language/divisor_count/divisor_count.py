"""Counting divisors: O(√n).  Count the divisors of n by trial division up to √n.

The same algorithm, on the same input, as divisor_count.cpp and divisor_count.rs: all three
print the same CHECKSUM.  Only the algorithm is timed.

    python divisor_count.py --n 1000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from harness import measure, parse_args, report  # noqa: E402


def divisor_count(n: int) -> int:
    """The number of divisors of n: they come in pairs (d, n/d) with d < √n."""
    count = 0
    d = 1
    while d * d < n:
        if n % d == 0:
            count += 2
        d += 1
    if d * d == n:
        count += 1
    return count


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    ms, count = measure(args.min_ms, lambda: divisor_count(args.n))
    report(count, ms)


if __name__ == "__main__":
    main()
