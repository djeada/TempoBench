"""Matrix multiplication: O(n³).  The naive product of two n×n matrices.

The same algorithm, on the same input, as matrix_multiply.cpp and matrix_multiply.rs: all three
print the same CHECKSUM.  Only the algorithm is timed.

    python matrix_multiply.py --n 1000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from harness import SplitMix64, hash_sequence, measure, parse_args, random_values, report  # noqa: E402


def matrix_multiply(a: list[int], b: list[int], c: list[int], n: int) -> None:
    """c = a × b for n×n matrices stored row by row, in i-k-j order."""
    for idx in range(n * n):
        c[idx] = 0
    for i in range(n):
        row = i * n
        for k in range(n):
            aik = a[row + k]
            col = k * n
            for j in range(n):
                c[row + j] += aik * b[col + j]


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    n = args.n
    rng = SplitMix64()
    a = random_values(rng, n * n, 10)
    b = random_values(rng, n * n, 10)
    c = [0] * (n * n)
    ms, _ = measure(args.min_ms, lambda: matrix_multiply(a, b, c, n))
    report(hash_sequence(c), ms)


if __name__ == "__main__":
    main()
