"""Travelling salesman (Held–Karp): O(n² 2ⁿ).  The shortest tour through n cities, by dynamic programming.

The same algorithm, on the same input, as held_karp.cpp and held_karp.rs: all three
print the same CHECKSUM.  Only the algorithm is timed.

    python held_karp.py --n 1000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from harness import SplitMix64, measure, parse_args, random_values, report  # noqa: E402

MAX_CITIES = 24  # the table has n·2ⁿ entries
INF = 1 << 61


def held_karp(dist: list[int], n: int) -> int:
    """The shortest tour from city 0 through every city and back.

    dp[S][j] is the cheapest path that starts at 0, visits exactly the cities
    in S, and ends at j; each entry is extended by one more city at a time.
    """
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
    """Copy the distance matrix with the cities shuffled.

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


def main() -> None:
    args = parse_args(__doc__.splitlines()[0], max_n=MAX_CITIES)
    n = args.n
    rng = SplitMix64()
    dist = [d + 1 for d in random_values(rng, n * n, 100)]
    relabelled = [0] * (n * n)
    ms, tour = measure(args.min_ms, lambda: held_karp(relabelled, n), lambda: relabel_cities(dist, relabelled, n, rng))
    report(tour, ms)


if __name__ == "__main__":
    main()
