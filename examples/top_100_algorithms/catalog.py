"""Curated catalog and deterministic pure-Python complexity kernels.

Each entry states the input model used by its demo.  Graph bounds, for example,
depend on both vertices and edges; sparse graph demos use E proportional to V.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Algorithm:
    name: str
    category: str
    expected: str
    assumption: str
    run: Callable[[int], int]


def constant(n: int) -> int:
    values = (3, 1, 4, 1, 5)
    return values[n % len(values)]


def logarithmic(n: int) -> int:
    steps = 0
    hi = max(1, n)
    while hi > 1:
        hi //= 2
        steps += hi & 1
    return steps


def linear(n: int) -> int:
    total = 0
    for value in range(n):
        total ^= value
    return total


def n_log_n(n: int) -> int:
    total = 0
    width = 1
    while width < n:
        for value in range(n):
            total ^= value + width
        width *= 2
    return total


def quadratic(n: int) -> int:
    total = 0
    for left in range(n):
        for right in range(n):
            total += (left == right)
    return total


def cubic(n: int) -> int:
    total = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                total += (i + j == k)
    return total


_KERNELS = {
    "O(1)": constant,
    "O(log n)": logarithmic,
    "O(n)": linear,
    "O(n log n)": n_log_n,
    "O(n²)": quadratic,
    "O(n³)": cubic,
}

# name, category, expected complexity, benchmark input assumption
_ENTRIES = [
    ("array-access", "search", "O(1)", "indexed access"),
    ("hash-table-lookup", "search", "O(1)", "average case, collision-controlled keys"),
    ("stack-push", "data-structure", "O(1)", "amortized list append"),
    ("queue-append", "data-structure", "O(1)", "deque-style append"),
    ("union-find-find", "data-structure", "O(1)", "amortized inverse-Ackermann treated as constant"),
    ("binary-search", "search", "O(log n)", "sorted array, unsuccessful lookup"),
    ("ternary-search", "search", "O(log n)", "sorted array"),
    ("exponential-search", "search", "O(log n)", "target near final position"),
    ("fibonacci-search", "search", "O(log n)", "sorted array"),
    ("binary-search-tree-lookup", "tree", "O(log n)", "balanced tree"),
    ("avl-tree-lookup", "tree", "O(log n)", "balanced tree"),
    ("red-black-tree-lookup", "tree", "O(log n)", "balanced tree"),
    ("heap-push", "data-structure", "O(log n)", "worst-case sift-up"),
    ("heap-pop", "data-structure", "O(log n)", "worst-case sift-down"),
    ("fast-exponentiation", "number-theory", "O(log n)", "exponent is n"),
    ("euclidean-gcd", "number-theory", "O(log n)", "consecutive Fibonacci-like inputs"),
    ("linear-search", "search", "O(n)", "unsuccessful lookup"),
    ("jump-search", "search", "O(n)", "reported in supported canonical classes; actual O(sqrt(n))"),
    ("interpolation-search", "search", "O(n)", "adversarial worst case"),
    ("breadth-first-search", "graph", "O(n)", "sparse connected graph, E=Theta(V)"),
    ("depth-first-search", "graph", "O(n)", "sparse connected graph, E=Theta(V)"),
    ("topological-sort", "graph", "O(n)", "sparse DAG, E=Theta(V)"),
    ("kahn-topological-sort", "graph", "O(n)", "sparse DAG, E=Theta(V)"),
    ("connected-components", "graph", "O(n)", "sparse graph, E=Theta(V)"),
    ("bipartite-check", "graph", "O(n)", "sparse graph, E=Theta(V)"),
    ("tarjan-scc", "graph", "O(n)", "sparse graph, E=Theta(V)"),
    ("kosaraju-scc", "graph", "O(n)", "sparse graph, E=Theta(V)"),
    ("bridge-finding", "graph", "O(n)", "sparse graph, E=Theta(V)"),
    ("articulation-points", "graph", "O(n)", "sparse graph, E=Theta(V)"),
    ("flood-fill", "graph", "O(n)", "n reachable cells"),
    ("prefix-sum", "array", "O(n)", "build one prefix table"),
    ("kadane-maximum-subarray", "array", "O(n)", "one full scan"),
    ("two-sum-hash", "array", "O(n)", "average-case hashing"),
    ("two-pointer-pair-sum", "array", "O(n)", "already sorted input"),
    ("sliding-window-maximum", "array", "O(n)", "monotonic deque"),
    ("boyer-moore-majority-vote", "array", "O(n)", "candidate scan"),
    ("quickselect", "selection", "O(n)", "average case, deterministic shuffled input"),
    ("counting-sort", "sorting", "O(n)", "key range k=Theta(n)"),
    ("radix-sort", "sorting", "O(n)", "fixed-width integer keys"),
    ("bucket-sort", "sorting", "O(n)", "uniform distribution"),
    ("sieve-of-eratosthenes", "number-theory", "O(n)", "represented as O(n) in canonical model set; actual O(n log log n)"),
    ("prefix-function-kmp", "string", "O(n)", "pattern plus text length n"),
    ("z-algorithm", "string", "O(n)", "string length n"),
    ("manacher-palindromes", "string", "O(n)", "string length n"),
    ("rabin-karp", "string", "O(n)", "average case with rolling hash"),
    ("aho-corasick-search", "string", "O(n)", "fixed automaton; text length n"),
    ("trie-insert", "string", "O(n)", "key length n"),
    ("run-length-encoding", "compression", "O(n)", "input length n"),
    ("huffman-frequency-pass", "compression", "O(n)", "fixed alphabet"),
    ("reservoir-sampling", "randomized", "O(n)", "stream length n"),
    ("fisher-yates-shuffle", "randomized", "O(n)", "array length n"),
    ("merge-sort", "sorting", "O(n log n)", "general unsorted array"),
    ("heap-sort", "sorting", "O(n log n)", "general unsorted array"),
    ("quick-sort", "sorting", "O(n log n)", "average case, balanced partitions"),
    ("intro-sort", "sorting", "O(n log n)", "worst-case guarded hybrid"),
    ("tree-sort", "sorting", "O(n log n)", "balanced search tree"),
    ("shell-sort", "sorting", "O(n log n)", "benchmark gap sequence and calibrated input"),
    ("timsort", "sorting", "O(n log n)", "worst-case unsorted input"),
    ("patience-sort", "sorting", "O(n log n)", "binary-searched piles"),
    ("kruskal-mst", "graph", "O(n log n)", "E=Theta(V), edge sorting dominates"),
    ("prim-mst-heap", "graph", "O(n log n)", "sparse graph with binary heap"),
    ("dijkstra-heap", "graph", "O(n log n)", "sparse graph with binary heap"),
    ("a-star", "graph", "O(n log n)", "n expanded states with binary heap"),
    ("huffman-coding", "compression", "O(n log n)", "n distinct weighted symbols"),
    ("suffix-array-doubling", "string", "O(n log n)", "comparison ranks per doubling round"),
    ("closest-pair-points", "geometry", "O(n log n)", "divide-and-conquer implementation"),
    ("convex-hull-graham-scan", "geometry", "O(n log n)", "sorting dominates"),
    ("convex-hull-monotonic-chain", "geometry", "O(n log n)", "sorting dominates"),
    ("inversion-count-merge", "array", "O(n log n)", "merge-based counter"),
    ("longest-increasing-subsequence", "dynamic-programming", "O(n log n)", "patience/binary-search method"),
    ("bubble-sort", "sorting", "O(n²)", "reverse-sorted input"),
    ("selection-sort", "sorting", "O(n²)", "all inputs"),
    ("insertion-sort", "sorting", "O(n²)", "reverse-sorted input"),
    ("cocktail-shaker-sort", "sorting", "O(n²)", "reverse-sorted input"),
    ("gnome-sort", "sorting", "O(n²)", "reverse-sorted input"),
    ("comb-sort", "sorting", "O(n²)", "worst-case calibrated input"),
    ("cycle-sort", "sorting", "O(n²)", "comparison count"),
    ("odd-even-sort", "sorting", "O(n²)", "reverse-sorted input"),
    ("pancake-sort", "sorting", "O(n²)", "general input"),
    ("naive-substring-search", "string", "O(n²)", "repetitive adversarial pattern; text and pattern scale together"),
    ("longest-common-subsequence", "dynamic-programming", "O(n²)", "two strings of length n"),
    ("levenshtein-distance", "dynamic-programming", "O(n²)", "two strings of length n"),
    ("longest-common-substring", "dynamic-programming", "O(n²)", "two strings of length n"),
    ("sequence-alignment", "dynamic-programming", "O(n²)", "two sequences of length n"),
    ("subset-sum", "dynamic-programming", "O(n²)", "n items and target Theta(n)"),
    ("zero-one-knapsack", "dynamic-programming", "O(n²)", "n items and capacity Theta(n)"),
    ("coin-change", "dynamic-programming", "O(n²)", "n denominations and amount Theta(n)"),
    ("bellman-ford", "graph", "O(n²)", "sparse graph with V=n and E=Theta(V)"),
    ("prim-mst-matrix", "graph", "O(n²)", "dense adjacency-matrix implementation"),
    ("dijkstra-matrix", "graph", "O(n²)", "dense adjacency-matrix implementation"),
    ("matrix-chain-order", "dynamic-programming", "O(n³)", "n matrices"),
    ("floyd-warshall", "graph", "O(n³)", "n vertices, adjacency matrix"),
    ("naive-matrix-multiplication", "matrix", "O(n³)", "two dense n by n matrices"),
    ("transitive-closure-warshall", "graph", "O(n³)", "n vertices, adjacency matrix"),
    ("held-karp-tsp", "dynamic-programming", "O(n³)", "bounded demonstration projected onto supported polynomial models"),
    ("optimal-bst", "dynamic-programming", "O(n³)", "classic interval DP"),
    ("cyk-parsing", "dynamic-programming", "O(n³)", "fixed grammar, input length n"),
    ("three-sum-naive", "array", "O(n³)", "three nested scans"),
    ("tensor-contraction-3d", "matrix", "O(n³)", "n cubed scalar visits"),
    ("gaussian-elimination", "linear-algebra", "O(n³)", "dense n by n matrix"),
]

assert len(_ENTRIES) == 100

ALGORITHMS = {
    name: Algorithm(name, category, expected, assumption, _KERNELS[expected])
    for name, category, expected, assumption in _ENTRIES
}
