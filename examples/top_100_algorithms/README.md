# Top 100 algorithm complexity demos

This suite contains 100 directly runnable, deterministic pure-Python demos.
Every entry declares the input assumption under which its expected bound is
measured. This suite also extends TempoBench's canonical model set with
`O(√n)` and `O(n² 2^n)`, alongside `O(1)`, `O(log n)`, `O(n)`,
`O(n log n)`, `O(n²)`, and `O(n³)`.

## Run one demo

From the repository root, with the development environment activated:

```bash
python -m examples.top_100_algorithms.demos.binary_search --n 1024
python -m examples.top_100_algorithms.demos.binary_search --n 1024 --mode steps
python -m examples.top_100_algorithms.demos.binary_search --n 1024 --mode time
```

The modules in `demos/` are deliberately tiny entry points. The genuine
implementations are grouped by family in `implementations_core.py`,
`implementations_graph.py`, and `implementations_advanced.py`; `catalog.py`
maps each entry point to one distinct callable.

## Verify all 100

```bash
python -m examples.top_100_algorithms.verify
```

This produces:

- `artifacts/top-100-complexity.csv`: expected and fitted classes per demo;
- `artifacts/top-100-measurements.csv`: every input size, Python step count,
  and wall-clock sample.

The Python-step check counts line events in the implementation and descendant
Python helpers that actually execute. It is deterministic and is the primary
complexity gate. Wall-clock data is collected without tracing and repeated per
algorithm; it remains supporting empirical evidence because operating-system
scheduling and CPU frequency changes cannot be made deterministic.

Important qualifications are recorded per row. For example, BFS is linear for
a sparse graph where `E = Θ(V)`, hash-table lookup is average-case constant,
and Held–Karp is measured as `O(n² 2^n)`. The sieve's tight
`O(n log log n)` bound is compared with the nearest supported canonical upper
bound, `O(n log n)`, and that qualification is never hidden.
