# Top 100 algorithm complexity demos

This suite contains 100 directly runnable, deterministic pure-Python demos.
Every entry declares the input assumption under which its expected bound is
measured. TempoBench's supported canonical classes are `O(1)`, `O(log n)`,
`O(n)`, `O(n log n)`, `O(n²)`, and `O(n³)`.

## Run one demo

From the repository root, with the development environment activated:

```bash
python -m examples.top_100_algorithms.demos.binary_search --n 1024
python -m examples.top_100_algorithms.demos.binary_search --n 1024 --mode steps
python -m examples.top_100_algorithms.demos.binary_search --n 1024 --mode time
```

The modules in `demos/` are deliberately tiny entry points. Shared metadata
and kernels live in `catalog.py`, preventing 100 copies of benchmarking code
from drifting apart.

## Verify all 100

```bash
python -m examples.top_100_algorithms.verify
```

This produces:

- `artifacts/top-100-complexity.csv`: expected and fitted classes per demo;
- `artifacts/top-100-measurements.csv`: every input size, Python step count,
  and wall-clock sample.

The Python-step check counts line events in the code that actually executes.
It is deterministic and is the pass/fail gate. Wall-clock data is collected
without tracing, repeated, and median-pooled by identical complexity kernel;
it is supporting empirical evidence because operating-system scheduling and
CPU frequency changes cannot be made deterministic.

Important qualifications are recorded per row. For example, BFS is linear for
a sparse graph where `E = Θ(V)`, hash-table lookup is average-case constant,
and jump search's `O(sqrt(n))` is represented as `O(n)` because TempoBench does
not currently expose a square-root candidate model.
