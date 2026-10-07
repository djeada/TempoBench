# Same algorithm, three languages, one complexity class

Seven textbook algorithms, one for each class TempoBench fits, implemented
line for line in C++ (`algorithms.cpp`), Rust (`algorithms.rs`) and Python
(`algorithms.py`). A language changes the constant factor: Python is 10 to 140
times slower here. It must not change the class. This example checks that it
doesn't.

| Algorithm         | Expected     | Input                                            |
|-------------------|--------------|--------------------------------------------------|
| `binary_search`   | O(log n)     | 65,536 lookups in a sorted array of n elements   |
| `divisor_count`   | O(√n)        | trial division of n up to √n                     |
| `max_subarray`    | O(n)         | Kadane's algorithm over n values                 |
| `merge_sort`      | O(n log n)   | top-down merge sort of n random values           |
| `insertion_sort`  | O(n²)        | insertion sort of n random values                |
| `matrix_multiply` | O(n³)        | naive product of two n×n matrices                |
| `held_karp`       | O(n² 2^n)    | exact travelling-salesman tour over n cities     |

## Run it

From the repository root (needs `g++` and `rustc`):

```bash
python examples/cross_language/run_all.py          # all seven, ~4 minutes
python examples/cross_language/run_all.py merge_sort held_karp
```

For every algorithm this runs `tembench run`, `summarize`, `plot` and `report`
into `artifacts/cross_language/<algorithm>/`, then checks two things:

1. **Same work.** All three programs generate their input from the same
   SplitMix64 stream and print a `CHECKSUM` of the result. Every language must
   print the same checksum at every input size.
2. **Same class.** Every language must be fitted the expected class.

It exits non-zero if either check fails. On a quiet machine every fit is the
expected class, nearly all of them at high confidence:

```
┃ Algorithm       ┃ Expected   ┃ cpp             ┃ rust            ┃ python          ┃ Checksums ┃
│ binary_search   │ O(log n)   │ O(log n) high   │ O(log n) high   │ O(log n) high   │ match     │
│ divisor_count   │ O(√n)      │ O(√n) high      │ O(√n) high      │ O(√n) high      │ match     │
│ max_subarray    │ O(n)       │ O(n) high       │ O(n) high       │ O(n) high       │ match     │
│ merge_sort      │ O(n log n) │ O(n log n) high │ O(n log n) high │ O(n log n) high │ match     │
│ insertion_sort  │ O(n²)      │ O(n²) high      │ O(n²) high      │ O(n²) high      │ match     │
│ matrix_multiply │ O(n³)      │ O(n³) high      │ O(n³) high      │ O(n³) high      │ match     │
│ held_karp       │ O(n² 2^n)  │ O(n² 2^n) high  │ O(n² 2^n) high  │ O(n² 2^n) high  │ match     │
```

Each config can also be run on its own, like any other:

```bash
tembench run --config examples/cross_language/merge_sort.yaml --out-dir artifacts/merge_sort
```

The `cpp` and `rust` benchmarks have a `build:` step, so the binaries are
compiled before the sweep starts.

## What it takes to measure the algorithm and not the machine

The programs follow a few rules. Without them, the three languages do *not*
agree, and that is worth knowing before you benchmark your own code:

- **Time only the kernel.** Each program prints `TEMPOBENCH_MS` for its hot
  section, so process startup and input generation are left out. Python's
  startup alone is longer than most of the C++ measurements.
- **Repeat fast kernels.** A call is repeated until 20 ms have been timed and
  the mean is reported (`--min-ms`), so a 2 µs C++ call isn't measured close to
  the clock's resolution.
- **Don't let the optimiser delete the work.** The C++ uses `escape`/`opaque`
  barriers (the trick Google Benchmark uses), and the Rust uses
  `std::hint::black_box`.
- **Don't feed the branch predictor the same input twice.** Sorting the very
  same array thousands of times lets the CPU learn every comparison, which
  makes small inputs faster than they should be and the curve too steep. The
  sorts get a rotated copy on every call, and Held–Karp gets the cities
  relabelled. Neither change affects the result.
- **Stay inside one level of the memory hierarchy, or expect to measure it.**
  Binary search over 4 million elements measures as O(√n) in all three
  languages, because the lookups stop fitting in cache. The configs keep every
  sweep within a range where the cost per operation is flat. See
  [What a measured class does and does not mean](../../README.md#what-a-measured-class-does-and-does-not-mean).
- **Watch for hardware fast paths.** Rust (via LLVM) divides 64-bit numbers
  that fit in 32 bits with a faster instruction, so `divisor_count` slows down
  in a step once n passes 2³². Its sweep starts above that.
