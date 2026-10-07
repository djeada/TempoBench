# Same algorithm, three languages, one complexity class

Seven textbook algorithms, one for each class TempoBench fits, each written
line for line in C++, Rust and Python. A language changes the constant factor:
Python is 10 to 140 times slower here. It must not change the class. This
example checks that it doesn't.

| Algorithm | Class | What it does | C++ | Rust | Python |
|-----------|-------|--------------|-----|------|--------|
| Binary search | O(log n) | 65,536 lookups in a sorted array of n elements | [`.cpp`](binary_search/binary_search.cpp) | [`.rs`](binary_search/binary_search.rs) | [`.py`](binary_search/binary_search.py) |
| Counting divisors | O(√n) | trial division of n up to √n | [`.cpp`](divisor_count/divisor_count.cpp) | [`.rs`](divisor_count/divisor_count.rs) | [`.py`](divisor_count/divisor_count.py) |
| Maximum subarray | O(n) | Kadane's algorithm over n values | [`.cpp`](max_subarray/max_subarray.cpp) | [`.rs`](max_subarray/max_subarray.rs) | [`.py`](max_subarray/max_subarray.py) |
| Merge sort | O(n log n) | top-down merge sort of n random values | [`.cpp`](merge_sort/merge_sort.cpp) | [`.rs`](merge_sort/merge_sort.rs) | [`.py`](merge_sort/merge_sort.py) |
| Insertion sort | O(n²) | insertion sort of n random values | [`.cpp`](insertion_sort/insertion_sort.cpp) | [`.rs`](insertion_sort/insertion_sort.rs) | [`.py`](insertion_sort/insertion_sort.py) |
| Matrix multiplication | O(n³) | naive product of two n×n matrices | [`.cpp`](matrix_multiply/matrix_multiply.cpp) | [`.rs`](matrix_multiply/matrix_multiply.rs) | [`.py`](matrix_multiply/matrix_multiply.py) |
| Travelling salesman | O(n² 2ⁿ) | Held–Karp: the exact shortest tour over n cities | [`.cpp`](held_karp/held_karp.cpp) | [`.rs`](held_karp/held_karp.rs) | [`.py`](held_karp/held_karp.py) |

Each folder holds the three implementations and the `benchmark.yaml` that
sweeps them. Each file is the algorithm plus a short `main`. What all of them
share lives once per language in [`common/`](common): the input generator, the
timing loop, and the two lines of output TempoBench reads
([`harness.hpp`](common/harness.hpp), [`harness.rs`](common/harness.rs),
[`harness.py`](common/harness.py)). Every program builds and runs on its own:

```bash
g++ -O2 -std=c++17 -o merge_sort examples/cross_language/merge_sort/merge_sort.cpp && ./merge_sort --n 100000
rustc -C opt-level=3 examples/cross_language/merge_sort/merge_sort.rs && ./merge_sort --n 100000
python examples/cross_language/merge_sort/merge_sort.py --n 100000
```

## Run it

From the repository root (needs `g++` and `rustc`):

```bash
python examples/cross_language/run_all.py          # all seven, ~4 minutes
python examples/cross_language/run_all.py merge_sort held_karp
python examples/cross_language/run_all.py --reels    # and a video for each
```

With `--reels`, each algorithm's folder in `artifacts/cross_language/` also gets
`reel.mp4` (see [Reels](../../README.md#reels)), a `reel.png` thumbnail, and a
`caption.txt` to post with it. The caption gives the result and links to
that algorithm's C++, Rust and Python files.

For every algorithm this runs `tembench run`, `summarize`, `plot` and `report`
into `artifacts/cross_language/<algorithm>/`, then checks two things:

1. **Same work.** All three programs generate their input from the same
   SplitMix64 stream and print a `CHECKSUM` of the result. Every language must
   print the same checksum at every input size.
2. **Same class.** Every language must be fitted the expected class.

It exits non-zero if either check fails. A run on an otherwise idle machine:

```
┃ Algorithm       ┃ Expected   ┃ cpp             ┃ rust            ┃ python          ┃ Checksums ┃
│ binary_search   │ O(log n)   │ O(log n) high   │ O(log n) high   │ O(log n) high   │ match     │
│ divisor_count   │ O(√n)      │ O(√n) high      │ O(√n) high      │ O(√n) high      │ match     │
│ max_subarray    │ O(n)       │ O(n) high       │ O(n) high       │ O(n) medium     │ match     │
│ merge_sort      │ O(n log n) │ O(n log n) high │ O(n log n) high │ O(n log n) high │ match     │
│ insertion_sort  │ O(n²)      │ O(n²) high      │ O(n²) high      │ O(n²) high      │ match     │
│ matrix_multiply │ O(n³)      │ O(n³) high      │ O(n³) high      │ O(n³) high      │ match     │
│ held_karp       │ O(n² 2^n)  │ O(n² 2^n) high  │ O(n² 2^n) high  │ O(n² 2^n) high  │ match     │
```

A rating below high gives its reason in `fits.csv`. For Python's max_subarray
here, O(n log n) fit almost as well (`ambiguous-class`). Which fits come out
medium changes from run to run; the classes have not changed in any run.

Each config can also be run on its own, like any other:

```bash
tembench run --config examples/cross_language/merge_sort/benchmark.yaml --out-dir artifacts/merge_sort
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
