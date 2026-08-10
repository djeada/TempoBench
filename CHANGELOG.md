# Changelog

## 0.1.0

First release.

TempoBench runs any shell command over a parameter sweep, records timing and
memory, estimates a Big-O class from the result, and reports how much the
measurements actually support that class.

### Measuring

- Parameter sweeps over arbitrary grids, with configurable repeats, warm-ups,
  timeouts, and optional parallel execution.
- Wall-clock time and peak RSS per trial, with Tukey outlier filtering.
- **Self-reported timing.** A command can print `TEMPOBENCH_MS: <ms>` to have
  its own hot section measured instead of the whole process. Process startup is
  a constant added to every reading, and a constant is what destroys a
  complexity fit — on the bundled example, wall clock reports O(n) for two
  implementations that self-reported timing correctly separates into O(n log n)
  and O(n).
- `{python}` expands to the running interpreter, so configs are portable.

### Reporting a class honestly

- Candidate models: O(1), O(log n), O(√n), O(n), O(n log n), O(n²), O(n³),
  O(n² 2ⁿ).
- Every fit carries a confidence rating and the specific caveats behind it:
  too few input sizes, too narrow a range, a flat signal, a wide exponent
  interval, constant overhead dominating the readings, a rival class that fits
  equally well, too few trials per point, or trials that disagree with each
  other.
- The runner-up class and its margin are reported, so a coin flip between two
  classes reads as one.

### Running in CI

- Failed runs, empty summaries, and regressions exit non-zero; failures are
  reported with the command's own message rather than a count.
- `tembench validate` checks a config and trial-runs its cheapest grid point in
  seconds, catching typos, missing interpreters, thin protocols, and configs
  that would silently fall back to wall-clock timing.
- Provenance records the machine that ran the benchmark, and reports read it
  back — so a report built elsewhere still describes where the numbers came
  from.

### Known limitations

- Empirical complexity measures the machine, not the algorithm. Once an input
  stops fitting in cache, the memory hierarchy sets the pace; the README
  documents this in "What a measured class does and does not mean".
- Sweeps must run on an idle machine. Contention produces smooth, wrong curves;
  TempoBench flags unrepeatable trials but cannot recover the measurement.
- Tests run on Linux in CI. Windows and macOS binaries are built and smoke
  tested, but the test suite is not run on them.
- The package and command are named `tembench`, while the project is
  TempoBench.
