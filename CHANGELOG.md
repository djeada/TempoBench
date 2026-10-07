# Changelog

## 0.3.0

This release makes complexity fits markedly more accurate and their confidence
more honest. It reworks the charts and reports, and adds `tembench reel` for
turning a run into a short video. A new example benchmarks seven algorithms in
C++, Rust and Python and checks that every language is fitted the same class.

### Fitting

- New O(2^n) class.  Exponential classes are considered when the largest
  input is at most 64, instead of when the raw log-log slope is steep, which
  startup overhead flattened: exponential growth plus overhead used to be
  reported as O(n³).
- A fitted intercept may only dip below zero by a quarter of the smallest
  reading, so `C·n − b` can no longer pass for O(n log n).
- The empirical exponent and its interval are measured net of the fitted
  overhead: O(n²) plus startup measures n^2.0, not n^0.5.
- Confidence is better calibrated.  New `poor-fit` and `constant-class`
  caveats; short sweeps need a larger AIC lead; `few-points` counts distinct
  sizes; exponential ranges are judged in doublings.  `fits.csv` gains a
  `caveats` column of machine-readable codes.
- On 2,286 synthetic series accuracy rises from 78% to 92%, and fits rated
  high are right 99.5% of the time.

### Running

- The `TEMPOBENCH_MS` marker is found anywhere in stdout, not only in the
  kept tail.
- Timeouts are enforced on time however coarse `rss_poll_interval_sec` is.
- `pin_cpu` pins only the benchmark and keeps TempoBench's own threads off
  that core; an unavailable CPU is an error.
- Configs are checked for what used to crash mid-sweep or corrupt a summary:
  commands that cannot expand at some grid point, axes named like trial
  fields, list-valued grid entries, duplicate benchmark names, null env
  values.  Invalid configs are reported without a traceback.
- Windows argument quoting handles a trailing backslash.

### Charts and reports

- The report leads with each series' class, confidence and bound, and draws
  its runtime chart from the summary instead of a separate `runtime.html`.
- Fit curves stay in their own benchmark's panel.  Axes go log
  automatically for wide ranges; labels sit at the line ends.
- One page layout, palette and number format across report, dashboard,
  comparison and chart pages, following the system's light or dark theme.
- `compare` leads with each point's verdict and change, separates slower
  points from ones that could not be checked, and colours only the metric
  that decides.
- Every chart and report command accepts `--output` and `--out-html`;
  `--bench` works on `dashboard`, `memory` and `heatmap`.  Asking for a
  metric or axis that is not there is an error instead of a blank chart.

### Reels

- `tembench reel` renders a finished run as a ~25-second vertical video:
  - a hook;
  - the trials replayed in run order;
  - one curve per series morphing through every complexity class, with lines
    showing how far each one misses;
  - a verdict in which series sharing a class collapse onto one curve.
- It has a soundtrack synthesised in step with the picture: a pad, plucks for
  trials, and a chime for the winner.
- It needs the `reel` extra (matplotlib) and ffmpeg.

### Examples

- `examples/cross_language`: seven algorithms, one per class, in C++, Rust
  and Python, one folder per algorithm with a file per language.  A script
  checks that every language computes the same result and is fitted the
  same class; `--reels` renders a video for each, with a caption linking
  the three implementations.
- `unique_bench.yaml` sweeps smaller sizes, so its quadratic implementation
  has enough points to fit.

## 0.2.0

This release improves measurement accuracy and makes benchmark results more
reliable to interpret.

### Measurement and execution

- Trial duration no longer depends on the memory polling interval, and peak
  memory uses the operating system's high-water mark plus process-tree sampling.
- Retries, launch errors, invalid output, build failures, timeouts, and skipped
  repetitions are recorded consistently. Background child processes are cleaned
  up after a benchmark exits.
- Commands receive grid values as quoted arguments. Config validation catches
  malformed grids, reserved axis names, mistyped limits, and unknown keys.

### Fitting and reporting

- Complexity models are fitted and ranked using relative error, with improved
  handling of overhead, outliers, non-finite values, and small inputs.
- Summaries retain attempted grid points without successful trials, and
  comparisons flag when a previously measured point has no timing.
- Charts and reports handle additional grid axes, empty points, log scales,
  benchmark filtering, trial statuses, and escaped output more reliably.

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
