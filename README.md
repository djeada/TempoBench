# TempoBench

A language-agnostic benchmarking CLI that runs any shell command with parameter sweeps, records timing and memory, estimates Big-O complexity with an honest confidence rating, and generates reports, all from a single YAML config.

<img width="1206" height="795" alt="Screenshot from 2026-02-12 21-53-22" src="https://github.com/user-attachments/assets/26c75949-de62-482d-8cfd-3d27db35eb95" />

## Features

- **Parameter sweeps** — define grids of inputs and implementations; every combination is executed with configurable repeats, warm-ups, and timeouts.
- **Parallel execution** — run grid points concurrently with `--workers`/`-j` to cut total benchmark time on multi-core machines.
- **Metrics collection** — wall-clock time and peak RSS are captured per trial using OS-native tools (`psutil`), with outlier filtering via Tukey fences.
- **Self-reported timing** — a benchmarked command can print its own inner timing so process startup is excluded from the measurement. See [Measuring the work, not the process](#measuring-the-work-not-the-process).
- **Complexity estimation** — observed runtimes are fitted against O(1) … O(n² 2ⁿ) models using a multi-layer algorithm (outlier-robust constant detection, log-log slope fallback, step-up OLS with adaptive thresholds, and tail-ratio guards). The best model is selected and overlaid on plots.
- **Fit confidence** — every fitted class is reported with a confidence rating and the caveats behind it, so a class the measurements cannot support is never presented as though they did.
- **CI-ready exit codes** — a run in which trials failed, or a summary with nothing in it, exits non-zero instead of reporting success.
- **Interactive charts** — Vega-Lite charts with click-to-toggle legend, crosshair tooltips, and smooth fit curves. Data points shown as discrete markers, fit lines as smooth interpolated curves.
- **Rich CLI output** — live progress bars, colored status tables, and system-info display powered by [Rich](https://github.com/Textualize/rich).
- **Reports & dashboards** — single-file HTML reports with heatmaps, comparison views, and regression detection. Chart data and styling are embedded; the Vega renderer loads from a CDN, so drawing charts needs network access.
- **Baseline comparison** — flag regressions against a previous run above a configurable threshold. Rows match on whatever grid columns the two summaries share, so any sweep works.
- **Reproducibility** — a provenance snapshot records the seed, invocation, and the CPU/memory of the machine that ran the benchmark. Reports read it back, so a report built on your laptop still describes the CI runner that produced the numbers.

## Installation

Requires Python 3.10+. Not published to PyPI yet, so install from a clone:

```bash
git clone https://github.com/djeada/TempoBench
cd TempoBench
pip install -e .
```

For development (adds pytest, ruff, and mypy):

```bash
pip install -e ".[dev]"
pytest          # ~16s
ruff check src tests
mypy
```

## Quickstart

**1. Define a benchmark** in YAML (see [`examples/unique_bench.yaml`](examples/unique_bench.yaml)):

```yaml
benchmarks:
  - name: py_unique_count
    # {python} expands to the interpreter TempoBench runs under.
    cmd: "{python} examples/unique_impl.py --n {n} --impl {impl} --seed 42"

grid:
  n: [10000, 50000, 100000, 500000, 1000000]
  impl: ["quadratic", "sort_scan", "hash_set"]

limits:
  timeout_sec: 30
  warmups: 1         # discard the first, cold measurement per grid point
  repeats: 5         # enough samples for the median to mean something
  metric: reported   # the command times itself; see below
```

> The measurement protocol matters as much as the fitting. On this very
> benchmark, dropping to `warmups: 0, repeats: 2` makes both implementations
> come out **O(n²)** — cold allocator state inflates the small inputs and tilts
> the whole curve. `tembench validate` warns about a protocol that thin.

**2. Check it before committing to a full sweep:**

```bash
tembench validate --config examples/unique_bench.yaml
```

This prints the planned sweep, expands a sample command, and runs the cheapest
grid point once — so a typo, a missing interpreter, or a config that would
silently fall back to wall-clock timing surfaces in seconds rather than after a
long run. It exits non-zero if the command does not run.

**3. Run the pipeline:**

```bash
# Sequential (default)
tembench run --config examples/unique_bench.yaml --out-dir artifacts

# Parallel — 4 workers for faster execution
tembench run --config examples/unique_bench.yaml --out-dir artifacts -j 4

tembench summarize --runs artifacts/runs.jsonl --out-csv artifacts/summary.csv
tembench plot --summary artifacts/summary.csv --out-html artifacts/runtime.html
```

**4. Generate a full report:**

```bash
tembench report --summary artifacts/summary.csv
```

## CLI Commands

| Command       | Description                                                |
|---------------|------------------------------------------------------------|
| `validate`    | Check a config and trial-run its cheapest grid point       |
| `run`         | Execute benchmarks and write JSONL results                 |
| `summarize`   | Aggregate runs into CSV with medians, percentiles, counts  |
| `plot`        | Generate a runtime chart with optional Big-O fit overlay   |
| `report`      | Build a single-file HTML report                            |
| `compare`     | Detect regressions against a baseline summary              |
| `dashboard`   | Create a multi-chart interactive dashboard                 |
| `inspect`     | Preview recent runs in a table                             |
| `memory`      | Generate a memory-usage chart                              |
| `heatmap`     | Generate a performance heatmap                             |
| `sysinfo`     | Display system information for reproducibility             |

Run `tembench --help` or `tembench <command> --help` for full option details.

### Arbitrary grids

`n` and `impl` are the bundled examples' names, not requirements. Every command
reads the sweep axes off the summary: the input size is the numeric grid column
swept most widely, and the series is whatever axis is left. So this works with no
extra flags:

```yaml
grid:
  vertices: [100, 1000, 10000, 100000]
  algorithm: ["dijkstra", "bellman_ford"]
limits:
  growth_key: vertices     # which axis `prune_on_timeout` treats as the size
```

```
$ tembench plot --summary artifacts/summary.csv
Axes  x = vertices, series = algorithm (inferred; override with --x / --color)
```

Pass `--x` and `--color` to override the inference; `compare` likewise joins on
whatever grid columns the two summaries share.

## Measuring the work, not the process

Timing a whole process also times everything around the work: interpreter or
runtime startup, dynamic linking, argument parsing, input generation. That
overhead is a **constant added to every reading**, and a constant is exactly
what destroys a complexity estimate — it flattens the curve most at small `n`,
where the signal is weakest.

Concretely: `examples/unique_impl.py` swept over `n = 50k … 800k`, where ~90 ms
of Python startup sits on top of every reading. Fitted on the wall-clock column
of that run, both `sort_scan` and `hash_set` come out **O(n)**. Fitted on the
self-reported column of *the same trials*, they separate correctly into
**O(n log n)** and **O(n)**. At smaller `n`, where startup is proportionally
larger still, wall clock degrades further — to **O(log n)** and **O(1)**.

To measure the work, have the command time its own hot section and print one
line on stdout:

```
TEMPOBENCH_MS: 12.345
```

The marker is deliberately trivial so any language can emit it — `:`, `=`, or a
space all work as the separator, and if it is printed more than once the last
occurrence wins. `examples/unique_impl.py` shows the pattern.

Then choose how durations are taken, via `limits.metric`:

| Value      | Behaviour                                                                    |
|------------|------------------------------------------------------------------------------|
| `auto`     | Default. Use the marker when the command prints one, else fall back to wall clock. |
| `reported` | Require the marker; a trial without one is recorded as failed.                |
| `wall`     | Always use wall-clock time of the whole process.                              |

Both readings are always kept in `runs.jsonl` (`wall_ms` and `reported_ms`), so
the startup cost stays inspectable. `summarize` picks the canonical duration per
series into `time_ms_*` and records which one it used in `time_source`. It never
mixes the two inside a series — a series where only some trials reported would
otherwise show a step no algorithm produces.

If the command cannot be modified, use wall clock but push `n` high enough that
the constant is negligible. TempoBench flags the fit as `overhead-dominated`
when it is not.

## Fit Confidence

Fitting always returns a class. Whether that class means anything depends on the
measurements behind it, so every fit is reported with a confidence rating and
the reasons for it, in `fits.csv` (`confidence`, `confidence_notes`), in the
terminal, in chart tooltips, and in the HTML report.

When a rival class fits nearly as well, `fits.csv` names it in `runner_up` and
records the gap in `model_margin`, and the terminal prints the pair as
`O(n log n) ≈ O(n²)`. On clean data the margin runs into the hundreds; on a
short, noisy sweep it collapses, and saying so is more useful than picking one
of the two and sounding certain.

A fit is downgraded once per caveat that applies — one caveat gives `medium`,
two or more give `low`:

| Caveat               | Meaning                                                    |
|----------------------|------------------------------------------------------------|
| `few-points`         | Fewer than 4 input sizes — classes cannot be separated.    |
| `narrow-n-range`     | Input sizes span less than 8×.                             |
| `flat-signal`        | Durations span less than 2× — nothing grew enough to fit.  |
| `wide-exponent-ci`   | The bootstrap exponent interval is wider than 0.5.         |
| `overhead-dominated` | Over 50% of the largest reading is constant overhead.      |
| `ambiguous-class`    | Another class explains the data about as well (ΔAIC < 2).  |
| `thin-samples`       | Fewer than 3 trials per input size.                        |
| `unstable-timings`   | Repeated trials of one point disagree by over 50% of its median — usually a busy machine. |

Confidence is advisory: it never changes the fitted bound, only how much you
should trust the label on it. Note that an `O(1)` result is never rated `high` —
a flat curve is also what a sweep that never reached interesting input sizes
looks like, and the two cannot be told apart from the data.

### Measure on a quiet machine

Timings are only as good as the machine that took them. A benchmark sharing a
CPU with a build, a video encode, or another benchmark will produce medians that
still trace a smooth curve — a smooth *wrong* curve. TempoBench flags this when
repeated trials of the same point disagree (`unstable-timings`), but the fix is
to run the sweep on an idle machine and, in CI, on a dedicated runner.

### What a measured class does and does not mean

A fit describes **elapsed time on the machine that ran it**, not the operation
count of the algorithm. Those diverge once an input stops fitting in cache: in
the example benchmark, `hash_set` performs a textbook O(n) number of operations
but measures around n^1.24 from 10⁴ to 5·10⁶ elements, because the memory
hierarchy, not the hash table, sets the pace at the top of that range. This is a
real property of the code on that hardware and TempoBench reports it faithfully;
it is not a substitute for analysing the algorithm.

If you want the operation count instead, have the command count operations and
emit that number through the marker. Nothing downstream inspects the units, so
the fit is over whatever you report — only the axis labels will still read "ms".
[`examples/top_100_algorithms`](examples/top_100_algorithms) takes this approach,
using deterministic Python step counts as the authoritative signal and treating
wall-clock timings as supporting evidence.

## Exit Codes

TempoBench is meant to run in CI, so commands fail loudly rather than producing
empty artifacts:

| Situation                                   | Exit | Override           |
|---------------------------------------------|------|--------------------|
| `validate` probe could not run the command  | 1    | `--no-probe`       |
| No trial succeeded                          | 1    | —                  |
| Some trials failed or errored               | 1    | `--allow-failures` |
| Trials timed out or were skipped            | 0    | `--strict` makes it 1 |
| `summarize` found no successful trials      | 1    | —                  |
| A chart or report was asked for empty data  | 1    | —                  |
| `compare` detected a regression             | 1    | —                  |

A failing run prints each distinct failure with the command's own last message,
rather than only a count.

## Parallel Execution

By default, benchmarks run sequentially (`workers: 1`) for the most accurate
timing. When wall-clock precision is less critical and throughput matters,
use multiple workers to run grid points concurrently:

```bash
# CLI flag (overrides config)
tembench run --config bench.yaml -j 4

# Or set in YAML
limits:
  workers: 4
```

> **Note:** Parallel runs share CPU and memory bandwidth, so individual timings
> may show more variance than sequential runs. Use `-j 1` (default) for
> publication-quality measurements; use `-j N` for rapid iteration and CI.

In sequential mode, `prune_on_timeout` and `pin_cpu` work as expected.
In parallel mode, CPU pinning is disabled and pruning is not applied (grid
points are dispatched independently).

## Artifacts

All output is written to the `--out-dir` directory (default `artifacts/`):

| File              | Format | Contents                                      |
|-------------------|--------|-----------------------------------------------|
| `runs.jsonl`      | JSONL  | One JSON object per trial (status, wall_ms, reported_ms, peak_rss_mb, stdout, stderr) |
| `provenance.json` | JSON   | Seed, worker count, CLI invocation, working directory, and the benchmark machine's platform/CPU/memory |
| `summary.csv`     | CSV    | Median/mean/p10/p90 per grid point, for both the canonical (`time_ms_*`) and wall-clock (`wall_ms_*`) durations |
| `runtime.html`    | HTML   | Vega-Lite runtime chart with complexity overlay |
| `fits.csv`        | CSV    | Best-fit model, runner-up and margin, exponent CI, coefficients, RSS, and confidence per series |
| `report.html`     | HTML   | Full report with charts, tables, and system info |

## Complexity Fitting

Candidate models: **O(1)**, **O(log n)**, **O(√n)**, **O(n)**, **O(n log n)**, **O(n²)**, **O(n³)**, **O(n² 2ⁿ)**.

The fitting algorithm uses a multi-layer approach:

1. **Constant detection** — outlier-robust CV test; if data (or data minus any single outlier) has CV < 8%, classify as O(1).
2. **Log-log slope fallback** — for ≤ 3 data points with low dynamic range, where OLS lacks degrees of freedom.
3. **Step-up OLS** — fit all models via OLS (`y = C·f(n) + baseline`), start from the simplest valid model, accept more complex only if RSS improves by a dynamic-range-dependent factor.
4. **Tail-ratio guard** — for O(n) vs O(n log n) disambiguation, verify using the growth ratio at the two largest measured n values.
5. **Confidence assessment** — rate how well the measurements support the chosen class; see [Fit Confidence](#fit-confidence).

The selected model is shifted up to form a proper **upper bound** — the fit line sits at or above every observed data point, as Big-O semantics require. The plot shows the Big-O class (e.g. `O(n log n)`) on the curve, and the legend shows the concrete bound formula (e.g. `T(n) ≤ 5.36e-05·n·log(n) + 55.7`). Use `--complexity-strategy strict` to surface an empirical exponent band like `O(n^1.08±0.07)` when the confidence interval overlaps a neighboring class boundary.

```bash
tembench plot --summary artifacts/summary.csv --export-fits artifacts/fits.csv
tembench plot --summary artifacts/summary.csv --complexity-strategy strict
tembench plot --summary artifacts/summary.csv --no-fit      # disable overlay
tembench plot --summary artifacts/summary.csv --out-html -  # Vega-Lite JSON to stdout
```

`plot` prints the fitted class, its upper bound, and its confidence for every
series, so the headline result is visible without opening the HTML.

## Configuration Reference

```yaml
benchmarks:
  - name: my_benchmark          # identifier for this benchmark
    cmd: "my_program --size {n}" # command template; {keys} are expanded from grid
                                 # {python} is built in: this interpreter's path
    build: "make release"        # optional build step run once before trials
    workdir: "."                 # optional working directory
    env: { MY_VAR: "1" }        # optional environment variables

grid:
  n: [100, 1000, 10000]         # parameter grid; all combinations are swept

limits:
  timeout_sec: 30               # per-trial timeout (soft SIGTERM, then SIGKILL)
  warmups: 1                    # discarded warm-up runs per grid point
  repeats: 3                    # measured repetitions per grid point
  rss_poll_interval_sec: 0.01   # RSS sampling cadence for peak-memory tracking
  workers: 1                    # parallel workers (1 = sequential, default)
  prune_on_timeout: false       # after a timeout, skip larger growth_key values
                                # for that series only (other grid axes unaffected)
  shuffle: true                 # randomize sweep order to reduce drift
  growth_key: "n"               # grid key treated as the input size
  metric: auto                  # auto | wall | reported — see "Measuring the work"

pin_cpu: 0                      # optional CPU affinity (Linux, sequential only)
```

## License

[MIT](LICENSE)
