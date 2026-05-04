# TempoBench

A language-agnostic benchmarking CLI that runs any shell command with parameter sweeps, records wall-time and memory, estimates Big-O complexity, and generates reports, all from a single YAML config.

<img width="1206" height="795" alt="Screenshot from 2026-02-12 21-53-22" src="https://github.com/user-attachments/assets/26c75949-de62-482d-8cfd-3d27db35eb95" />

## Features

- **Parameter sweeps** — define grids of inputs and implementations; every combination is executed with configurable repeats, warm-ups, and timeouts.
- **Parallel execution** — run grid points concurrently with `--workers`/`-j` to cut total benchmark time on multi-core machines.
- **Metrics collection** — wall-clock time and peak RSS are captured per trial using OS-native tools (`psutil`), with outlier filtering via Tukey fences.
- **Complexity estimation** — observed runtimes are fitted against O(1) … O(n³) models using a multi-layer algorithm (outlier-robust constant detection, log-log slope fallback, step-up OLS with adaptive thresholds, and tail-ratio guards). The best model is selected and overlaid on plots.
- **Interactive charts** — Vega-Lite charts with click-to-toggle legend, crosshair tooltips, and smooth fit curves. Data points shown as discrete markers, fit lines as smooth interpolated curves.
- **Rich CLI output** — live progress bars, colored status tables, and system-info display powered by [Rich](https://github.com/Textualize/rich).
- **Reports & dashboards** — self-contained HTML reports with embedded Vega-Lite charts, heatmaps, comparison views, and regression detection.
- **Baseline comparison** — compare a new run against a previous baseline and flag regressions above a configurable threshold.
- **Reproducibility** — provenance snapshots record seeds, Python version, worker count, working directory, and hardware details alongside every run.

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

For development (adds pytest and ruff):

```bash
pip install -e ".[dev]"
```

## Quickstart

**1. Define a benchmark** in YAML (see [`examples/unique_bench.yaml`](examples/unique_bench.yaml)):

```yaml
benchmarks:
  - name: py_unique_count
    cmd: "python examples/unique_impl.py --n {n} --impl {impl} --seed 42"

grid:
  n: [10000, 50000, 100000, 500000, 1000000, 5000000]
  impl: ["quadratic", "sort_scan", "hash_set"]

limits:
  timeout_sec: 30
  warmups: 0
  repeats: 2
```

**2. Run the pipeline:**

```bash
# Sequential (default)
tembench run --config examples/unique_bench.yaml --out-dir artifacts

# Parallel — 4 workers for faster execution
tembench run --config examples/unique_bench.yaml --out-dir artifacts -j 4

tembench summarize --runs artifacts/runs.jsonl --out-csv artifacts/summary.csv
tembench plot --summary artifacts/summary.csv --out-html artifacts/runtime.html
```

**3. Generate a full report:**

```bash
tembench report --summary artifacts/summary.csv
```

## CLI Commands

| Command       | Description                                                |
|---------------|------------------------------------------------------------|
| `run`         | Execute benchmarks and write JSONL results                 |
| `summarize`   | Aggregate runs into CSV with medians, percentiles, counts  |
| `plot`        | Generate a runtime chart with optional Big-O fit overlay   |
| `report`      | Build a self-contained HTML report                         |
| `compare`     | Detect regressions against a baseline summary              |
| `dashboard`   | Create a multi-chart interactive dashboard                 |
| `inspect`     | Preview recent runs in a table                             |
| `memory`      | Generate a memory-usage chart                              |
| `heatmap`     | Generate a performance heatmap                             |
| `sysinfo`     | Display system information for reproducibility             |

Run `tembench --help` or `tembench <command> --help` for full option details.

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
| `runs.jsonl`      | JSONL  | One JSON object per trial (status, wall_ms, peak_rss_mb, stdout, stderr) |
| `provenance.json` | JSON   | Seed, Python version, worker count, CLI invocation, working directory |
| `summary.csv`     | CSV    | Median/mean/p10/p90 per grid point            |
| `runtime.html`    | HTML   | Vega-Lite runtime chart with complexity overlay |
| `fits.csv`        | CSV    | Best-fit model, exponent CI, coefficients, and RSS per series |
| `report.html`     | HTML   | Full report with charts, tables, and system info |

## Complexity Fitting

Candidate models: **O(1)**, **O(log n)**, **O(n)**, **O(n log n)**, **O(n²)**, **O(n³)**.

The fitting algorithm uses a multi-layer approach:

1. **Constant detection** — outlier-robust CV test; if data (or data minus any single outlier) has CV < 8%, classify as O(1).
2. **Log-log slope fallback** — for ≤ 3 data points with low dynamic range, where OLS lacks degrees of freedom.
3. **Step-up OLS** — fit all models via OLS (`y = C·f(n) + baseline`), start from the simplest valid model, accept more complex only if RSS improves by a dynamic-range-dependent factor.
4. **Tail-ratio guard** — for O(n) vs O(n log n) disambiguation, verify using the growth ratio at the two largest measured n values.

The selected model is shifted up to form a proper **upper bound** — the fit line sits at or above every observed data point, as Big-O semantics require. The plot shows the Big-O class (e.g. `O(n log n)`) on the curve, and the legend shows the concrete bound formula (e.g. `T(n) ≤ 5.36e-05·n·log(n) + 55.7`). Use `--complexity-strategy strict` to surface an empirical exponent band like `O(n^1.08±0.07)` when the confidence interval overlaps a neighboring class boundary.

```bash
tembench plot --summary artifacts/summary.csv --export-fits artifacts/fits.csv
tembench plot --summary artifacts/summary.csv --complexity-strategy strict
tembench plot --summary artifacts/summary.csv --no-fit   # disable overlay
```

## Configuration Reference

```yaml
benchmarks:
  - name: my_benchmark          # identifier for this benchmark
    cmd: "my_program --size {n}" # command template; {keys} are expanded from grid
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
  prune_on_timeout: false       # skip larger values after a timeout
  shuffle: true                 # randomize sweep order to reduce drift

pin_cpu: 0                      # optional CPU affinity (Linux, sequential only)
```

## License

[MIT](LICENSE)
