# TempoBench

A language-agnostic benchmarking CLI that runs any shell command with parameter sweeps, records wall-time and memory, estimates Big-O complexity, and generates reports—all from a single YAML config.

## Features

- **Parameter sweeps** — define grids of inputs and implementations; every combination is executed with configurable repeats, warm-ups, and timeouts.
- **Metrics collection** — wall-clock time and peak RSS are captured per trial using OS-native tools (`psutil`), with outlier filtering via Tukey fences.
- **Complexity estimation** — observed runtimes are fitted against O(1) … O(n³) models in log-space; the best model is selected by AIC and overlaid on plots.
- **Rich CLI output** — live progress bars, colored status tables, and system-info display powered by [Rich](https://github.com/Textualize/rich).
- **Reports & dashboards** — self-contained HTML reports with embedded Vega-Lite charts, heatmaps, comparison views, and regression detection.
- **Baseline comparison** — compare a new run against a previous baseline and flag regressions above a configurable threshold.
- **Reproducibility** — provenance snapshots record seeds, Python version, working directory, and hardware details alongside every run.

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
  n: [1000, 2000, 4000, 8000]
  impl: ["quadratic", "sort_scan", "hash_set"]

limits:
  timeout_sec: 20
  warmups: 0
  repeats: 2
```

**2. Run the pipeline:**

```bash
tembench run --config examples/unique_bench.yaml --out-dir artifacts
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

## Artifacts

All output is written to the `--out-dir` directory (default `artifacts/`):

| File              | Format | Contents                                      |
|-------------------|--------|-----------------------------------------------|
| `runs.jsonl`      | JSONL  | One JSON object per trial (status, wall_ms, peak_rss_mb, stdout, stderr) |
| `provenance.json` | JSON   | Seed, Python version, CLI invocation, working directory |
| `summary.csv`     | CSV    | Median/mean/p10/p90 per grid point            |
| `runtime.html`    | HTML   | Vega-Lite runtime chart with complexity overlay |
| `fits.csv`        | CSV    | Best-fit model, coefficients, and AIC per series |
| `report.html`     | HTML   | Full report with charts, tables, and system info |

## Complexity Fitting

Candidate models: **O(1)**, **O(log n)**, **O(n)**, **O(n log n)**, **O(n²)**, **O(n³)**.

The fitter selects the best growth class via OLS + AIC, then shifts the curve up to form a proper **upper bound** — the fit line always sits at or above every observed data point, as Big-O requires. The plot shows the Big-O class (e.g. `O(n log n)`) on the curve, and the legend shows the concrete bound formula (e.g. `T(n) ≤ 1.614e-05·n·log(n) + 37.3`).

```bash
tembench plot --summary artifacts/summary.csv --export-fits artifacts/fits.csv
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
  prune_on_timeout: false       # skip larger values after a timeout
  shuffle: true                 # randomize sweep order to reduce drift

pin_cpu: 0                      # optional CPU affinity (Linux only)
```

## License

[MIT](LICENSE)
