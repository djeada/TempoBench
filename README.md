# TempoBench

* A language-agnostic runner and reporter executes any shell command while sweeping parameters and recording metrics, and if omitted you lose comparable evidence across variations; for example, using a *wrapper* lets you test `python sort.py --n 10000` and `./sort --n 10000` side by side.
* Parameter sweeps with retries and time limits give reproducible timing and memory records, and if omitted repeated runs can diverge silently; for example, enabling *timeouts* prevents a quadratic variant from hanging a batch at `n=1e7`.
* Observed curves are fitted to common classes for time and space and exported with tabular data and plots, and if omitted the growth pattern remains ambiguous; for example, estimating *complexity* reveals whether runtime follows `O(n log n)` on `n ∈ [1e3,1e6]`.

# CLI Orchestrator

* A focused set of subcommands—run, plot, report, compare, inspect—organizes the workflow end to end, and if omitted users invent ad-hoc steps; for example, using *commands* like `compare` highlights a 12% slowdown from last week.
* One declarative config in YAML/TOML defines benchmarks, parameter grids, repetitions, limits, and plotting preferences, and if omitted settings drift between runs; for example, a shared *configuration* with `n: [1e3,1e4,1e5]` keeps teams aligned.
* Environment controls for CPU pinning, warm-ups, working dirs, env vars, and optional container images improve stability, and if omitted noise masks regressions; for example, per-language *isolation* via Docker stops a new Python package from skewing C++ results.

# Executors (language-agnostic)

* A pure terminal runner executes each command exactly as written without hidden shims, and if omitted wrappers may skew timings; for example, a transparent *executor* treats `node app.js` and `go run main.go` identically.
* Optional build hooks run once per implementation to prepare binaries or bundles, and if omitted stale artifacts taint results; for example, a pre-step *build* with `cargo build --release` avoids measuring debug code.
* A per-run lifecycle covers setup, execution, metric capture, and teardown, and if omitted temp files and processes accumulate; for example, a defined *lifecycle* cleans cache directories between trials.
* Cross-platform collection records wall/CPU time and peak memory using native OS tools, and if omitted Windows and Linux numbers cannot be compared; for example, capturing peak *RSS* aligns `/usr/bin/time -v` with Windows APIs.
* A two-stage timeout sends a soft signal then a hard kill and marks the point as timed out, and if omitted a single job can block the sweep; for example, issuing *SIGTERM* then `SIGKILL` frees the queue when `n=1e8` explodes.

# Sampler & Statistics

* Multiple repetitions per point with robust summaries reduce fluctuation, and if omitted single shots mislead; for example, using the *median* and IQR stabilizes results on a noisy laptop.
* An explicit outlier policy flags but excludes anomalies from summaries, and if omitted extreme values warp charts; for example, Tukey fence *outliers* from background updates are retained for audit but not for medians.
* Default warm-ups discard the first run to minimize cold caches or JIT effects, and if omitted early runs look slower; for example, skipping the first *JIT* warm-up avoids overstating Node.js latency.
* Scaling guardrails prune growth when time or memory limits are met, and if omitted sweeps waste hours past useful ranges; for example, automatic *pruning* stops at the first timeout for a given algorithm branch.

# Metric Collectors

* Wall and CPU time are recorded per trial with precise clocks, and if omitted regressions hide under coarse timestamps; for example, consistent *wall-time* shows a 20 ms increase after a dependency bump.
* Peak memory uses OS measurements with optional language hooks, and if omitted leaks go unnoticed; for example, enabling Python’s *tracemalloc* correlates RSS peaks with allocation hotspots.
* Rich run metadata logs hardware, OS, git info, seeds, and tool versions, and if omitted reruns cannot be faithfully reproduced; for example, stored *provenance* reveals that turbo boost flipped mid-series.

# Complexity Estimator (observed)

* Candidate models from constant through exponential are fitted on log-transformed data with model selection, and if omitted growth class becomes guesswork; for example, choosing by *AIC* distinguishes `O(n)` from `O(n log n)` on broad ranges.
* Fit quality is reported with residuals and explicit warnings when evidence is weak, and if omitted charts overstate conclusions; for example, marking a fit as *inconclusive* when the `n` span is too narrow prevents misclaims.

# Data & Artifacts

* Raw results are written as JSONL and CSV with identifiers, parameters, timing, memory, status, and errors, and if omitted downstream tools cannot parse runs; for example, line-oriented *JSONL* streams into BigQuery easily.
* Derived summaries provide medians, percentiles, and timeout rates per point, and if omitted notebooks must recompute basics; for example, ready *percentiles* speed up dashboard creation.
* Reproducibility bundles save the config, command templates, seeds, and environment snapshot, and if omitted later audits stall; for example, an embedded *snapshot* captures the exact CPU model and Docker tags.

# Plotting & Reports

* A single plotting backend uses *Altair* to render clean line and heat maps with optional interactive HTML export, and if omitted teams juggle inconsistent charting libraries; for example, one Vega-Lite spec yields both PNG and HTML.
* Standard charts cover runtime vs size, memory vs size, fitted overlays, speedup ratios, commit trends, and timeout maps, and if omitted insights stay buried in tables; for example, enabling *log-log* axes reveals near-quadratic bends.
* The report command assembles a self-contained HTML dossier with charts and findings plus optional static images, and if omitted readers must stitch results manually; for example, a shareable *HTML* file summarizes a PR’s performance impact.

# Typical workflow

* Benchmarks are defined with command templates, parameter grids, repetitions, and limits, and if omitted ad-hoc runs become irreproducible; for example, templated *templates* like `--n {n} --impl {impl}` keep invocations uniform.
* Data collection runs from the config to produce raw logs and per-point summaries, and if omitted plots have nothing trustworthy to show; for example, saved *summaries* list median wall-ms by `n` and implementation.
* Plotting and reporting generate artifacts and an interactive overview for stakeholders, and if omitted reviewers lack visual evidence; for example, exported *SVG* charts embed crisply in docs.
* Optional comparison evaluates a current run against a chosen baseline to surface regressions, and if omitted slowdowns ship unnoticed; for example, selecting a previous *baseline* highlights a 1.3× slowdown in Python only.

# Extensibility

* A plugin interface allows custom steps for ecosystems like cargo, gradle, or GPU warm-ups, and if omitted niche builds are fragile; for example, a cargo *plugins* hook performs release builds before timing.
* Data adapters import logs from existing benchmark tools for unified analysis, and if omitted results remain siloed; for example, converting Google Benchmark via *adapters* enables shared plots.
* Lightweight CI modes run small grids on pull requests with thresholds for alerts, and if omitted regressions land unnoticed; for example, a 5% *regression* gate fails the job with an attached report.

# Reliability & fairness notes

* Single-core pinning reduces cross-trial variance in multi-tenant environments, and if omitted scheduler noise blurs results; for example, CPU *affinity* stabilizes latency on laptops.
* Recorded CPU frequency and turbo state provide context for speed shifts, and if omitted thermal effects look like code changes; for example, noting governor *scaling* explains a midday slowdown.
* Randomized point order mitigates drift from background activity over time, and if omitted monotone sweeps inherit bias; for example, shuffling reduces time-of-day *drift* in long campaigns.
* A short calibration benchmark reports ambient system variability, and if omitted false positives rise; for example, measuring background *noise* sets expectations for ±2% jitter.
* Visible dispersion metrics keep small deltas in perspective, and if omitted minor blips get overinterpreted; for example, plotting IQR-based *variance* warns when differences fall inside noise.

# Output at a glance

* A structured artifacts directory contains the config, raw runs, summaries, plots, and the report, and if omitted teammates hunt across machines; for example, centralized *artifacts* simplify attachment to PRs.
* Clear numeric statuses communicate success, partial success with timeouts/skips, config errors, and runtime failures, and if omitted automation cannot act on results; for example, standardized *exit codes* let CI fail fast on performance regressions.

## Quickstart

This repository now includes a minimal, runnable scaffold of the TempoBench CLI.

- Requirements: Python 3.10+ on Linux/macOS, with pip.
- Install in editable mode:

```
pip install -e .
```

- Try the included example (parameter sweep over input size and data shape):

```
tembench run --config examples/sort_bench.yaml --out-dir artifacts
tembench summarize --runs artifacts/runs.jsonl --out-csv artifacts/summary.csv
tembench plot --summary artifacts/summary.csv --out-html artifacts/runtime.html
```

Artifacts are written under `artifacts/`:
- `runs.jsonl`: raw per-trial results (status, wall_ms, peak_rss_mb, stderr/stdout)
- `summary.csv`: medians/means and counts per grid point
- `runtime.html`: an Altair plot of runtime vs `n`, colored by `impl`

Notes:
- CPU pinning is enabled in the example (`pin_cpu: 0`) when supported.
- Warm-ups and repeats are configurable under `limits`.
- The CLI also provides `inspect` to preview recent runs; `report` and `compare` are placeholders for future work.

## Complexity fit overlay

TempoBench can fit a Big-O class to observed runtimes and overlay it on the plot. Candidate models:

- O(1), O(log n), O(n), O(n log n), O(n^2), O(n^3)

The fitter linearizes in log space: log y = a + b log f(n) and selects the model with the lowest AIC.

Usage:

```
tembench plot --summary artifacts/summary.csv --out-html artifacts/runtime.html --export-fits artifacts/fits.csv
```

Options:
- `--no-fit` to disable the overlay
- `--export-fits PATH` to save per-series parameters and AIC

Limitations and tips:
- Ensure `y` is positive; defaults to `wall_ms_median`.
- Fits are per series (default by `impl`) and require at least two distinct n values.
- The model is descriptive of observed scaling; insufficient span in `n` may be inconclusive.
