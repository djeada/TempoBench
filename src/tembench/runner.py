from __future__ import annotations

import json
import os
import random
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, List

import psutil

from .config import Benchmark, Config


def expand_grid(grid: Dict[str, List]) -> List[Dict[str, object]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    points = []
    for combo in product(*values) if values else [()]:
        params = dict(zip(keys, combo))
        points.append(params)
    return points


def format_cmd(template: str, params: Dict[str, object]) -> str:
    return template.format(**params)


def run_once(cmd: str, env: Dict[str, str], cwd: Path | None, timeout: float | None) -> Dict:
    start = time.perf_counter()
    ts = datetime.now(timezone.utc).isoformat()
    peak_rss = 0
    status = "ok"
    rc = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as out_f, tempfile.NamedTemporaryFile(mode="w+", delete=False) as err_f:
            out_path, err_path = out_f.name, err_f.name
        with open(out_path, "w") as out_handle, open(err_path, "w") as err_handle:
            with subprocess.Popen(
                shlex.split(cmd),
                cwd=str(cwd) if cwd else None,
                env={**os.environ, **env},
                stdout=out_handle,
                stderr=err_handle,
                text=True,
                preexec_fn=os.setsid,
            ) as proc:
                p = psutil.Process(proc.pid)
                # sample memory periodically to avoid tight loops; check for timeout
                while True:
                    if timeout is not None and (time.perf_counter() - start) > timeout:
                        status = "timeout"
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        except Exception:
                            pass
                        try:
                            proc.wait(2)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                            except Exception:
                                pass
                        break
                    # Track peak RSS for the process tree
                    try:
                        procs = [p] + p.children(recursive=True)
                        rss = sum(ch.memory_info().rss for ch in procs if ch.is_running())
                        peak_rss = max(peak_rss, rss)
                    except psutil.Error:
                        pass
                    if proc.poll() is not None:
                        break
                    time.sleep(0.01)
                rc = proc.returncode
        # read tail of outputs
        stdout_data = Path(out_path).read_text()[-10000:]
        stderr_data = Path(err_path).read_text()[-10000:]
        try:
            os.remove(out_path)
            os.remove(err_path)
        except Exception:
            pass
        if rc not in (0, None) and status == "ok":
            status = "failed"
        wall_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ts": ts,
            "status": status,
            "rc": rc,
            "wall_ms": round(wall_ms, 3),
            "peak_rss_mb": round(peak_rss / (1024**2), 3),
            "stdout": stdout_data,
            "stderr": stderr_data,
        }
    except FileNotFoundError as e:
        return {
            "ts": ts,
            "status": "error",
            "rc": None,
            "wall_ms": round((time.perf_counter() - start) * 1000.0, 3),
            "peak_rss_mb": round(peak_rss / (1024**2), 3),
            "stderr": str(e),
        }


def build_once(bench: Benchmark):
    if not bench.build:
        return
    subprocess.run(bench.build, shell=True, check=False, cwd=bench.workdir or None)


# ---------------------------------------------------------------------------
# Grid-point execution (one function for both serial & parallel paths)
# ---------------------------------------------------------------------------

def _run_grid_point(
    bench: Benchmark,
    params: Dict[str, object],
    timeout: float | None,
    warmups: int,
    repeats: int,
    retries: int,
) -> List[Dict]:
    """Execute warmups + repeats for a single (bench, params) combination.

    Returns a list of result records (one per repeat, plus any retry attempts).
    """
    cmd = format_cmd(bench.cmd, params)
    rec_base = {"bench": bench.name, "cmd": cmd, "params": params}
    cwd = Path(bench.workdir) if bench.workdir else None

    # Warmups
    for _ in range(max(0, warmups)):
        run_once(cmd, bench.env, cwd, timeout)

    # Repeats
    results: List[Dict] = []
    attempt = 0
    done = 0
    reps = max(1, repeats)
    while done < reps:
        attempt += 1
        rec = run_once(cmd, bench.env, cwd, timeout)
        rec.update(rec_base)
        results.append(rec)
        if rec["status"] == "ok":
            done += 1
        else:
            if attempt - done <= retries:
                continue
            else:
                done += 1  # count as a failed repetition
    return results


def run_benchmarks(
    cfg: Config,
    out_path: Path,
    seed: int = 42,
    retries: int = 0,
    on_trial: object = None,
    append: bool = False,
):
    """Run all benchmarks and write results.

    Args:
        on_trial: Optional callback(bench_name, params, rep, total_reps, result)
                  called after each trial for progress reporting.
    """
    workers = max(1, cfg.limits.workers)
    points = expand_grid(cfg.grid)
    if cfg.limits.shuffle:
        random.Random(seed).shuffle(points)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # optional CPU pinning (Linux, serial mode only)
    if cfg.pin_cpu is not None and workers == 1:
        try:
            p = psutil.Process()
            p.cpu_affinity([cfg.pin_cpu])
        except Exception:
            pass

    # provenance snapshot
    prov = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "cmdline": " ".join(shlex.quote(a) for a in sys.argv),
        "workers": workers,
    }
    (out_path.parent / "provenance.json").write_text(json.dumps(prov, indent=2))

    if workers == 1:
        _run_serial(cfg, out_path, points, retries, on_trial, append)
    else:
        _run_parallel(cfg, out_path, points, retries, on_trial, append, workers)


def _run_serial(
    cfg: Config,
    out_path: Path,
    points: List[Dict],
    retries: int,
    on_trial: object,
    append: bool,
):
    """Original sequential execution path — preserves pruning behaviour."""
    mode = "a" if append else "w"
    reps = max(1, cfg.limits.repeats)
    with out_path.open(mode) as f:
        for bench in cfg.benchmarks:
            build_once(bench)
            timed_out_keys: set = set()
            for params in points:
                gk = cfg.limits.growth_key
                key_val = params.get(gk) if gk else None
                if cfg.limits.prune_on_timeout and gk and key_val in timed_out_keys:
                    skip_rec = {
                        "bench": bench.name,
                        "cmd": format_cmd(bench.cmd, params),
                        "params": params,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "status": "skipped",
                    }
                    f.write(json.dumps(skip_rec) + "\n")
                    f.flush()
                    continue

                results = _run_grid_point(
                    bench, params, cfg.limits.timeout_sec,
                    cfg.limits.warmups, cfg.limits.repeats, retries,
                )
                for i, rec in enumerate(results):
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if on_trial:
                        on_trial(bench.name, params, i + 1, reps, rec)
                    if rec["status"] == "timeout" and cfg.limits.prune_on_timeout and gk and key_val is not None:
                        timed_out_keys.add(key_val)


def _run_parallel(
    cfg: Config,
    out_path: Path,
    points: List[Dict],
    retries: int,
    on_trial: object,
    append: bool,
    workers: int,
):
    """Parallel execution using ProcessPoolExecutor."""
    import threading

    mode = "a" if append else "w"
    reps = max(1, cfg.limits.repeats)
    lock = threading.Lock()

    with out_path.open(mode) as f:
        for bench in cfg.benchmarks:
            build_once(bench)

            # Submit all grid points to the pool
            with ProcessPoolExecutor(max_workers=workers) as pool:
                future_to_params = {}
                for params in points:
                    fut = pool.submit(
                        _run_grid_point,
                        bench, params, cfg.limits.timeout_sec,
                        cfg.limits.warmups, cfg.limits.repeats, retries,
                    )
                    future_to_params[fut] = params

                for fut in as_completed(future_to_params):
                    params = future_to_params[fut]
                    try:
                        results = fut.result()
                    except Exception as exc:
                        # If a worker crashes, record an error
                        results = [{
                            "bench": bench.name,
                            "cmd": format_cmd(bench.cmd, params),
                            "params": params,
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "status": "error",
                            "rc": None,
                            "wall_ms": 0.0,
                            "peak_rss_mb": 0.0,
                            "stderr": str(exc),
                        }]

                    with lock:
                        for i, rec in enumerate(results):
                            f.write(json.dumps(rec) + "\n")
                            f.flush()
                            if on_trial:
                                on_trial(bench.name, params, i + 1, reps, rec)
