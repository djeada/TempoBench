from __future__ import annotations

import json
import sys
import os
import random
import shlex
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List

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
    err = None
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


def run_benchmarks(cfg: Config, out_path: Path, seed: int = 42, retries: int = 0):
    points = expand_grid(cfg.grid)
    if cfg.limits.shuffle:
        random.Random(seed).shuffle(points)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # optional CPU pinning (Linux)
    if cfg.pin_cpu is not None:
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
    }
    (out_path.parent / "provenance.json").write_text(json.dumps(prov, indent=2))

    with out_path.open("a") as f:
        for bench in cfg.benchmarks:
            build_once(bench)
            timed_out_keys = set()
            for params in points:
                cmd = format_cmd(bench.cmd, params)
                rec_base = {
                    "bench": bench.name,
                    "cmd": cmd,
                    "params": params,
                }
                # prune if growth key exceeded after timeout
                gk = cfg.limits.growth_key
                key_val = params.get(gk) if gk else None
                if cfg.limits.prune_on_timeout and gk and key_val in timed_out_keys:
                    skip_rec = {**rec_base, "ts": datetime.now(timezone.utc).isoformat(), "status": "skipped"}
                    f.write(json.dumps(skip_rec) + "\n")
                    f.flush()
                    continue
                # warmups
                for _ in range(max(0, cfg.limits.warmups)):
                    run_once(cmd, bench.env, Path(bench.workdir) if bench.workdir else None, cfg.limits.timeout_sec)
                # repeats
                reps = max(1, cfg.limits.repeats)
                attempt = 0
                done = 0
                while done < reps:
                    attempt += 1
                    rec = run_once(cmd, bench.env, Path(bench.workdir) if bench.workdir else None, cfg.limits.timeout_sec)
                    rec.update(rec_base)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if rec["status"] == "ok":
                        done += 1
                    else:
                        if rec["status"] == "timeout" and cfg.limits.prune_on_timeout and gk and key_val is not None:
                            timed_out_keys.add(key_val)
                        if attempt - done <= retries:
                            continue
                        else:
                            done += 1  # count as a failed repetition
