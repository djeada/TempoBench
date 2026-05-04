"""Single-process subprocess execution: run a command once, capture metrics."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from ..config import Benchmark
from .process_group import popen_process_group_kwargs, terminate_process_group
from .result import TrialResult, TrialStatus


def run_once(
    cmd: str,
    env: dict[str, str],
    cwd: Path | None,
    timeout: float | None,
    poll_interval_sec: float = 0.01,
) -> TrialResult:
    start = time.perf_counter()
    ts = datetime.now(timezone.utc).isoformat()
    peak_rss = 0
    status: TrialStatus = "ok"
    rc = None
    sleep_interval = max(0.0, poll_interval_sec)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "stdout.txt"
            err_path = Path(tmpdir) / "stderr.txt"
            with out_path.open("w+") as out_handle, err_path.open("w+") as err_handle:
                popen_kwargs: dict[str, Any] = {
                    "cwd": str(cwd) if cwd else None,
                    "env": {**os.environ, **env},
                    "stdout": out_handle,
                    "stderr": err_handle,
                    "text": True,
                }
                popen_kwargs.update(popen_process_group_kwargs())
                with subprocess.Popen(shlex.split(cmd), **popen_kwargs) as proc:
                    p = psutil.Process(proc.pid)
                    while True:
                        if timeout is not None and (time.perf_counter() - start) > timeout:
                            status = "timeout"
                            terminate_process_group(proc)
                            break
                        try:
                            procs = [p] + p.children(recursive=True)
                            rss = sum(
                                ch.memory_info().rss for ch in procs if ch.is_running()
                            )
                            peak_rss = max(peak_rss, rss)
                        except psutil.Error:
                            pass
                        if proc.poll() is not None:
                            break
                        time.sleep(sleep_interval)
                    rc = proc.wait()

                out_handle.flush()
                err_handle.flush()
                out_handle.seek(0)
                err_handle.seek(0)
                stdout_data = out_handle.read()[-10000:]
                stderr_data = err_handle.read()[-10000:]

        if rc not in (0, None) and status == "ok":
            status = "failed"
        wall_ms = (time.perf_counter() - start) * 1000.0
        return TrialResult(
            ts=ts,
            status=status,
            rc=rc,
            wall_ms=round(wall_ms, 3),
            peak_rss_mb=round(peak_rss / (1024**2), 3),
            stdout=stdout_data,
            stderr=stderr_data,
        )
    except FileNotFoundError as e:
        return TrialResult(
            ts=ts,
            status="error",
            rc=None,
            wall_ms=round((time.perf_counter() - start) * 1000.0, 3),
            peak_rss_mb=round(peak_rss / (1024**2), 3),
            stdout="",
            stderr=str(e),
        )


def build_once(bench: Benchmark):
    if not bench.build:
        return
    subprocess.run(bench.build, shell=True, check=False, cwd=bench.workdir or None)
