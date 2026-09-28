"""Single-process subprocess execution: run a command once, capture metrics."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

import psutil

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]

from ..command import split_command
from ..config import Benchmark
from .process_group import (
    kill_leftover_group,
    popen_process_group_kwargs,
    terminate_process_group,
)
from .reported import parse_reported_ms
from .result import TrialResult, TrialStatus

#: How much of each output stream is kept.  Only the tail matters: it holds the
#: self-reported timing and the last error message.
OUTPUT_TAIL_BYTES = 10000

#: Descendants are looked for at most this often.  Finding them means scanning
#: every process on the machine (~3 ms), and measuring them without counting
#: shared pages twice means reading their page maps (~7 ms per GB), so doing
#: either at the default 10 ms cadence would steal a sizeable share of a core
#: from the benchmark.  A single process's peak is exact regardless (see
#: `_ExitWatcher`); only memory held by several processes *at once* is sampled.
TREE_SAMPLE_INTERVAL_SEC = 0.1

# Linux and the BSDs report ru_maxrss in KiB; macOS reports bytes.
_MAXRSS_UNIT = 1 if sys.platform == "darwin" else 1024


class _ExitWatcher:
    """Block on the child's exit in a thread, so the exit is stamped when it happens.

    Polling for the exit would round every duration up to the poll interval and
    add the cost of whatever sampling ran just before it.  On POSIX the child is
    reaped with `wait4`, whose rusage carries the kernel's own high-water mark
    of resident memory, blind to no spike however short.  It covers the child
    and the largest descendant it waited for — but the child's figure also
    includes the image it replaced at exec, which (vfork or fork) is the runner
    itself.  It is therefore only trusted when it exceeds the runner's own peak.
    """

    def __init__(self, proc: subprocess.Popen[bytes]):
        self.proc = proc
        self.end: float | None = None
        self.maxrss_bytes = 0
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._wait, daemon=True)
        self._thread.start()

    def _wait(self) -> None:
        try:
            if hasattr(os, "wait4") and resource is not None:
                try:
                    _, status, usage = os.wait4(self.proc.pid, 0)
                except ChildProcessError:  # already reaped elsewhere
                    self.end = time.perf_counter()
                    self.proc.wait()
                else:
                    self.end = time.perf_counter()
                    # Tell Popen the child is gone, or its own wait would find
                    # nothing to reap and report a clean exit.
                    self.proc.returncode = os.waitstatus_to_exitcode(status)
                    maxrss = usage.ru_maxrss * _MAXRSS_UNIT
                    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    if maxrss > own * _MAXRSS_UNIT:
                        self.maxrss_bytes = maxrss
            else:
                self.proc.wait()
                self.end = time.perf_counter()
        finally:
            if self.end is None:
                self.end = time.perf_counter()
            self._done.set()

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout)


class _MemorySampler:
    """Track peak resident memory of the child and, coarsely, of its whole tree."""

    def __init__(self, pid: int, start: float, poll_interval_sec: float):
        self.peak = 0
        try:
            self._root: psutil.Process | None = psutil.Process(pid)
        except psutil.Error:
            self._root = None
        status = Path(f"/proc/{pid}/status")
        self._status_path = status if sys.platform.startswith("linux") else None
        self._next_tree = start + poll_interval_sec
        self._tree_interval = max(poll_interval_sec, TREE_SAMPLE_INTERVAL_SEC)

    def sample(self, now: float) -> None:
        root = self._root
        if root is None:
            return
        try:
            self.peak = max(self.peak, self._root_peak(root))
        except (psutil.Error, OSError, ValueError):
            return
        if now < self._next_tree:
            return
        self._next_tree = now + self._tree_interval
        try:
            children = root.children(recursive=True)
        except psutil.Error:
            return
        if not children:
            return
        # Summing rss over a tree counts every copy-on-write page shared after
        # a fork once per process.  PSS splits shared pages between the
        # processes mapping them, so the sum is the real total (Linux).  Where
        # PSS is unavailable, count shared pages with the root only.
        total = self._unique_bytes(root, is_root=True)
        for child in children:
            total += self._unique_bytes(child)
        self.peak = max(self.peak, total)

    def _root_peak(self, root: psutil.Process) -> int:
        """Return the child's own high-water mark where the OS keeps one.

        Linux and Windows track the peak since exec per process, so a spike
        between two samples is still seen as long as the process outlives it;
        elsewhere only the current resident size is available.
        """
        if self._status_path is not None:
            with open(self._status_path, "rb") as fh:
                for line in fh:
                    if line.startswith(b"VmHWM:"):
                        return int(line.split()[1]) * 1024
            return 0
        info = root.memory_info()
        return max(getattr(info, "peak_wset", 0), info.rss)

    @staticmethod
    def _unique_bytes(proc: psutil.Process, is_root: bool = False) -> int:
        try:
            full = proc.memory_full_info()
        except psutil.Error:
            return 0
        pss = getattr(full, "pss", None)
        if pss is not None:
            return int(pss)
        if is_root:
            return int(full.rss)
        return int(getattr(full, "uss", 0))


def _read_tail(handle: IO[bytes]) -> str:
    """Decode the last `OUTPUT_TAIL_BYTES` of a captured stream.

    Benchmarks may print anything, including bytes that are not UTF-8; that must
    not cost the measurement, so undecodable bytes are replaced.
    """
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(max(0, size - OUTPUT_TAIL_BYTES))
    data = handle.read()
    if size > OUTPUT_TAIL_BYTES:
        # Do not start mid-character.
        data = data.lstrip(bytes(range(0x80, 0xC0)))
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


def _error(ts: str, message: str) -> TrialResult:
    return TrialResult(ts=ts, status="error", rc=None, stdout="", stderr=message)


def run_once(
    cmd: str,
    env: dict[str, str],
    cwd: Path | None,
    timeout: float | None,
    poll_interval_sec: float = 0.01,
) -> TrialResult:
    ts = datetime.now(timezone.utc).isoformat()
    try:
        argv = split_command(cmd)
    except ValueError as e:  # e.g. an unbalanced quote
        return _error(ts, f"cannot parse command {cmd!r}: {e}")

    status: TrialStatus = "ok"
    sleep_interval = max(0.0, poll_interval_sec)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "stdout.txt"
        err_path = Path(tmpdir) / "stderr.txt"
        with out_path.open("w+b") as out_handle, err_path.open("w+b") as err_handle:
            popen_kwargs: dict[str, Any] = {
                "cwd": str(cwd) if cwd else None,
                "env": {**os.environ, **env},
                "stdout": out_handle,
                "stderr": err_handle,
            }
            popen_kwargs.update(popen_process_group_kwargs())
            start = time.perf_counter()
            try:
                proc = subprocess.Popen(argv, **popen_kwargs)
            except (OSError, ValueError) as e:
                # Missing program, no execute permission, bad cwd, NUL in args...
                return _error(ts, str(e))

            with proc:
                watcher = _ExitWatcher(proc)
                memory = _MemorySampler(proc.pid, start, sleep_interval)
                try:
                    while True:
                        memory.sample(time.perf_counter())
                        if watcher.wait(sleep_interval):
                            break
                        if timeout is not None and time.perf_counter() - start > timeout:
                            status = "timeout"
                            terminate_process_group(proc, wait_exit=watcher.wait)
                            break
                except BaseException:
                    # Ctrl-C, or the runner being killed mid-sweep.  The child
                    # runs in its own process group so signals do not reach it,
                    # and `Popen.__exit__` only *waits* — without this, aborting
                    # a sweep would hang on the current trial and, if the parent
                    # died anyway, leave a heavy benchmark orphaned.
                    terminate_process_group(proc, wait_exit=watcher.wait)
                    raise
                finally:
                    kill_leftover_group(proc)
                watcher.wait(5.0)
                end = watcher.end if watcher.end is not None else time.perf_counter()
                wall_ms = (end - start) * 1000.0
                rc = proc.returncode

            stdout_data = _read_tail(out_handle)
            stderr_data = _read_tail(err_handle)

    if rc not in (0, None) and status == "ok":
        status = "failed"
    reported_ms = parse_reported_ms(stdout_data) if status == "ok" else None
    peak_rss = max(memory.peak, watcher.maxrss_bytes)
    return TrialResult(
        ts=ts,
        status=status,
        rc=rc,
        wall_ms=round(wall_ms, 3),
        reported_ms=None if reported_ms is None else round(reported_ms, 6),
        # No reading at all is not the same as a process that used no memory.
        peak_rss_mb=round(peak_rss / (1024**2), 3) if peak_rss else None,
        stdout=stdout_data,
        stderr=stderr_data,
    )


def build_once(bench: Benchmark) -> str | None:
    """Run the benchmark's build step; return why it failed, or None.

    The output is captured rather than inherited: it would otherwise scroll
    over the progress bar, and when the build fails its last line is the most
    useful thing to show.
    """
    if not bench.build:
        return None
    try:
        proc = subprocess.run(
            bench.build,
            shell=True,
            check=False,
            cwd=bench.workdir or None,
            env={**os.environ, **bench.env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as e:
        return f"build step {bench.build!r} could not start: {e}"
    if proc.returncode == 0:
        return None
    output = proc.stdout.decode("utf-8", errors="replace").strip().splitlines()
    last = output[-1].strip() if output else "no output"
    return f"build step {bench.build!r} failed (exit {proc.returncode}): {last}"
