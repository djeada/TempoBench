"""Regression tests for how a single trial is launched, timed, and measured."""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest
from typer.testing import CliRunner

from tembench.cli import app
from tembench.config import Benchmark, Config, Limits
from tembench.runner import _run_grid_point, format_cmd, run_benchmarks, run_once
from tembench.runner.process import OUTPUT_TAIL_BYTES

posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX process semantics")
linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="memory figures checked on Linux"
)

PY = sys.executable


def _script(tmp_path: Path, body: str, name: str = "prog.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


# ---------- timing ----------


def test_wall_time_is_not_rounded_up_to_the_poll_interval():
    """Exit is noticed when it happens, not at the next memory sample."""
    result = run_once("sleep 0.05", {}, None, timeout=5.0, poll_interval_sec=0.5)
    assert result.status == "ok"
    assert result.wall_ms is not None
    assert 45 <= result.wall_ms < 250, result.wall_ms


# ---------- memory ----------


@linux_only
def test_forked_children_do_not_multiply_shared_memory(tmp_path: Path):
    """Copy-on-write pages shared after fork are one allocation, not four."""
    script = _script(tmp_path, """
        import os, time
        block = b"x" * (150 * 2**20)
        pids = []
        for _ in range(3):
            pid = os.fork()
            if pid == 0:
                time.sleep(0.35)
                os._exit(0)
            pids.append(pid)
        for pid in pids:
            os.waitpid(pid, 0)
    """)
    result = run_once(f"{PY} {script}", {}, None, timeout=10.0)
    assert result.status == "ok"
    assert result.peak_rss_mb is not None
    assert 140 <= result.peak_rss_mb < 300, result.peak_rss_mb


@linux_only
def test_a_spike_between_samples_is_still_counted(tmp_path: Path):
    script = _script(tmp_path, """
        import time
        block = bytearray(150 * 2**20)
        del block
        time.sleep(0.4)
    """)
    result = run_once(f"{PY} {script}", {}, None, timeout=10.0, poll_interval_sec=0.25)
    assert result.peak_rss_mb is not None and result.peak_rss_mb >= 140


def test_an_unmeasured_process_has_no_memory_reading_rather_than_zero():
    result = run_once("true" if os.name != "nt" else "cmd /c exit 0", {}, None, 5.0)
    assert result.status == "ok"
    assert result.peak_rss_mb is None or result.peak_rss_mb > 0


# ---------- process handling ----------


def test_output_that_is_not_utf8_does_not_crash_the_sweep(tmp_path: Path):
    script = _script(tmp_path, """
        import sys
        sys.stdout.buffer.write(b"bad \\xff\\xfe bytes\\n")
    """)
    cfg = Config(
        benchmarks=[Benchmark(name="bytes", cmd=f"{PY} {script} {{n}}")],
        grid={"n": [1]},
        limits=Limits(warmups=0, repeats=1, shuffle=False),
    )
    out = tmp_path / "runs.jsonl"
    run_benchmarks(cfg, out)
    rec = json.loads(out.read_text())
    assert rec["status"] == "ok"
    assert "bad" in rec["stdout"] and "�" in rec["stdout"]


def test_only_the_tail_of_a_large_output_is_kept(tmp_path: Path):
    script = _script(tmp_path, """
        for i in range(200000):
            print(i)
        print("TEMPOBENCH_MS: 7")
    """)
    result = run_once(f"{PY} {script}", {}, None, 10.0)
    assert result.stdout is not None
    assert len(result.stdout.encode()) <= OUTPUT_TAIL_BYTES
    assert result.reported_ms == 7


@posix_only
def test_a_command_that_cannot_be_executed_is_an_error_not_a_crash(tmp_path: Path):
    script = tmp_path / "noexec.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o644)
    result = run_once(str(script), {}, None, 5.0)
    assert result.status == "error"
    assert "ermission" in (result.stderr or "")


@posix_only
def test_background_processes_are_stopped_when_the_benchmark_exits():
    result = run_once("sh -c 'sleep 30 & echo $!'", {}, None, 5.0)
    assert result.status == "ok"
    pid = int((result.stdout or "").strip())
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"background process {pid} outlived the benchmark")


# ---------- command substitution ----------


def test_grid_values_are_substituted_as_single_arguments(tmp_path: Path):
    assert format_cmd("prog {x}", {"x": "two words"}) != "prog two words"
    script = _script(tmp_path, "import sys, json; print(json.dumps(sys.argv[1:]))\n")
    for value in ("two words", "it's", 'say "hi"'):
        cmd = format_cmd(f"{PY} {script} {{x}} {{n:03d}}", {"x": value, "n": 7})
        result = run_once(cmd, {}, None, 5.0)
        assert result.status == "ok", result.stderr
        assert json.loads(result.stdout or "") == [value, "007"]


def test_raw_substitution_splices_a_whole_command():
    assert format_cmd("{c:raw}", {"c": "echo a b"}) == "echo a b"


@posix_only
def test_shell_pipelines_run_through_sh_with_positional_arguments():
    """The documented way to use a pipe, which survives any grid value."""
    cmd = format_cmd("""sh -c 'printf %s "$1" | wc -c' sh {x}""", {"x": "it's"})
    result = run_once(cmd, {}, None, 5.0)
    assert result.status == "ok", result.stderr
    assert (result.stdout or "").strip() == "4"


def test_a_command_that_cannot_be_split_is_a_trial_error():
    result = run_once("echo 'unbalanced", {}, None, 5.0)
    assert result.status == "error"
    assert "cannot parse" in (result.stderr or "")


# ---------- retries, pruning, and record shape ----------


def _flaky(tmp_path: Path) -> Path:
    """A command that fails every other launch."""
    counter = tmp_path / "count"
    return _script(tmp_path, f"""
        import pathlib, sys
        p = pathlib.Path({str(counter)!r})
        n = int(p.read_text()) if p.exists() else 0
        p.write_text(str(n + 1))
        sys.exit(1 if n % 2 == 0 else 0)
    """, name="flaky.py")


def test_retries_are_budgeted_per_repetition_and_only_the_last_attempt_is_kept(
    tmp_path: Path,
):
    bench = Benchmark(name="flaky", cmd=f"{PY} {_flaky(tmp_path)}")
    results = _run_grid_point(bench, {}, 5.0, warmups=0, repeats=3, retries=1)
    assert [r.status for r in results] == ["ok", "ok", "ok"]
    assert [r.attempts for r in results] == [2, 2, 2]
    assert all(r.to_dict()["attempts"] == 2 for r in results)


def test_a_run_whose_retries_all_succeeded_exits_zero(tmp_path: Path):
    cfg = tmp_path / "bench.yaml"
    cfg.write_text(textwrap.dedent(f"""
        benchmarks:
          - name: flaky
            cmd: "{PY} {_flaky(tmp_path)} {{n}}"
        grid:
          n: [1, 2]
        limits:
          warmups: 0
          repeats: 2
    """))
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app, ["run", "--config", str(cfg), "--out-dir", str(out), "--retries", "1"]
    )
    assert result.exit_code == 0, result.output
    lines = (out / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert "needed a retry" in result.output


def test_a_timeout_skips_the_rest_of_the_point_when_pruning(tmp_path: Path):
    bench = Benchmark(name="slow", cmd="sleep 5")
    results = _run_grid_point(
        bench, {"n": 1}, 0.1, warmups=1, repeats=3, retries=0, prune_on_timeout=True
    )
    assert [r.status for r in results] == ["timeout", "skipped", "skipped"]


def test_every_record_carries_the_configured_metric(tmp_path: Path):
    cfg = Config(
        benchmarks=[Benchmark(name="m", cmd="echo {n}")],
        grid={"n": [1]},
        limits=Limits(warmups=0, repeats=1, shuffle=False, metric="wall"),
    )
    out = tmp_path / "runs.jsonl"
    run_benchmarks(cfg, out)
    assert json.loads(out.read_text())["metric"] == "wall"


# ---------- build step ----------


@pytest.mark.parametrize("workers", ["1", "2"])
def test_a_failing_build_fails_the_run(tmp_path: Path, workers: str):
    cfg = tmp_path / "bench.yaml"
    cfg.write_text(textwrap.dedent("""
        benchmarks:
          - name: compiled
            build: "echo compiling; echo 'error: no such file' >&2; exit 3"
            cmd: "echo {n}"
        grid:
          n: [1, 2]
        limits:
          warmups: 0
          repeats: 2
    """))
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app, ["run", "--config", str(cfg), "--out-dir", str(out), "-j", workers]
    )
    assert result.exit_code != 0
    records = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    # One record per would-be repetition, so the plan still adds up.
    assert len(records) == 4
    assert {r["status"] for r in records} == {"error"}
    assert "exit 3" in records[0]["stderr"] and "no such file" in records[0]["stderr"]
    assert "build step" in result.output


# ---------- audit regressions ----------


def test_marker_is_found_however_much_is_printed_after_it(tmp_path: Path):
    # Only the tail of stdout is kept; the marker must not be lost with the rest.
    prog = _script(tmp_path, f"""
        print("TEMPOBENCH_MS: 1.5")
        print("x" * {OUTPUT_TAIL_BYTES * 3})
    """)
    result = run_once(f"{PY} {prog}", {}, None, timeout=10.0)
    assert result.status == "ok"
    assert result.reported_ms == 1.5
    assert len(result.stdout or "") <= OUTPUT_TAIL_BYTES


@posix_only
def test_timeout_does_not_wait_for_the_next_memory_sample():
    started = time.perf_counter()
    result = run_once("sleep 5", {}, None, timeout=0.2, poll_interval_sec=2.0)
    assert result.status == "timeout"
    assert time.perf_counter() - started < 1.5


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux CPU affinity")
def test_pin_cpu_pins_the_benchmark_but_not_the_runner(tmp_path: Path):
    cpu = max(os.sched_getaffinity(0))
    before = os.sched_getaffinity(0)
    prog = _script(tmp_path, "import os; print('CPUS', sorted(os.sched_getaffinity(0)))")
    result = run_once(f"{PY} {prog}", {}, None, timeout=10.0, cpu=cpu)
    assert f"CPUS [{cpu}]" in (result.stdout or "")
    assert os.sched_getaffinity(0) == before


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux CPU affinity")
def test_unavailable_pin_cpu_is_a_config_error(tmp_path: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("benchmarks:\n  - name: b\n    cmd: 'true'\npin_cpu: 4096\n")
    result = CliRunner().invoke(app, ["run", "--config", str(cfg), "--out-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "CPU 4096 is not available" in result.output
    assert "Traceback" not in result.output


def test_invalid_config_is_reported_without_a_traceback(tmp_path: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("benchmarks:\n  - name: b\n    cmd: 'echo {nope}'\n")
    for command in ("run", "validate"):
        result = CliRunner().invoke(app, [command, "--config", str(cfg)])
        assert result.exit_code == 1
        assert "Invalid config" in result.output and "nope" in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)


def test_windows_quoting_doubles_backslashes_before_a_closing_quote(monkeypatch):
    from tembench import command

    monkeypatch.setattr(command, "WINDOWS", True)
    assert command.quote_argument("C:\\a b\\") == '"C:\\a b\\\\"'
    assert command.quote_argument("plain") == "plain"
    assert command.quote_argument("") == '""'
