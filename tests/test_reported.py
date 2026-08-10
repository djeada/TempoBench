from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from tembench.config import Benchmark
from tembench.placeholders import builtin_placeholders
from tembench.runner.grid import _run_grid_point, format_cmd
from tembench.runner.process import run_once
from tembench.runner.reported import parse_reported_ms


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ("TEMPOBENCH_MS: 12.5\n", 12.5),
        ("TEMPOBENCH_MS 12.5\n", 12.5),
        ("TEMPOBENCH_MS=12.5\n", 12.5),
        ("  TEMPOBENCH_MS:   12.5  \n", 12.5),
        ("result=7\nTEMPOBENCH_MS: 0.125\n", 0.125),
        ("TEMPOBENCH_MS: 1.5e2\n", 150.0),
        ("TEMPOBENCH_MS: 0\n", 0.0),
        ("TEMPOBENCH_MS: .5\n", 0.5),
        ("TEMPOBENCH_MS: 3\n", 3.0),
    ],
)
def test_parse_reported_ms_accepts_marker_variants(stdout: str, expected: float):
    assert parse_reported_ms(stdout) == expected


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "no marker here\n",
        "TEMPOBENCH_MS: abc\n",
        "TEMPOBENCH_MS: -1\n",
        "TEMPOBENCH_MS: nan\n",
        "tempobench_ms: 5\n",  # case-sensitive on purpose
        "prefix TEMPOBENCH_MS: 5\n",  # must be alone on its line
        "TEMPOBENCH_MS: 5 trailing\n",
    ],
)
def test_parse_reported_ms_rejects_non_markers(stdout: str):
    assert parse_reported_ms(stdout) is None


def test_parse_reported_ms_takes_the_last_marker():
    assert parse_reported_ms("TEMPOBENCH_MS: 1.0\nTEMPOBENCH_MS: 2.0\n") == 2.0


def test_format_cmd_expands_builtin_python_placeholder():
    cmd = format_cmd("{python} script.py --n {n}", {"n": 5})
    assert cmd == f"{builtin_placeholders()['python']} script.py --n 5"


def test_format_cmd_lets_the_grid_shadow_a_builtin():
    cmd = format_cmd("{python} script.py", {"python": "pypy3"})
    assert cmd == "pypy3 script.py"


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "prog.py"
    path.write_text(textwrap.dedent(body))
    return path


def test_reported_metric_captures_self_reported_duration(tmp_path: Path):
    script = _script(tmp_path, """
        print("work done")
        print("TEMPOBENCH_MS: 4.25")
    """)
    bench = Benchmark(name="t", cmd=f"{sys.executable} {script}")
    results = _run_grid_point(bench, {}, None, 0, 1, 0, 0.01, "reported")
    assert [r.status for r in results] == ["ok"]
    reported_ms, wall_ms = results[0].reported_ms, results[0].wall_ms
    assert reported_ms == 4.25
    # Wall time still measures the whole process, and includes startup.
    assert wall_ms is not None and wall_ms > reported_ms


def test_reported_metric_fails_a_trial_without_a_marker(tmp_path: Path):
    script = _script(tmp_path, 'print("work done")\n')
    bench = Benchmark(name="t", cmd=f"{sys.executable} {script}")
    results = _run_grid_point(bench, {}, None, 0, 1, 0, 0.01, "reported")
    assert [r.status for r in results] == ["failed"]
    assert "TEMPOBENCH_MS" in (results[0].stderr or "")


def test_auto_metric_tolerates_a_missing_marker(tmp_path: Path):
    script = _script(tmp_path, 'print("work done")\n')
    bench = Benchmark(name="t", cmd=f"{sys.executable} {script}")
    results = _run_grid_point(bench, {}, None, 0, 1, 0, 0.01, "auto")
    assert [r.status for r in results] == ["ok"]
    assert results[0].reported_ms is None


def test_reported_duration_survives_the_jsonl_round_trip(tmp_path: Path):
    script = _script(tmp_path, 'print("TEMPOBENCH_MS: 1.5")\n')
    bench = Benchmark(name="t", cmd=f"{sys.executable} {script}")
    rec = _run_grid_point(bench, {}, None, 0, 1, 0, 0.01, "reported")[0]
    from tembench.runner.result import TrialResult

    assert TrialResult.from_dict(rec.to_dict()).reported_ms == 1.5


def test_posix_commands_are_tokenised_for_exec():
    from tembench.command import split_command

    if sys.platform == "win32":
        pytest.skip("POSIX tokenisation only")
    assert split_command("prog --n 5") == ["prog", "--n", "5"]
    assert split_command("'a b/prog' --n 5") == ["a b/prog", "--n", "5"]


def test_windows_commands_are_passed_through_untouched(monkeypatch):
    """POSIX splitting eats backslashes, turning C:\\tools\\p.exe into C:toolsp.exe."""
    import tembench.command as command

    monkeypatch.setattr(command, "WINDOWS", True)
    assert command.split_command(r"C:\tools\prog.exe --n 5") == r"C:\tools\prog.exe --n 5"


def test_windows_quoting_uses_double_quotes(monkeypatch):
    import tembench.command as command

    monkeypatch.setattr(command, "WINDOWS", True)
    assert command.quote_argument(r"C:\Python\python.exe") == r"C:\Python\python.exe"
    assert command.quote_argument(r"C:\Program Files\python.exe") == r'"C:\Program Files\python.exe"'


def test_interrupting_a_trial_kills_the_child(tmp_path: Path, monkeypatch):
    """Ctrl-C must not hang on the current trial, nor orphan it.

    The child runs in its own process group so terminal signals never reach it;
    if the runner does not kill it explicitly, a long benchmark keeps running
    with nothing watching it.
    """
    import time as _time

    import psutil

    import tembench.runner.process as process_mod

    script = _script(tmp_path, """
        import time
        time.sleep(60)
    """)

    seen: list[int] = []
    real_sleep = _time.sleep

    def interrupt_during_monitoring(seconds):
        # The runner polls in a sleep loop; a Ctrl-C lands here, not in wait().
        raise KeyboardInterrupt

    def capture_pid(pid):
        seen.append(pid)
        return psutil.Process(pid)

    monkeypatch.setattr(process_mod.time, "sleep", interrupt_during_monitoring)
    monkeypatch.setattr(process_mod.psutil, "Process", capture_pid)

    with pytest.raises(KeyboardInterrupt):
        run_once(f"{sys.executable} {script}", {}, None, None, 0.01)

    assert seen, "the child never started"
    deadline = _time.time() + 10
    while _time.time() < deadline:
        if not psutil.pid_exists(seen[0]) or psutil.Process(seen[0]).status() == psutil.STATUS_ZOMBIE:
            break
        real_sleep(0.1)
    else:
        raise AssertionError("child survived the interrupt")
