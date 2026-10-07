from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tembench.config import Benchmark, Limits, load_config


def test_load_config_basic(tmp_path: Path):
    cfg_file = tmp_path / "bench.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        benchmarks:
          - name: echo_test
            cmd: "echo {n}"
        grid:
          n: [1, 2, 3]
        limits:
          timeout_sec: 10
          warmups: 0
          repeats: 2
          rss_poll_interval_sec: 0.05
    """))
    cfg = load_config(cfg_file)
    assert len(cfg.benchmarks) == 1
    assert cfg.benchmarks[0].name == "echo_test"
    assert cfg.grid["n"] == [1, 2, 3]
    assert cfg.limits.timeout_sec == 10
    assert cfg.limits.warmups == 0
    assert cfg.limits.repeats == 2
    assert cfg.limits.rss_poll_interval_sec == 0.05
    assert cfg.pin_cpu is None


def test_load_config_with_pin_cpu(tmp_path: Path):
    cfg_file = tmp_path / "bench.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        benchmarks:
          - name: test
            cmd: "true"
        grid: {}
        pin_cpu: 0
    """))
    cfg = load_config(cfg_file)
    assert cfg.pin_cpu == 0


def test_load_config_rejects_unknown_cmd_placeholder(tmp_path: Path):
    cfg_file = tmp_path / "bench.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        benchmarks:
          - name: invalid
            cmd: "echo {missing}"
        grid:
          n: [1]
    """))
    with pytest.raises(ValueError, match="unknown placeholder\\(s\\): missing"):
        load_config(cfg_file)


def test_load_config_accepts_builtin_python_placeholder(tmp_path: Path):
    cfg_file = tmp_path / "bench.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        benchmarks:
          - name: portable
            cmd: "{python} script.py --n {n}"
        grid:
          n: [1]
    """))
    cfg = load_config(cfg_file)
    assert cfg.benchmarks[0].cmd == "{python} script.py --n {n}"


def test_load_config_rejects_empty_benchmarks(tmp_path: Path):
    cfg_file = tmp_path / "bench.yaml"
    cfg_file.write_text("grid:\n  n: [1]\n")
    with pytest.raises(ValueError, match="no benchmarks defined"):
        load_config(cfg_file)


def test_load_config_rejects_empty_grid_axis(tmp_path: Path):
    cfg_file = tmp_path / "bench.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        benchmarks:
          - name: t
            cmd: "echo {n}"
        grid:
          n: []
    """))
    with pytest.raises(ValueError, match="empty sweep: n"):
        load_config(cfg_file)


@pytest.mark.parametrize(
    "limits, message",
    [
        ("metric: nonsense", "limits.metric must be one of"),
        ("repeats: 0", "limits.repeats must be at least 1"),
        ("warmups: -1", "limits.warmups must not be negative"),
        ("workers: 0", "limits.workers must be at least 1"),
        ("timeout_sec: 0", "limits.timeout_sec must be positive"),
    ],
)
def test_load_config_rejects_unusable_limits(tmp_path: Path, limits: str, message: str):
    cfg_file = tmp_path / "bench.yaml"
    cfg_file.write_text(textwrap.dedent(f"""\
        benchmarks:
          - name: t
            cmd: "echo {{n}}"
        grid:
          n: [1]
        limits:
          {limits}
    """))
    with pytest.raises(ValueError, match=message):
        load_config(cfg_file)


def test_benchmark_defaults():
    b = Benchmark(name="t", cmd="echo hi")
    assert b.build is None
    assert b.workdir is None
    assert b.env == {}


def test_limits_defaults():
    lim = Limits()
    assert lim.timeout_sec is None
    assert lim.warmups == 1
    assert lim.repeats == 3
    assert lim.rss_poll_interval_sec == 0.01
    assert lim.shuffle is True
    assert lim.growth_key == "n"
    assert lim.metric == "auto"


@pytest.mark.parametrize(
    "body, message",
    [
        ("grid:\n  impl: hash\n", "must be a list"),
        ("grid:\n  n: 100\n", "must be a list"),
        ("grid:\n  timeout: [1, 2]\n", "reserved"),
        ("grid:\n  n: [1]\nlimits:\n  prune_on_timeout: 'false'\n", "true or false"),
        ("grid:\n  n: [1]\nlimits:\n  repeats: 2.5\n", "whole number"),
        ("grid:\n  n: [1]\nlimits:\n  repeats: true\n", "whole number"),
        ("grid:\n  n: [1]\nlimits:\n  timeout_sec: '30'\n", "a number"),
        ("grid:\n  n: [1]\nlimits:\n  repeat: 5\n", "unknown limits key"),
        ("grid:\n  n: [1]\npin_cpu: first\n", "pin_cpu"),
        # Trial record fields: they would collide with the axis in a summary.
        ("grid:\n  status: [1, 2]\n", "reserved"),
        ("grid:\n  metric: [a]\n", "reserved"),
        ("grid:\n  n: [[1, 2], [3]]\n", "numbers or strings"),
        ("grid:\n  n: [1, 'big']\nlimits:\n  growth_key: 3\n", "growth_key"),
    ],
)
def test_nonsense_is_rejected_with_a_clear_message(tmp_path: Path, body: str, message: str):
    p = tmp_path / "c.yaml"
    p.write_text("benchmarks:\n  - name: b\n    cmd: 'echo {n}'\n" + body)
    with pytest.raises(ValueError, match=message):
        load_config(p)


def test_empty_grid_and_limits_sections_mean_defaults(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("benchmarks:\n  - name: b\n    cmd: 'true'\ngrid:\nlimits:\n")
    cfg = load_config(p)
    assert cfg.grid == {}
    assert cfg.limits.repeats == 3


def test_env_values_become_strings(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("benchmarks:\n  - name: b\n    cmd: 'true'\n    env: {THREADS: 4}\n")
    assert load_config(p).benchmarks[0].env == {"THREADS": "4"}


def _config(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


@pytest.mark.parametrize(
    "cmd, grid",
    [
        ("echo {n:05d}", "n: [1, 'big']"),  # format spec unsuited to one value
        ("echo {}", "n: [1]"),  # positional field
        ("echo {n.real.nope}", "n: [1]"),
    ],
)
def test_command_must_expand_at_every_grid_point(tmp_path: Path, cmd: str, grid: str):
    # Checked up front: otherwise the sweep crashes at that point, after the
    # points before it have already run.
    p = _config(tmp_path, f"benchmarks:\n  - name: b\n    cmd: '{cmd}'\ngrid:\n  {grid}\n")
    with pytest.raises(ValueError, match="cannot be expanded"):
        load_config(p)


def test_builtin_placeholders_accept_raw(tmp_path: Path):
    p = _config(tmp_path, "benchmarks:\n  - name: b\n    cmd: '{python:raw} -V'\n")
    assert load_config(p).benchmarks[0].cmd == "{python:raw} -V"


def test_duplicate_benchmark_names_are_rejected(tmp_path: Path):
    p = _config(
        tmp_path,
        "benchmarks:\n  - name: a\n    cmd: 'true'\n  - name: a\n    cmd: 'false'\n",
    )
    with pytest.raises(ValueError, match="unique: a"):
        load_config(p)


def test_env_booleans_are_lowercase_and_nulls_rejected(tmp_path: Path):
    p = _config(tmp_path, "benchmarks:\n  - name: b\n    cmd: 'true'\n    env: {A: true, B: 1.5}\n")
    assert load_config(p).benchmarks[0].env == {"A": "true", "B": "1.5"}
    p = _config(tmp_path, "benchmarks:\n  - name: b\n    cmd: 'true'\n    env: {A: }\n")
    with pytest.raises(ValueError, match="env.A has no value"):
        load_config(p)


def test_build_must_be_a_string(tmp_path: Path):
    p = _config(tmp_path, "benchmarks:\n  - name: b\n    cmd: 'true'\n    build: true\n")
    with pytest.raises(ValueError, match="build must be a string"):
        load_config(p)
