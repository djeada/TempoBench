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
    with pytest.raises(ValueError, match="missing grid key\\(s\\): missing"):
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
