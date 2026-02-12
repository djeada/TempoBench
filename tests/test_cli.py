from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tembench.cli import app

runner = CliRunner()


@pytest.fixture
def bench_dir(tmp_path: Path) -> Path:
    """Create a minimal benchmark config and run it."""
    cfg = tmp_path / "bench.yaml"
    cfg.write_text(textwrap.dedent("""\
        benchmarks:
          - name: echo_test
            cmd: "echo {n}"
        grid:
          n: [1, 2]
        limits:
          timeout_sec: 5
          warmups: 0
          repeats: 1
    """))
    out = tmp_path / "out"
    result = runner.invoke(app, ["run", "--config", str(cfg), "--out-dir", str(out), "--quiet"])
    assert result.exit_code == 0, result.output
    return out


def test_run_creates_jsonl(bench_dir: Path):
    assert (bench_dir / "runs.jsonl").exists()


def test_summarize(bench_dir: Path):
    runs = bench_dir / "runs.jsonl"
    csv = bench_dir / "summary.csv"
    result = runner.invoke(app, ["summarize", "--runs", str(runs), "--out-csv", str(csv)])
    assert result.exit_code == 0
    assert csv.exists()


def test_sysinfo():
    result = runner.invoke(app, ["sysinfo"])
    assert result.exit_code == 0
    assert "Platform" in result.output


def test_inspect(bench_dir: Path):
    runs = bench_dir / "runs.jsonl"
    result = runner.invoke(app, ["inspect", "--runs", str(runs), "--count", "5"])
    assert result.exit_code == 0
    assert "ok" in result.output
