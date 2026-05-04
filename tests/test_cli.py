from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pytest
import yaml
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


def test_plot_exports_fits_with_strict_strategy(bench_dir: Path):
    runs = bench_dir / "runs.jsonl"
    csv_path = bench_dir / "summary.csv"
    fits_path = bench_dir / "fits.csv"
    summarize = runner.invoke(
        app, ["summarize", "--runs", str(runs), "--out-csv", str(csv_path)]
    )
    assert summarize.exit_code == 0, summarize.output

    result = runner.invoke(
        app,
        [
            "plot",
            "--summary",
            str(csv_path),
            "--export-fits",
            str(fits_path),
            "--complexity-strategy",
            "strict",
        ],
    )
    assert result.exit_code == 0, result.output
    with fits_path.open(newline="") as f:
        header = next(csv.reader(f))
    assert "display_model" in header
    assert "empirical_exponent" in header


def test_report_autogenerates_fits_with_strict_strategy(bench_dir: Path):
    runs = bench_dir / "runs.jsonl"
    csv_path = bench_dir / "summary.csv"
    report_path = bench_dir / "report.html"
    summarize = runner.invoke(
        app, ["summarize", "--runs", str(runs), "--out-csv", str(csv_path)]
    )
    assert summarize.exit_code == 0, summarize.output

    result = runner.invoke(
        app,
        [
            "report",
            "--summary",
            str(csv_path),
            "--runs",
            str(runs),
            "--output",
            str(report_path),
            "--complexity-strategy",
            "strict",
        ],
    )
    assert result.exit_code == 0, result.output
    assert report_path.exists()
    assert "Complexity Analysis" in report_path.read_text()


def test_sort_bench_example_smoke(tmp_path: Path):
    example_cfg = Path("examples/sort_bench.yaml")
    data = yaml.safe_load(example_cfg.read_text())
    data["grid"]["n"] = [100, 1000]
    data["grid"]["impl"] = ["random", "sorted"]
    data["limits"]["warmups"] = 0
    data["limits"]["repeats"] = 1
    data["limits"]["timeout_sec"] = 5
    data.pop("pin_cpu", None)

    cfg = tmp_path / "sort_bench.yaml"
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))
    out = tmp_path / "out"

    run_res = runner.invoke(
        app, ["run", "--config", str(cfg), "--out-dir", str(out), "--quiet"]
    )
    assert run_res.exit_code == 0, run_res.output

    summary_res = runner.invoke(
        app,
        [
            "summarize",
            "--runs",
            str(out / "runs.jsonl"),
            "--out-csv",
            str(out / "summary.csv"),
        ],
    )
    assert summary_res.exit_code == 0, summary_res.output

    plot_res = runner.invoke(
        app,
        [
            "plot",
            "--summary",
            str(out / "summary.csv"),
            "--out-html",
            str(out / "runtime.html"),
            "--no-fit",
        ],
    )
    assert plot_res.exit_code == 0, plot_res.output
    assert (out / "runtime.html").exists()
