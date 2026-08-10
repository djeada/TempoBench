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


def test_run_output_clearly_summarizes_results(tmp_path: Path):
    cfg = tmp_path / "bench.yaml"
    cfg.write_text(textwrap.dedent("""\
        benchmarks:
          - name: echo_test
            cmd: "echo {n}"
        grid:
          n: [1]
        limits:
          warmups: 0
          repeats: 1
    """))
    result = runner.invoke(
        app, ["run", "--config", str(cfg), "--out-dir", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output
    assert "Run complete" in result.output
    assert "Successful" in result.output
    assert "Raw benchmark data ready" in result.output
    assert "Next step" in result.output


def _broken_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "bench.yaml"
    cfg.write_text(textwrap.dedent("""\
        benchmarks:
          - name: broken
            cmd: "definitely-not-a-real-command-xyz {n}"
        grid:
          n: [1, 2]
        limits:
          warmups: 0
          repeats: 1
    """))
    return cfg


def test_run_exits_non_zero_when_every_trial_fails(tmp_path: Path):
    result = runner.invoke(
        app,
        ["run", "--config", str(_broken_config(tmp_path)), "--out-dir", str(tmp_path / "out")],
    )
    assert result.exit_code == 1, result.output
    assert "No trial succeeded" in result.output


def test_run_shows_why_trials_did_not_succeed(tmp_path: Path):
    result = runner.invoke(
        app,
        ["run", "--config", str(_broken_config(tmp_path)), "--out-dir", str(tmp_path / "out")],
    )
    assert "Why trials did not succeed" in result.output
    assert "definitely-not-a-real-command-xyz" in result.output


def test_run_reports_failures_even_when_quiet(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "run",
            "--config", str(_broken_config(tmp_path)),
            "--out-dir", str(tmp_path / "out"),
            "--quiet",
        ],
    )
    assert result.exit_code == 1, result.output


def test_run_partial_failure_exits_non_zero(tmp_path: Path):
    cfg = tmp_path / "bench.yaml"
    cfg.write_text(textwrap.dedent("""\
        benchmarks:
          - name: half_broken
            cmd: "sh -c 'test {n} -lt 2'"
        grid:
          n: [1, 2]
        limits:
          warmups: 0
          repeats: 1
    """))
    out = tmp_path / "out"
    result = runner.invoke(app, ["run", "--config", str(cfg), "--out-dir", str(out)])
    assert result.exit_code == 1, result.output
    assert "failed or errored" in result.output

    allowed = runner.invoke(
        app, ["run", "--config", str(cfg), "--out-dir", str(out), "--allow-failures"]
    )
    assert allowed.exit_code == 0, allowed.output
    assert "Next step" in allowed.output


def test_summarize_exits_non_zero_on_runs_without_successes(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    runs.write_text('{"ts": "t", "status": "error", "bench": "b", "params": {"n": 1}}\n')
    out_csv = tmp_path / "summary.csv"
    result = runner.invoke(
        app, ["summarize", "--runs", str(runs), "--out-csv", str(out_csv)]
    )
    assert result.exit_code == 1, result.output
    assert "no successful trials" in result.output
    assert not out_csv.exists(), "an empty summary must not be presented as an artifact"


@pytest.mark.parametrize(
    "command, extra",
    [
        ("plot", ["--out-html"]),
        ("heatmap", ["--output"]),
        ("memory", ["--output"]),
        ("dashboard", ["--output"]),
        ("report", ["--output"]),
    ],
)
def test_chart_commands_refuse_an_empty_summary(tmp_path: Path, command, extra):
    empty = tmp_path / "summary.csv"
    empty.write_text("bench,n,impl,time_ms_median\n")
    result = runner.invoke(
        app,
        [command, "--summary", str(empty), *extra, str(tmp_path / f"{command}.html")],
    )
    assert result.exit_code == 1, result.output
    assert "no rows" in result.output


def test_summarize_reports_which_metric_it_used(bench_dir: Path):
    result = runner.invoke(
        app,
        [
            "summarize",
            "--runs", str(bench_dir / "runs.jsonl"),
            "--out-csv", str(bench_dir / "summary.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "wall clock" in result.output


def test_summarize(bench_dir: Path):
    runs = bench_dir / "runs.jsonl"
    csv = bench_dir / "summary.csv"
    result = runner.invoke(app, ["summarize", "--runs", str(runs), "--out-csv", str(csv)])
    assert result.exit_code == 0
    assert csv.exists()
    assert "2 configuration(s)" in result.output
    assert "Summary ready" in result.output


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


def test_inspect_shows_why_a_trial_failed(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        '{"ts": "t", "status": "error", "bench": "b", "cmd": "nope", '
        '"params": {"n": 1}, "stderr": "spawn-failed-xyz"}\n'
    )
    result = runner.invoke(app, ["inspect", "--runs", str(runs)])
    assert result.exit_code == 0, result.output
    assert "spawn-failed-xyz" in result.output
    assert "Errors" in result.output


def test_inspect_shows_self_reported_durations(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        '{"ts": "t", "status": "ok", "bench": "b", "cmd": "prog", '
        '"params": {"n": 1}, "wall_ms": 90.0, "reported_ms": 1.25}\n'
    )
    result = runner.invoke(app, ["inspect", "--runs", str(runs)])
    assert result.exit_code == 0, result.output
    assert "1.250" in result.output
    assert "90.00" in result.output


def test_plot_prints_the_fitted_classes_with_confidence(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "bench,impl,n,time_ms_median\n"
        + "".join(
            f"b,linear,{n},{n * 0.001}\n" for n in [1000, 4000, 16000, 64000, 256000]
        )
    )
    result = runner.invoke(
        app,
        ["plot", "--summary", str(summary), "--out-html", str(tmp_path / "rt.html")],
    )
    assert result.exit_code == 0, result.output
    assert "Complexity fits" in result.output
    assert "Confidence" in result.output
    assert "high" in result.output


def test_plot_without_out_html_emits_only_json_on_stdout(tmp_path: Path):
    """The chart JSON is meant to be piped; nothing else may share stdout."""
    import json as _json

    summary = tmp_path / "summary.csv"
    summary.write_text(
        "bench,impl,n,time_ms_median\n"
        + "".join(f"b,linear,{n},{n * 0.001}\n" for n in [1000, 4000, 16000, 64000])
    )
    result = runner.invoke(
        app,
        [
            "plot",
            "--summary", str(summary),
            "--out-html", "-",
            "--export-fits", str(tmp_path / "fits.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert _json.loads(result.stdout), "stdout must parse as the Vega-Lite spec"
    assert (tmp_path / "fits.csv").exists()


def test_summarize_distinguishes_no_successes_from_no_timings(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    runs.write_text('{"ts": "t", "status": "ok", "bench": "b", "params": {"n": 1}}\n')
    result = runner.invoke(
        app,
        ["summarize", "--runs", str(runs), "--out-csv", str(tmp_path / "s.csv")],
    )
    assert result.exit_code == 1, result.output
    assert "usable duration" in result.output


def test_report_warns_when_it_cannot_describe_the_benchmark_machine(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    summary.write_text("bench,impl,n,time_ms_median\nb,a,100,1.0\n")
    result = runner.invoke(
        app,
        ["report", "--summary", str(summary), "--output", str(tmp_path / "r.html")],
    )
    assert result.exit_code == 0, result.output
    assert "No provenance snapshot found" in result.output


def test_report_auto_detects_provenance_next_to_the_summary(bench_dir: Path):
    summarize = runner.invoke(
        app,
        [
            "summarize",
            "--runs", str(bench_dir / "runs.jsonl"),
            "--out-csv", str(bench_dir / "summary.csv"),
        ],
    )
    assert summarize.exit_code == 0, summarize.output

    assert (bench_dir / "provenance.json").exists()
    result = runner.invoke(
        app,
        [
            "report",
            "--summary", str(bench_dir / "summary.csv"),
            "--output", str(bench_dir / "report.html"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Auto-detected provenance" in result.output
    assert "Recorded when the benchmark ran" in (bench_dir / "report.html").read_text()


def test_report_survives_a_summary_too_short_to_fit(tmp_path: Path):
    """One row per series cannot be fitted; that must not crash the report."""
    summary = tmp_path / "summary.csv"
    summary.write_text("bench,impl,n,time_ms_median\nb,a,100,1.0\n")
    output = tmp_path / "r.html"
    result = runner.invoke(
        app, ["report", "--summary", str(summary), "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "Not enough input sizes" in result.output
    assert output.exists()
    assert "Complexity Analysis" not in output.read_text()


def test_chart_commands_infer_axes_for_an_unconventional_grid(tmp_path: Path):
    """The whole pipeline must work for a sweep not named `n`/`impl`."""
    summary = tmp_path / "summary.csv"
    rows = ["bench,algorithm,vertices,time_ms_median,peak_rss_mb_median"]
    for algo, coeff in [("dijkstra", 0.01), ("bellman", 0.05)]:
        for v in [100, 1000, 10000, 100000]:
            rows.append(f"g,{algo},{v},{v * coeff},12.0")
    summary.write_text("\n".join(rows) + "\n")

    result = runner.invoke(
        app, ["plot", "--summary", str(summary), "--out-html", str(tmp_path / "rt.html")]
    )
    assert result.exit_code == 0, result.output
    assert "x = vertices" in result.output
    assert "series = algorithm" in result.output
    assert "Algorithm" in result.output, "the fits table must be keyed by the real axis"

    for command in ["heatmap", "memory", "dashboard"]:
        res = runner.invoke(
            app,
            [command, "--summary", str(summary), "--output", str(tmp_path / f"{command}.html")],
        )
        assert res.exit_code == 0, f"{command}: {res.output}"
        assert "x = vertices" in res.output


def test_explicit_axis_flags_still_win(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    rows = ["bench,algorithm,vertices,time_ms_median"]
    for algo in ["a", "b"]:
        for v in [100, 1000, 10000]:
            rows.append(f"g,{algo},{v},{v * 0.01}")
    summary.write_text("\n".join(rows) + "\n")

    result = runner.invoke(
        app,
        [
            "plot",
            "--summary", str(summary),
            "--x", "vertices",
            "--color", "algorithm",
            "--out-html", str(tmp_path / "rt.html"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "inferred; override" not in result.output


def test_plot_explains_itself_when_no_input_size_axis_exists(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    summary.write_text("bench,impl,time_ms_median\ng,a,1.0\n")
    result = runner.invoke(
        app, ["plot", "--summary", str(summary), "--out-html", str(tmp_path / "r.html")]
    )
    assert result.exit_code == 1, result.output
    assert "which column is the input size" in result.output
