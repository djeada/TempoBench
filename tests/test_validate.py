from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from typer.testing import CliRunner

from tembench.cli import app

runner = CliRunner()


def _config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "bench.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_validate_reports_the_planned_sweep(tmp_path: Path):
    cfg = _config(tmp_path, """
        benchmarks:
          - name: t
            cmd: "echo {n}"
        grid:
          n: [1, 2, 4, 8]
          mode: ["a", "b"]
        limits:
          warmups: 1
          repeats: 3
    """)
    result = runner.invoke(app, ["validate", "--config", str(cfg), "--no-probe"])
    assert result.exit_code == 0, result.output
    # 1 benchmark x 8 grid points x (3 repeats + 1 warm-up) = 32 launches.
    assert "32" in result.output
    assert "growth key" in result.output


def test_validate_probes_the_cheapest_grid_point(tmp_path: Path):
    cfg = _config(tmp_path, """
        benchmarks:
          - name: t
            cmd: "echo {n}"
        grid:
          n: [64, 2, 16]
    """)
    result = runner.invoke(app, ["validate", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "{'n': 2}" in result.output, "a probe must not cost more than it has to"
    assert "Config is runnable" in result.output


def test_validate_fails_on_a_command_that_cannot_run(tmp_path: Path):
    cfg = _config(tmp_path, """
        benchmarks:
          - name: broken
            cmd: "definitely-not-a-real-command-xyz {n}"
        grid:
          n: [1, 2]
    """)
    result = runner.invoke(app, ["validate", "--config", str(cfg)])
    assert result.exit_code == 1, result.output
    assert "did not run" in result.output
    assert "definitely-not-a-real-command-xyz" in result.output


def test_validate_warns_that_wall_clock_will_be_used(tmp_path: Path):
    """Silently falling back to wall time is the trap this command exists for."""
    cfg = _config(tmp_path, """
        benchmarks:
          - name: t
            cmd: "echo {n}"
        grid:
          n: [1, 2, 4, 8]
    """)
    result = runner.invoke(app, ["validate", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "TEMPOBENCH_MS" in result.output
    assert "wall-clock time will be used" in result.output


def test_validate_reports_a_self_reported_measurement(tmp_path: Path):
    script = tmp_path / "prog.py"
    script.write_text('print("TEMPOBENCH_MS: 2.5")\n')
    cfg = _config(tmp_path, f"""
        benchmarks:
          - name: t
            cmd: "{sys.executable} {script} {{n}}"
        grid:
          n: [1, 2, 4, 8]
        limits:
          metric: reported
    """)
    result = runner.invoke(app, ["validate", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "self-reported" in result.output
    assert "wall-clock time will be used" not in result.output


def test_validate_warns_when_there_are_too_few_input_sizes(tmp_path: Path):
    cfg = _config(tmp_path, """
        benchmarks:
          - name: t
            cmd: "echo {n}"
        grid:
          n: [1, 2]
    """)
    result = runner.invoke(app, ["validate", "--config", str(cfg), "--no-probe"])
    assert "Only 2 input size(s)" in result.output


def test_validate_warns_when_the_growth_key_is_not_a_grid_axis(tmp_path: Path):
    cfg = _config(tmp_path, """
        benchmarks:
          - name: t
            cmd: "echo {size}"
        grid:
          size: [1, 2, 4, 8]
    """)
    result = runner.invoke(app, ["validate", "--config", str(cfg), "--no-probe"])
    assert "nothing to prune" in result.output


def test_validate_surfaces_a_bad_config_without_running_anything(tmp_path: Path):
    cfg = _config(tmp_path, """
        benchmarks:
          - name: t
            cmd: "echo {typo}"
        grid:
          n: [1, 2]
    """)
    result = runner.invoke(app, ["validate", "--config", str(cfg)])
    assert result.exit_code != 0
    assert isinstance(result.exception, (ValueError, SystemExit))


def test_validate_warns_about_a_protocol_too_thin_to_trust(tmp_path: Path):
    """Too few repeats and no warm-up is the usual cause of a wrong-but-confident class."""
    cfg = _config(tmp_path, """
        benchmarks:
          - name: t
            cmd: "echo {n}"
        grid:
          n: [1, 2, 4, 8]
        limits:
          warmups: 0
          repeats: 2
    """)
    result = runner.invoke(app, ["validate", "--config", str(cfg), "--no-probe"])
    assert "repeats=2" in result.output
    assert "warmups=0" in result.output


def test_validate_is_quiet_about_a_sound_protocol(tmp_path: Path):
    cfg = _config(tmp_path, """
        benchmarks:
          - name: t
            cmd: "echo {n}"
        grid:
          n: [1, 2, 4, 8]
        limits:
          warmups: 1
          repeats: 5
    """)
    result = runner.invoke(app, ["validate", "--config", str(cfg), "--no-probe"])
    assert "repeats=" not in result.output
    assert "warmups=" not in result.output


def test_bundled_examples_pass_their_own_validation():
    """The configs the README points at must not model bad practice."""
    for name in ["unique_bench.yaml", "sort_bench.yaml"]:
        path = Path("examples") / name
        result = runner.invoke(app, ["validate", "--config", str(path), "--no-probe"])
        assert result.exit_code == 0, result.output
        assert "!" not in result.output, f"{name} raises a warning:\n{result.output}"
