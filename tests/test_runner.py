from __future__ import annotations

import json
from pathlib import Path

from tembench.config import Benchmark, Config, Limits
from tembench.runner import _run_grid_point, expand_grid, format_cmd, run_benchmarks, run_once


def test_expand_grid_single():
    grid = {"n": [1, 2]}
    result = expand_grid(grid)
    assert result == [{"n": 1}, {"n": 2}]


def test_expand_grid_multi():
    grid = {"n": [1, 2], "impl": ["a", "b"]}
    result = expand_grid(grid)
    assert len(result) == 4
    assert {"n": 1, "impl": "a"} in result
    assert {"n": 2, "impl": "b"} in result


def test_expand_grid_empty():
    assert expand_grid({}) == [{}]


def test_format_cmd():
    assert format_cmd("echo {n} {impl}", {"n": 10, "impl": "fast"}) == "echo 10 fast"


def test_run_once_success():
    result = run_once("echo hello", {}, None, timeout=5.0)
    assert result["status"] == "ok"
    assert result["rc"] == 0
    assert result["wall_ms"] > 0
    assert "hello" in result["stdout"]


def test_run_once_failure():
    result = run_once("false", {}, None, timeout=5.0)
    assert result["status"] == "failed"


def test_run_once_timeout():
    result = run_once("sleep 10", {}, None, timeout=0.2)
    assert result["status"] == "timeout"


def test_run_once_missing_command():
    result = run_once("nonexistent_cmd_xyz", {}, None, timeout=2.0)
    assert result["status"] == "error"


def test_run_benchmarks_writes_jsonl(tmp_path: Path):
    cfg = Config(
        benchmarks=[Benchmark(name="test", cmd="echo {n}")],
        grid={"n": [1, 2]},
        limits=Limits(timeout_sec=5, warmups=0, repeats=1, shuffle=False),
    )
    out = tmp_path / "runs.jsonl"
    run_benchmarks(cfg, out, seed=42)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert rec["status"] == "ok"
        assert rec["bench"] == "test"


def test_run_benchmarks_callback(tmp_path: Path):
    cfg = Config(
        benchmarks=[Benchmark(name="cb", cmd="echo {n}")],
        grid={"n": [1]},
        limits=Limits(timeout_sec=5, warmups=0, repeats=1, shuffle=False),
    )
    out = tmp_path / "runs.jsonl"
    calls = []
    run_benchmarks(cfg, out, seed=0, on_trial=lambda *a: calls.append(a))
    assert len(calls) == 1
    assert calls[0][0] == "cb"  # bench_name


def test_run_benchmarks_overwrites_by_default(tmp_path: Path):
    cfg = Config(
        benchmarks=[Benchmark(name="overwrite_test", cmd="echo {n}")],
        grid={"n": [1]},
        limits=Limits(timeout_sec=5, warmups=0, repeats=1, shuffle=False),
    )
    out = tmp_path / "runs.jsonl"
    run_benchmarks(cfg, out, seed=0)
    run_benchmarks(cfg, out, seed=0)
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 1


def test_run_benchmarks_append_mode(tmp_path: Path):
    cfg = Config(
        benchmarks=[Benchmark(name="append_test", cmd="echo {n}")],
        grid={"n": [1]},
        limits=Limits(timeout_sec=5, warmups=0, repeats=1, shuffle=False),
    )
    out = tmp_path / "runs.jsonl"
    run_benchmarks(cfg, out, seed=0, append=True)
    run_benchmarks(cfg, out, seed=0, append=True)
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 2


# ---------- parallel execution tests ----------


def test_run_grid_point_returns_results():
    bench = Benchmark(name="gp", cmd="echo hello")
    results = _run_grid_point(bench, {"n": 1}, timeout=5.0, warmups=0, repeats=2, retries=0)
    assert len(results) == 2
    assert all(r["status"] == "ok" for r in results)
    assert all(r["bench"] == "gp" for r in results)


def test_run_benchmarks_parallel(tmp_path: Path):
    cfg = Config(
        benchmarks=[Benchmark(name="par", cmd="echo {n}")],
        grid={"n": [1, 2, 3]},
        limits=Limits(timeout_sec=5, warmups=0, repeats=1, shuffle=False, workers=2),
    )
    out = tmp_path / "runs.jsonl"
    run_benchmarks(cfg, out, seed=42)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 3
    recs = [json.loads(line) for line in lines]
    assert all(r["status"] == "ok" for r in recs)
    assert all(r["bench"] == "par" for r in recs)
    # All three n values must be present (order may vary in parallel)
    n_vals = sorted(r["params"]["n"] for r in recs)
    assert n_vals == [1, 2, 3]


def test_run_benchmarks_parallel_callback(tmp_path: Path):
    cfg = Config(
        benchmarks=[Benchmark(name="pcb", cmd="echo {n}")],
        grid={"n": [1, 2]},
        limits=Limits(timeout_sec=5, warmups=0, repeats=1, shuffle=False, workers=2),
    )
    out = tmp_path / "runs.jsonl"
    calls = []
    run_benchmarks(cfg, out, seed=0, on_trial=lambda *a: calls.append(a))
    assert len(calls) == 2


def test_run_benchmarks_parallel_repeats(tmp_path: Path):
    cfg = Config(
        benchmarks=[Benchmark(name="prep", cmd="echo {n}")],
        grid={"n": [1]},
        limits=Limits(timeout_sec=5, warmups=0, repeats=3, shuffle=False, workers=2),
    )
    out = tmp_path / "runs.jsonl"
    run_benchmarks(cfg, out, seed=0)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 3


def test_run_benchmarks_parallel_handles_failure(tmp_path: Path):
    cfg = Config(
        benchmarks=[Benchmark(name="pfail", cmd="false")],
        grid={"n": [1]},
        limits=Limits(timeout_sec=5, warmups=0, repeats=1, shuffle=False, workers=2),
    )
    out = tmp_path / "runs.jsonl"
    run_benchmarks(cfg, out, seed=0)
    lines = out.read_text().strip().split("\n")
    assert len(lines) >= 1
    rec = json.loads(lines[0])
    assert rec["status"] == "failed"


def test_workers_config_from_yaml(tmp_path: Path):
    """workers field in YAML limits section is loaded correctly."""
    yaml_text = """
benchmarks:
  - name: wtest
    cmd: "echo {n}"
grid:
  n: [1]
limits:
  workers: 4
  warmups: 0
  repeats: 1
"""
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text(yaml_text)
    from tembench.config import load_config
    cfg = load_config(cfg_path)
    assert cfg.limits.workers == 4
