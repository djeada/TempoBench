from __future__ import annotations

import json
from pathlib import Path

from tembench.summarize import read_jsonl, summarize_runs


def _write_runs(path: Path, records: list[dict]):
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_read_jsonl(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    _write_runs(p, [{"a": 1}, {"a": 2}])
    rows = read_jsonl(p)
    assert len(rows) == 2


def test_read_jsonl_skips_bad_lines(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a":1}\nnot json\n{"a":2}\n')
    rows = read_jsonl(p)
    assert len(rows) == 2


def test_summarize_runs_basic(tmp_path: Path):
    p = tmp_path / "runs.jsonl"
    records = [
        {"bench": "t", "status": "ok", "wall_ms": 10.0, "peak_rss_mb": 1.0, "params": {"impl": "a", "n": 100}},
        {"bench": "t", "status": "ok", "wall_ms": 12.0, "peak_rss_mb": 1.5, "params": {"impl": "a", "n": 100}},
        {"bench": "t", "status": "ok", "wall_ms": 11.0, "peak_rss_mb": 1.2, "params": {"impl": "a", "n": 100}},
    ]
    _write_runs(p, records)
    df = summarize_runs(p)
    assert len(df) == 1
    assert "wall_ms_median" in df.columns
    assert "wall_ms_p10" in df.columns
    assert "wall_ms_p90" in df.columns
    assert df.iloc[0]["wall_ms_median"] == 11.0


def test_summarize_runs_empty(tmp_path: Path):
    p = tmp_path / "runs.jsonl"
    p.write_text("")
    df = summarize_runs(p)
    assert df.empty


def test_summarize_keeps_all_n_points_per_impl(tmp_path: Path):
    p = tmp_path / "runs.jsonl"
    records = []
    for impl, walls in {
        "hash_set": [36.0, 37.0, 38.0, 42.0],
        "sort_scan": [38.0, 39.0, 40.0, 44.0],
        "quadratic": [48.0, 56.0, 91.0, 210.0],
    }.items():
        for n, wall in zip([1000, 2000, 4000, 8000], walls):
            records.append({"bench": "u", "status": "ok", "wall_ms": wall, "peak_rss_mb": 1.0, "params": {"impl": impl, "n": n}})
            records.append({"bench": "u", "status": "ok", "wall_ms": wall + 0.2, "peak_rss_mb": 1.1, "params": {"impl": impl, "n": n}})
    _write_runs(p, records)

    df = summarize_runs(p)
    # 3 implementations × 4 n values
    assert len(df) == 12
    assert set(df["n"].tolist()) == {1000, 2000, 4000, 8000}
