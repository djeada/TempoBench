"""Regression tests for the HTML report: escaping, counts, and number formatting."""

from __future__ import annotations

import json
import re
from pathlib import Path

import altair as alt
import pandas as pd

from tembench.plotting import plot_runtime, save_chart
from tembench.reporting import generate_report
from tembench.reporting.formatting import _fmt_val, format_number

EVIL = "</script><script>alert(1)</script>"


def _summary(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "summary.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_report_escapes_every_interpolated_value(tmp_path: Path):
    summary = _summary(
        tmp_path,
        [{"bench": EVIL, "impl": "<img src=x onerror=alert(2)>", "n": n, "time_ms_median": n / 10} for n in (10, 100)],
    )
    chart = tmp_path / "runtime.html"
    save_chart(plot_runtime(summary), chart)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "ts": "t", "cwd": "/tmp/<b>", "cmdline": f"tembench run {EVIL}",
        "system": {"platform": "<i>linux</i>", "hostname": "h"},
    }))
    page = generate_report(summary, chart_html=chart, title=f"T {EVIL}", provenance_json=provenance)

    assert "<script>alert" not in page
    assert "<img src=x" not in page
    assert "<i>linux</i>" not in page and "/tmp/<b>" not in page
    assert "&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_report_loads_the_vega_versions_altair_targets(tmp_path: Path):
    summary = _summary(tmp_path, [{"impl": "a", "n": n, "time_ms_median": n} for n in (1, 10)])
    chart = tmp_path / "runtime.html"
    save_chart(plot_runtime(summary, show_fit=False), chart)
    page = generate_report(summary, chart_html=chart)
    assert f"vega-lite@{alt.VEGALITE_VERSION}" in page


def _cards(page: str) -> dict[str, str]:
    return {
        label: value
        for value, label in re.findall(
            r'stat-value">([^<]*)</div><div class="stat-label">([^<]*)', page
        )
    }


def test_run_statistics_count_every_status(tmp_path: Path):
    summary = _summary(tmp_path, [{"impl": "a", "n": 1, "time_ms_median": 1.0}])
    runs = tmp_path / "runs.jsonl"
    statuses = ["ok", "error", "error", "skipped", "retried"]
    runs.write_text("".join(json.dumps({"status": s}) + "\n" for s in statuses))
    cards = _cards(generate_report(summary, runs_jsonl=runs))
    assert cards["Successful"] == "1"
    assert cards["Errors"] == "2"
    assert cards["Skipped"] == "1"
    assert cards["Total Trials"] == "4"  # the superseded attempt is not a trial
    assert cards["Retried Attempts"] == "1"
    assert cards["Failed"] == "0"


def test_sub_millisecond_timings_keep_significant_figures(tmp_path: Path):
    summary = _summary(
        tmp_path,
        [{"impl": "a", "n": 1, "time_ms_median": 0.0019}, {"impl": "a", "n": 2, "time_ms_median": 0.0603}],
    )
    page = generate_report(summary)
    assert _cards(page)["Fastest"] == "0.00190 ms"
    assert "0.0603" in page and "e-02" not in page


def test_format_number():
    assert format_number(0.0019) == "0.00190"
    assert format_number(17.295) == "17.3"
    assert format_number(123456.7) == "123,457"
    assert format_number(0.0) == "0"
    assert format_number(2e-6) == "2.00e-06"
    assert _fmt_val(3.0, "time_ms_count") == "3"
    assert _fmt_val("<b>", "impl") == "&lt;b&gt;"


def test_report_lists_grid_points_without_a_measurement(tmp_path: Path):
    summary = _summary(
        tmp_path,
        [
            {"impl": "a", "n": 1, "time_ms_median": 1.0, "ok": 3, "timeout": 0},
            {"impl": "a", "n": 2, "time_ms_median": float("nan"), "ok": 0, "timeout": 3},
        ],
    )
    page = generate_report(summary)
    assert "Grid Points Without a Measurement" in page
    assert _cards(page)["Without a Measurement"] == "1"
    assert _cards(page)["Fastest"] == "1.00 ms"
