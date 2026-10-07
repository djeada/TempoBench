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
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "ts": "t", "cwd": "/tmp/<b>", "cmdline": f"tembench run {EVIL}",
        "system": {"platform": "<i>linux</i>", "hostname": "h"},
    }))
    page = generate_report(summary, title=f"T {EVIL}", provenance_json=provenance)

    assert "<script>alert" not in page
    assert "<img src=x" not in page
    assert "<i>linux</i>" not in page and "/tmp/<b>" not in page
    assert "&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_report_loads_the_vega_versions_altair_targets(tmp_path: Path):
    summary = _summary(tmp_path, [{"impl": "a", "n": n, "time_ms_median": n} for n in (1, 10)])
    page = generate_report(summary)
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
    assert "<td class=\"num\">0.00190</td>" in page
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


def _growing(tmp_path: Path) -> Path:
    return _summary(
        tmp_path,
        [
            {"impl": impl, "n": n, "time_ms_median": k * n}
            for impl, k in (("a", 1e-3), ("b", 3e-3))
            for n in (1000, 4000, 16000, 64000, 256000)
        ],
    )


def test_report_leads_with_the_complexity_classes(tmp_path: Path):
    page = generate_report(_growing(tmp_path))
    assert page.index("Complexity Classes") < page.index("Detailed Results")
    assert page.index("Complexity Analysis") < page.index("Detailed Results")
    assert 'class="fit-class">O(n)' in page
    # An average over different input sizes means nothing, so it is gone.
    assert "Average" not in _cards(page) and "Fastest" not in _cards(page)


def test_report_draws_its_chart_from_the_summary(tmp_path: Path):
    """No runtime.html needs to exist next to the summary for a chart."""
    page = generate_report(_growing(tmp_path))
    assert 'id="chart-0"' in page and "tb-chart-specs" in page
    assert not (tmp_path / "runtime.html").exists()


def test_fits_table_names_residuals_apart_from_memory(tmp_path: Path):
    page = generate_report(_growing(tmp_path))
    headline = page[page.index("Complexity Analysis"): page.index("Every fit column")]
    assert ">RSS<" not in page
    assert "Fitted class" not in headline, "the raw class only shows when it differs"
    assert "Resid. SS" in page  # still available, in the full fit columns


def test_report_cards_show_a_close_rival_class(tmp_path: Path):
    fits = pd.DataFrame([{
        "impl": "a", "model": "O(n)", "display_model": "O(n)", "formula": "T(n) ≤ n",
        "runner_up": "O(n log n)", "confidence": "medium",
        "confidence_notes": "another class fits almost as well", "caveats": "ambiguous-class",
    }])
    page = generate_report(_growing(tmp_path), fits_csv=fits)
    assert '<span class="fit-alt">or O(n log n)</span>' in page
    assert "medium confidence" in page


def test_every_page_type_shares_one_shell(tmp_path: Path):
    from tembench.reporting.comparison import compare_summaries, generate_comparison_report

    report = generate_report(_growing(tmp_path))
    comparison = generate_comparison_report(compare_summaries(_growing(tmp_path), _growing(tmp_path)))
    chart = tmp_path / "rt.html"
    save_chart(plot_runtime(_growing(tmp_path)), chart)
    for page in (report, comparison, chart.read_text()):
        assert '<header class="page-header">' in page
        assert page.index('id="themeToggle"') < page.index("</header>"), "the toggle sits in the header"
        assert "linear-gradient" not in page


def test_fits_without_caveats_read_from_csv_show_no_nan(tmp_path: Path):
    # Empty caveats in fits.csv read back as NaN, which must not print as "nan".
    from tembench.complexity import fit_models

    rows = [{"bench": "a", "n": n, "time_ms_median": n * 1e-3} for n in (100, 1000, 10000, 100000)]
    summary = _summary(tmp_path, rows)
    fits_path = tmp_path / "fits.csv"
    fit_models(pd.read_csv(summary), "n", "time_ms_median", ["bench"]).to_csv(fits_path, index=False)
    assert pd.read_csv(fits_path)["caveats"].isna().all()

    page = generate_report(summary, fits_csv=fits_path)
    assert not re.search(r">\s*nan\s*<", page, re.IGNORECASE)
