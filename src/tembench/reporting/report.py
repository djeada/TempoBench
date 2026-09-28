"""Main HTML report builder."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import PROJECT_URL
from ..runner.provenance import read_provenance
from ..summarize import preferred_time_column
from ..system import get_system_info
from .extract import _extract_vega_spec
from .formatting import _stat_card, _table_html, format_number
from .resources import (
    json_for_script,
    render_head_assets,
    render_theme_toggle,
    vega_script_tags,
)

#: Trial statuses that mean the trial did not produce a measurement.
_FAILURE_STATUSES = ("failed", "timeout", "error")
#: Per-status counts a summary may carry, in the order the report shows them.
_STATUS_COLUMNS = ("ok", "failed", "timeout", "error", "skipped")


def _run_statistics(rows: list[dict]) -> str:
    """Stat cards counting every trial status in the raw runs.

    A "retried" record is an attempt superseded by a later one of the same
    trial, so it is shown on its own and not counted as a trial or a failure.
    """
    counts = Counter(str(r.get("status", "unknown")) for r in rows)
    retried = counts.pop("retried", 0)
    cards = [
        _stat_card(str(counts.pop("ok", 0)), "Successful", "ok"),
        _stat_card(str(counts.pop("timeout", 0)), "Timeouts", "warn"),
        _stat_card(str(counts.pop("failed", 0)), "Failed", "err"),
    ]
    # Rarer outcomes only get a card when they happened.
    for status, title, variant in (("error", "Errors", "err"), ("skipped", "Skipped", "warn")):
        if counts.get(status):
            cards.append(_stat_card(str(counts.pop(status)), title, variant))
    for status, count in sorted(counts.items()):
        cards.append(_stat_card(str(count), html.escape(status.title()), "warn"))
    cards.append(_stat_card(str(len(rows) - retried), "Total Trials", ""))
    if retried:
        cards.append(_stat_card(str(retried), "Retried Attempts", ""))
    return "\n        ".join(cards)


def _unmeasured_section(df: pd.DataFrame, time_col: str | None) -> str:
    """List the grid points where no trial succeeded, with what went wrong.

    They stay in the summary so a point that stopped working is not mistaken
    for one never measured; charts cannot draw them, so the report names them.
    """
    if time_col is None or df.empty:
        return ""
    missing = df[pd.to_numeric(df[time_col], errors="coerce").isna()]
    if missing.empty:
        return ""
    keep = [
        c
        for c in missing.columns
        if not str(c).endswith(("_median", "_mean", "_count", "_p10", "_p90"))
        and c != "time_source"
    ]
    return f"""
    <div class="section">
      <h2><span class="icon">⚠️</span> Grid Points Without a Measurement</h2>
      <p class="desc">No trial succeeded at these {len(missing)} grid point(s), so they
        have no timing and are left out of the charts and fits. The status counts say why.</p>
      {_table_html(missing[keep])}
    </div>"""


def generate_report(
    summary_csv: Path,
    runs_jsonl: Optional[Path] = None,
    fits_csv: Optional[Path] = None,
    chart_html: Optional[Path] = None,
    title: str = "TempoBench Report",
    output_path: Optional[Path] = None,
    provenance_json: Optional[Path] = None,
) -> str:
    """Generate a comprehensive HTML report.

    The System Information section describes the machine the benchmark ran on,
    taken from the provenance snapshot written next to the results.  Falling
    back to the current machine is only correct when the report is produced
    where the run happened, so that case is labelled rather than assumed.
    """
    df = pd.read_csv(summary_csv)

    recorded = read_provenance(provenance_json) if provenance_json else None
    recorded_system = (recorded or {}).get("system")
    if isinstance(recorded_system, dict) and recorded_system:
        sysinfo = recorded_system
        sysinfo_origin = "Recorded when the benchmark ran"
    else:
        sysinfo = get_system_info()
        sysinfo_origin = "This machine — no provenance snapshot was found"
    esc = html.escape

    # Results built up with `--append` can span several runs, and comparing
    # timings measured on different hardware is meaningless — so say so.
    earlier_runs = (recorded or {}).get("previous") or []
    if isinstance(earlier_runs, list) and earlier_runs:
        hosts = {
            str((run.get("system") or {}).get("hostname", "unknown"))
            for run in earlier_runs
            if isinstance(run, dict)
        }
        hosts.add(str(sysinfo.get("hostname", "unknown")))
        total_runs = len(earlier_runs) + 1
        sysinfo_origin = (
            f"Recorded when the benchmark ran — results combine {total_runs} appended runs"
        )
        if len(hosts) > 1:
            sysinfo_origin += (
                f'. <strong class="warn-text">Those runs used {len(hosts)} different '
                "machines, so their timings are not comparable.</strong>"
            )

    cards = []
    time_col = preferred_time_column(df.columns)
    times = pd.to_numeric(df[time_col], errors="coerce").dropna() if time_col else None
    if times is not None and not times.empty:
        cards.append(_stat_card(f"{format_number(times.min())} ms", "Fastest"))
        cards.append(_stat_card(f"{format_number(times.max())} ms", "Slowest"))
        cards.append(_stat_card(f"{format_number(times.mean())} ms", "Average"))
    if "peak_rss_mb_median" in df.columns:
        rss = pd.to_numeric(df["peak_rss_mb_median"], errors="coerce").dropna()
        if not rss.empty:
            cards.append(_stat_card(f"{format_number(rss.max())} MB", "Peak Memory"))
    cards.append(_stat_card(str(len(df)), "Configurations"))
    if times is not None and len(times) < len(df):
        cards.append(
            _stat_card(str(len(df) - len(times)), "Without a Measurement", "err")
        )
    overview_cards = "\n".join(cards)

    runs_section = ""
    if runs_jsonl and runs_jsonl.exists():
        rows = []
        with runs_jsonl.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if rows:
            runs_section = f"""
    <div class="section">
      <h2><span class="icon">🏃</span> Run Statistics</h2>
      <div class="stat-grid">
        {_run_statistics(rows)}
      </div>
    </div>"""

    chart_section = ""
    if chart_html and chart_html.exists():
        raw = chart_html.read_text(encoding="utf-8")
        spec_json = _extract_vega_spec(raw)
        if spec_json:
            chart_section = f"""
    <div class="section">
      <h2><span class="icon">📊</span> Performance Charts</h2>
      <div class="chart-container"><div id="vis"></div></div>
{vega_script_tags("      ")}
      <script>vegaEmbed('#vis', {json_for_script(spec_json)}, {{renderer:'svg',actions:false}});</script>
    </div>"""
        else:
            chart_section = f"""
    <div class="section">
      <h2><span class="icon">📊</span> Performance Charts</h2>
      <iframe src="{esc(chart_html.name, quote=True)}" style="width:100%;height:520px;border:none;border-radius:8px;"></iframe>
    </div>"""

    fits_section = ""
    if fits_csv and fits_csv.exists():
        try:
            fits_df = pd.read_csv(fits_csv)
        except pd.errors.EmptyDataError:
            fits_df = pd.DataFrame()
        if not fits_df.empty:
            fits_section = f"""
    <div class="section">
      <h2><span class="icon">📐</span> Complexity Analysis</h2>
      <p class="desc">Best-fit Big-O complexity class per implementation, selected via AIC.
        The upper-bound curve satisfies T(n) ≤ C·f(n) + baseline for all observed data.
        <strong>Confidence</strong> states whether the measurements can support the class:
        anything below <em>high</em> lists the caveats that weakened it.</p>
      {_table_html(fits_df, highlight_col='formula')}
    </div>"""

    summary_table = _table_html(df)
    unmeasured_section = _unmeasured_section(df, time_col)

    def _si(key: str, label: str) -> str:
        val = sysinfo.get(key, "N/A") or "N/A"
        return _si_value(label, str(val))

    def _si_value(label: str, value: str) -> str:
        return (
            '<div class="sysinfo-row">'
            f'<span class="sysinfo-key">{esc(label)}</span>'
            f'<span class="sysinfo-val">{esc(value)}</span>'
            "</div>"
        )

    cpu_cores = (
        f"{sysinfo.get('cpu_count_physical', 'N/A')} physical / "
        f"{sysinfo.get('cpu_count_logical', 'N/A')} logical"
    )
    memory_total = f"{sysinfo.get('memory_total_gb', 'N/A')} GB"

    run_rows = ""
    if recorded:
        run_rows = "".join(
            _si_value(label, str(recorded[key]))
            for label, key in [
                ("Run At", "ts"),
                ("Seed", "seed"),
                ("Workers", "workers"),
                ("Working Dir", "cwd"),
                ("Invocation", "cmdline"),
            ]
            if key in recorded
        )

    sysinfo_html = f"""
      <p class="desc">{sysinfo_origin}</p>
      <div class="sysinfo-grid">
        <div>
          {_si('platform', 'Platform')}
          {_si('python_version', 'Python')}
          {_si('processor', 'Processor')}
        </div>
        <div>
          {_si_value('CPU Cores', cpu_cores)}
          {_si_value('Memory', memory_total)}
          {_si('architecture', 'Architecture')}
        </div>
      </div>
      {f'<div class="sysinfo-grid"><div>{run_rows}</div></div>' if run_rows else ''}"""

    generated_at = get_system_info()["timestamp"]
    date_str = generated_at[:10]
    ts_str = generated_at
    head_assets = render_head_assets()
    theme_toggle_html = render_theme_toggle()

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{esc(title)}</title>
{head_assets}
</head>
<body>
  <div class="container">

    <div class="report-header">
      <h1>{esc(title)}</h1>
      <div class="meta">Generated on {date_str}</div>
    </div>

    <div class="section">
      <h2><span class="icon">⚡</span> Performance Overview</h2>
      <div class="stat-grid">{overview_cards}</div>
    </div>

    {runs_section}

    {chart_section}

    <div class="section">
      <h2><span class="icon">📋</span> Detailed Results</h2>
      {summary_table}
    </div>

    {unmeasured_section}

    {fits_section}

    <div class="section">
      <h2><span class="icon">🖥️</span> System Information</h2>
      {sysinfo_html}
    </div>

    <div class="report-footer">
      <p>Generated by <a href="{PROJECT_URL}">TempoBench</a> · {ts_str}</p>
    </div>

  </div>

{theme_toggle_html}
</body>
</html>"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(page, encoding="utf-8")

    return page
