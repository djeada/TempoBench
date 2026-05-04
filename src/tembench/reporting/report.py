"""Main HTML report builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .extract import _extract_vega_spec
from .formatting import _stat_card, _table_html
from .resources import render_head_assets, render_theme_toggle
from .system import get_system_info


def generate_report(
    summary_csv: Path,
    runs_jsonl: Optional[Path] = None,
    fits_csv: Optional[Path] = None,
    chart_html: Optional[Path] = None,
    title: str = "TempoBench Report",
    output_path: Optional[Path] = None,
) -> str:
    """Generate a comprehensive HTML report."""
    df = pd.read_csv(summary_csv)
    sysinfo = get_system_info()

    cards = []
    if "wall_ms_median" in df.columns:
        cards.append(_stat_card(f"{df['wall_ms_median'].min():.1f} ms", "Fastest"))
        cards.append(_stat_card(f"{df['wall_ms_median'].max():.1f} ms", "Slowest"))
        cards.append(_stat_card(f"{df['wall_ms_median'].mean():.1f} ms", "Average"))
    if "peak_rss_mb_median" in df.columns:
        cards.append(
            _stat_card(f"{df['peak_rss_mb_median'].max():.1f} MB", "Peak Memory")
        )
    cards.append(_stat_card(str(len(df)), "Configurations"))
    overview_cards = "\n".join(cards)

    runs_section = ""
    if runs_jsonl and runs_jsonl.exists():
        rows = []
        with runs_jsonl.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if rows:
            ok = sum(1 for r in rows if r.get("status") == "ok")
            fail = sum(1 for r in rows if r.get("status") == "failed")
            tout = sum(1 for r in rows if r.get("status") == "timeout")
            runs_section = f"""
    <div class="section">
      <h2><span class="icon">🏃</span> Run Statistics</h2>
      <div class="stat-grid">
        {_stat_card(str(ok), 'Successful', 'ok')}
        {_stat_card(str(tout), 'Timeouts', 'warn')}
        {_stat_card(str(fail), 'Failed', 'err')}
        {_stat_card(str(len(rows)), 'Total Runs', '')}
      </div>
    </div>"""

    chart_section = ""
    if chart_html and chart_html.exists():
        raw = chart_html.read_text()
        spec_json = _extract_vega_spec(raw)
        if spec_json:
            chart_section = f"""
    <div class="section">
      <h2><span class="icon">📊</span> Performance Charts</h2>
      <div class="chart-container"><div id="vis"></div></div>
      <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
      <script>vegaEmbed('#vis', {spec_json}, {{renderer:'svg',actions:false}});</script>
    </div>"""
        else:
            chart_section = f"""
    <div class="section">
      <h2><span class="icon">📊</span> Performance Charts</h2>
      <iframe src="{chart_html.name}" style="width:100%;height:520px;border:none;border-radius:8px;"></iframe>
    </div>"""

    fits_section = ""
    if fits_csv and fits_csv.exists():
        fits_df = pd.read_csv(fits_csv)
        if not fits_df.empty:
            fits_section = f"""
    <div class="section">
      <h2><span class="icon">📐</span> Complexity Analysis</h2>
      <p class="desc">Best-fit Big-O complexity class per implementation, selected via AIC.
        The upper-bound curve satisfies T(n) ≤ C·f(n) + baseline for all observed data.</p>
      {_table_html(fits_df, highlight_col='formula')}
    </div>"""

    summary_table = _table_html(df)

    def _si(key: str, label: str) -> str:
        val = sysinfo.get(key, "N/A") or "N/A"
        return _si_value(label, str(val))

    def _si_value(label: str, value: str) -> str:
        return (
            '<div class="sysinfo-row">'
            f'<span class="sysinfo-key">{label}</span>'
            f'<span class="sysinfo-val">{value}</span>'
            "</div>"
        )

    cpu_cores = (
        f"{sysinfo['cpu_count_physical']} physical / "
        f"{sysinfo['cpu_count_logical']} logical"
    )
    memory_total = f"{sysinfo['memory_total_gb']} GB"

    sysinfo_html = f"""
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
      </div>"""

    date_str = sysinfo["timestamp"][:10]
    ts_str = sysinfo["timestamp"]
    head_assets = render_head_assets()
    theme_toggle_html = render_theme_toggle()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
{head_assets}
</head>
<body>
  <div class="container">

    <div class="report-header">
      <h1>{title}</h1>
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

    {fits_section}

    <div class="section">
      <h2><span class="icon">🖥️</span> System Information</h2>
      {sysinfo_html}
    </div>

    <div class="report-footer">
      <p>Generated by <a href="https://github.com/tempobench/tempobench">TempoBench</a> · {ts_str}</p>
    </div>

  </div>

{theme_toggle_html}
</body>
</html>"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html
