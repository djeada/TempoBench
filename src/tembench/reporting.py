"""HTML report generation for TempoBench."""
from __future__ import annotations

import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import psutil


def get_system_info() -> dict:
    """Gather system information for reproducibility."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "hostname": platform.node(),
    }
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            info["cpu_freq_mhz"] = round(cpu_freq.current, 1)
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Column display names for tables
# ---------------------------------------------------------------------------

_COL_LABELS = {
    "bench": "Benchmark",
    "impl": "Impl",
    "n": "n",
    "wall_ms_median": "Time Med (ms)",
    "wall_ms_mean": "Time Mean (ms)",
    "wall_ms_count": "Runs",
    "wall_ms_p10": "Time P10 (ms)",
    "wall_ms_p90": "Time P90 (ms)",
    "peak_rss_mb_median": "RSS Med (MB)",
    "peak_rss_mb_mean": "RSS Mean (MB)",
    "ok": "OK",
    "model": "Complexity",
    "C": "C",
    "C_ols": "C (OLS)",
    "baseline": "Baseline",
    "offset": "Offset",
    "formula": "Upper Bound",

    "rss": "RSS",
    "nobs": "Obs",
}


def _col_label(col: str) -> str:
    return _COL_LABELS.get(col, col)


def _fmt_val(val, col: str) -> str:
    """Format a cell value depending on its type and column name."""
    if pd.isna(val):
        return '<span class="na">—</span>'
    if isinstance(val, float):
        if "pct" in col:
            return f"{val:.1f}%"
        if abs(val) >= 100:
            return f"{val:,.1f}"
        if abs(val) >= 1:
            return f"{val:.3f}"
        return f"{val:.3e}"
    return str(val)


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------

def _table_html(df: pd.DataFrame, cls: str = "data-table", highlight_col: str | None = None) -> str:
    """Render a DataFrame as a styled HTML table."""
    if df.empty:
        return '<p class="empty-msg">No data available.</p>'

    h = [f'<div class="table-wrap"><table class="{cls}">']
    h.append("<thead><tr>")
    for col in df.columns:
        h.append(f"<th>{_col_label(col)}</th>")
    h.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        h.append("<tr>")
        for col in df.columns:
            val = row[col]
            css = ""
            if highlight_col and col == highlight_col:
                css = ' class="highlight"'
            h.append(f"<td{css}>{_fmt_val(val, col)}</td>")
        h.append("</tr>")

    h.append("</tbody></table></div>")
    return "\n".join(h)


def _stat_card(value: str, label: str, variant: str = "") -> str:
    cls = f"stat-card {variant}".strip()
    return f'<div class="{cls}"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>'


def _extract_vega_spec(chart_html_text: str) -> str | None:
    """Extract the Vega-Lite JSON spec from an Altair-generated HTML file.

    Returns the spec as a string, or None if not found.
    """
    m = re.search(r'var\s+spec\s*=\s*(\{.*?\});\s*\n', chart_html_text, re.DOTALL)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --c-blue: #2563eb;
  --c-blue-dark: #1d4ed8;
  --c-green: #16a34a;
  --c-amber: #d97706;
  --c-red: #dc2626;
  --c-bg: #f8fafc;
  --c-surface: #ffffff;
  --c-text: #0f172a;
  --c-text2: #475569;
  --c-text3: #94a3b8;
  --c-border: #e2e8f0;
  --radius: 12px;
  --shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
  --shadow-md: 0 4px 6px rgba(0,0,0,.06), 0 2px 4px rgba(0,0,0,.04);
  --font: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, 'Cascadia Code', 'Fira Code', monospace;
}

[data-theme="dark"] {
  --c-blue: #3b82f6;
  --c-blue-dark: #2563eb;
  --c-green: #22c55e;
  --c-amber: #fbbf24;
  --c-red: #ef4444;
  --c-bg: #0f172a;
  --c-surface: #1e293b;
  --c-text: #f1f5f9;
  --c-text2: #cbd5e1;
  --c-text3: #64748b;
  --c-border: #334155;
  --shadow: 0 1px 3px rgba(0,0,0,.3), 0 1px 2px rgba(0,0,0,.2);
  --shadow-md: 0 4px 6px rgba(0,0,0,.3), 0 2px 4px rgba(0,0,0,.2);
}

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: var(--font);
  background: var(--c-bg);
  color: var(--c-text);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.container { max-width: 1100px; margin: 0 auto; padding: 2.5rem 1.5rem; }

/* ---- Header ---- */
.report-header {
  background: linear-gradient(135deg, var(--c-blue) 0%, var(--c-blue-dark) 100%);
  color: #fff;
  padding: 2.5rem 2rem;
  border-radius: var(--radius);
  margin-bottom: 2rem;
  text-align: center;
}
.report-header h1 {
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 0.35rem;
}
.report-header .meta {
  opacity: 0.85;
  font-size: 0.95rem;
}

/* ---- Sections ---- */
.section {
  background: var(--c-surface);
  border-radius: var(--radius);
  padding: 1.75rem;
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow);
}
.section h2 {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--c-text);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.section h2 .icon { font-size: 1.25rem; }
.section p.desc {
  color: var(--c-text2);
  font-size: 0.9rem;
  margin-bottom: 1rem;
}

/* ---- Stat cards ---- */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
}
.stat-card {
  background: var(--c-bg);
  border-radius: 10px;
  padding: 1.25rem 1rem;
  text-align: center;
  border: 1px solid var(--c-border);
  transition: box-shadow .15s;
}
.stat-card:hover { box-shadow: var(--shadow-md); }
.stat-card .stat-value {
  font-size: 1.65rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--c-text);
}
.stat-card .stat-label {
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--c-text3);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-top: 0.2rem;
}
.stat-card.ok    { border-left: 4px solid var(--c-green); }
.stat-card.warn  { border-left: 4px solid var(--c-amber); }
.stat-card.err   { border-left: 4px solid var(--c-red);   }

/* ---- Tables ---- */
.table-wrap { overflow-x: auto; border-radius: 8px; border: 1px solid var(--c-border); }
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
  white-space: nowrap;
}
.data-table th {
  background: var(--c-bg);
  font-weight: 600;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--c-text2);
  padding: 0.7rem 0.85rem;
  text-align: left;
  position: sticky;
  top: 0;
  border-bottom: 2px solid var(--c-border);
}
.data-table td {
  padding: 0.6rem 0.85rem;
  border-bottom: 1px solid var(--c-border);
  font-variant-numeric: tabular-nums;
}
.data-table tbody tr:last-child td { border-bottom: none; }
.data-table tbody tr:hover { background: #f1f5f9; }
[data-theme="dark"] .data-table tbody tr:hover { background: #334155; }
.data-table td.highlight {
  font-weight: 600;
  font-family: var(--font-mono);
  font-size: 0.82rem;
}
.data-table .na { color: var(--c-text3); }

/* ---- Chart ---- */
.chart-container {
  display: flex;
  justify-content: center;
  padding: 0.5rem 0;
}
.chart-container .vega-embed {
  width: 100% !important;
}
.chart-container .vega-embed details { display: none; }

/* ---- System info ---- */
.sysinfo-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 0 2rem;
}
.sysinfo-row {
  display: flex;
  justify-content: space-between;
  padding: 0.55rem 0;
  border-bottom: 1px dashed var(--c-border);
  font-size: 0.88rem;
}
.sysinfo-row:last-child { border-bottom: none; }
.sysinfo-key { color: var(--c-text2); }
.sysinfo-val { font-weight: 600; font-variant-numeric: tabular-nums; }

/* ---- Footer ---- */
.report-footer {
  text-align: center;
  padding: 2rem 0 1rem;
  color: var(--c-text3);
  font-size: 0.8rem;
}
.report-footer a { color: var(--c-blue); text-decoration: none; }
.report-footer a:hover { text-decoration: underline; }

/* ---- Comparison-specific ---- */
.status-banner {
  padding: 1rem 1.5rem;
  border-radius: var(--radius);
  margin-bottom: 1.5rem;
  text-align: center;
  font-size: 1.05rem;
  font-weight: 700;
}
.status-banner.pass {
  background: #f0fdf4; color: var(--c-green); border: 2px solid var(--c-green);
}
.status-banner.fail {
  background: #fef2f2; color: var(--c-red); border: 2px solid var(--c-red);
}
.data-table td.regression  { background: #fef2f2; color: var(--c-red);   font-weight: 600; }
.data-table td.improvement { background: #f0fdf4; color: var(--c-green); font-weight: 600; }
[data-theme="dark"] .data-table td.regression  { background: #4c1d1d; color: #fca5a5; }
[data-theme="dark"] .data-table td.improvement { background: #14532d; color: #86efac; }
[data-theme="dark"] .status-banner.pass {
  background: #14532d; color: #86efac;
}
[data-theme="dark"] .status-banner.fail {
  background: #4c1d1d; color: #fca5a5;
}

/* ---- Responsive ---- */
@media (max-width: 640px) {
  .container { padding: 1rem; }
  .report-header { padding: 2rem 1rem; }
  .report-header h1 { font-size: 1.5rem; }
  .stat-card .stat-value { font-size: 1.3rem; }
}

/* ---- Theme Toggle ---- */
.theme-toggle {
  position: fixed;
  top: 1.5rem;
  right: 1.5rem;
  background: var(--c-surface);
  border: 2px solid var(--c-border);
  border-radius: 50px;
  padding: 0.5rem 1rem;
  cursor: pointer;
  font-size: 1.2rem;
  box-shadow: var(--shadow-md);
  transition: all 0.3s ease;
  z-index: 1000;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  color: var(--c-text);
}
.theme-toggle:hover {
  transform: scale(1.05);
  box-shadow: 0 8px 12px rgba(0,0,0,.15);
}
.theme-toggle-icon {
  font-size: 1.3rem;
  transition: transform 0.3s ease;
}
.theme-toggle:active .theme-toggle-icon {
  transform: rotate(180deg);
}
"""


# ---------------------------------------------------------------------------
# Theme toggle JavaScript
# ---------------------------------------------------------------------------

_THEME_TOGGLE_HTML = """\
  <button class="theme-toggle" id="themeToggle" aria-label="Toggle theme">
    <span class="theme-toggle-icon" id="themeIcon">🌙</span>
    <span id="themeLabel">Dark</span>
  </button>

  <script>
    (function() {{
      const toggle = document.getElementById('themeToggle');
      const icon = document.getElementById('themeIcon');
      const label = document.getElementById('themeLabel');
      const html = document.documentElement;

      // Check for saved theme preference or default to light mode
      const currentTheme = localStorage.getItem('theme') || 'light';
      html.setAttribute('data-theme', currentTheme);

      // Update button state
      if (currentTheme === 'dark') {{
        icon.textContent = '☀️';
        label.textContent = 'Light';
      }}

      toggle.addEventListener('click', function() {{
        const theme = html.getAttribute('data-theme');
        const newTheme = theme === 'light' ? 'dark' : 'light';

        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);

        if (newTheme === 'dark') {{
          icon.textContent = '☀️';
          label.textContent = 'Light';
        }} else {{
          icon.textContent = '🌙';
          label.textContent = 'Dark';
        }}
      }});
    }})();
  </script>"""


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

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

    # ---- overview cards ----
    cards = []
    if "wall_ms_median" in df.columns:
        cards.append(_stat_card(f"{df['wall_ms_median'].min():.1f} ms", "Fastest"))
        cards.append(_stat_card(f"{df['wall_ms_median'].max():.1f} ms", "Slowest"))
        cards.append(_stat_card(f"{df['wall_ms_median'].mean():.1f} ms", "Average"))
    if "peak_rss_mb_median" in df.columns:
        cards.append(_stat_card(f"{df['peak_rss_mb_median'].max():.1f} MB", "Peak Memory"))
    cards.append(_stat_card(str(len(df)), "Configurations"))
    overview_cards = "\n".join(cards)

    # ---- run statistics ----
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

    # ---- chart embed ----
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

    # ---- fits table ----
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

    # ---- summary table ----
    summary_table = _table_html(df)

    # ---- sysinfo ----
    def _si(key: str, label: str) -> str:
        val = sysinfo.get(key, "N/A") or "N/A"
        return f'<div class="sysinfo-row"><span class="sysinfo-key">{label}</span><span class="sysinfo-val">{val}</span></div>'

    sysinfo_html = f"""
      <div class="sysinfo-grid">
        <div>
          {_si('platform', 'Platform')}
          {_si('python_version', 'Python')}
          {_si('processor', 'Processor')}
        </div>
        <div>
          <div class="sysinfo-row"><span class="sysinfo-key">CPU Cores</span><span class="sysinfo-val">{sysinfo['cpu_count_physical']} physical / {sysinfo['cpu_count_logical']} logical</span></div>
          <div class="sysinfo-row"><span class="sysinfo-key">Memory</span><span class="sysinfo-val">{sysinfo['memory_total_gb']} GB</span></div>
          {_si('architecture', 'Architecture')}
        </div>
      </div>"""

    date_str = sysinfo["timestamp"][:10]
    ts_str = sysinfo["timestamp"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>{_CSS}</style>
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

{_THEME_TOGGLE_HTML}
</body>
</html>"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compare_summaries(
    current_csv: Path,
    baseline_csv: Path,
    threshold_pct: float = 5.0,
) -> pd.DataFrame:
    """Compare current results against a baseline."""
    current = pd.read_csv(current_csv)
    baseline = pd.read_csv(baseline_csv)

    group_cols = [c for c in ["bench", "impl", "n"] if c in current.columns and c in baseline.columns]
    if not group_cols:
        group_cols = ["bench"] if "bench" in current.columns else []
    if not group_cols:
        return pd.DataFrame()

    merged = current.merge(baseline, on=group_cols, suffixes=("_current", "_baseline"), how="outer")
    result_cols = list(group_cols)

    for metric in ["wall_ms_median", "wall_ms_mean"]:
        curr_col, base_col = f"{metric}_current", f"{metric}_baseline"
        if curr_col in merged.columns and base_col in merged.columns:
            merged[f"{metric}_delta"] = merged[curr_col] - merged[base_col]
            merged[f"{metric}_delta_pct"] = ((merged[curr_col] - merged[base_col]) / merged[base_col] * 100).round(2)
            merged[f"{metric}_regression"] = merged[f"{metric}_delta_pct"] > threshold_pct
            result_cols.extend([curr_col, base_col, f"{metric}_delta", f"{metric}_delta_pct", f"{metric}_regression"])

    for metric in ["peak_rss_mb_median", "peak_rss_mb_mean"]:
        curr_col, base_col = f"{metric}_current", f"{metric}_baseline"
        if curr_col in merged.columns and base_col in merged.columns:
            merged[f"{metric}_delta"] = merged[curr_col] - merged[base_col]
            merged[f"{metric}_delta_pct"] = ((merged[curr_col] - merged[base_col]) / merged[base_col] * 100).round(2)
            result_cols.extend([curr_col, base_col, f"{metric}_delta", f"{metric}_delta_pct"])

    result_cols = [c for c in result_cols if c in merged.columns]
    return merged[result_cols]


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def generate_comparison_report(
    comparison_df: pd.DataFrame,
    title: str = "TempoBench Comparison Report",
    threshold_pct: float = 5.0,
    output_path: Optional[Path] = None,
) -> str:
    """Generate an HTML comparison report."""
    sysinfo = get_system_info()

    regression_cols = [c for c in comparison_df.columns if c.endswith("_regression")]
    total_regressions = sum(comparison_df[col].sum() for col in regression_cols)
    total_configs = len(comparison_df)

    delta_pct_cols = [c for c in comparison_df.columns if c.endswith("_delta_pct")]
    improvements = sum((comparison_df[col] < -threshold_pct).sum() for col in delta_pct_cols)

    # Build comparison table with conditional styling
    tbl = ['<div class="table-wrap"><table class="data-table">']
    tbl.append("<thead><tr>")
    for col in comparison_df.columns:
        tbl.append(f"<th>{_col_label(col)}</th>")
    tbl.append("</tr></thead><tbody>")

    for _, row in comparison_df.iterrows():
        tbl.append("<tr>")
        for col in comparison_df.columns:
            val = row[col]
            css = ""
            if col.endswith("_regression"):
                if val:
                    css = ' class="regression"'
                    val = "⚠ YES"
                else:
                    val = "✓ NO"
            elif col.endswith("_delta_pct") and pd.notna(val):
                if val > threshold_pct:
                    css = ' class="regression"'
                    val = f"+{val:.1f}%"
                elif val < -threshold_pct:
                    css = ' class="improvement"'
                    val = f"{val:.1f}%"
                else:
                    val = f"{val:.1f}%"
            elif isinstance(val, float):
                val = f"{val:.3f}"
            tbl.append(f"<td{css}>{val}</td>")
        tbl.append("</tr>")
    tbl.append("</tbody></table></div>")
    table_html = "\n".join(tbl)

    banner_cls = "pass" if total_regressions == 0 else "fail"
    banner_icon = "✓" if total_regressions == 0 else "⚠"
    banner_text = "No regressions detected" if total_regressions == 0 else f"{int(total_regressions)} regression{'s' if total_regressions != 1 else ''} detected"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>{_CSS}</style>
</head>
<body>
  <div class="container">

    <div class="report-header" style="background:linear-gradient(135deg,#7c3aed,#5b21b6)">
      <h1>{title}</h1>
      <div class="meta">Regression threshold: {threshold_pct}%</div>
    </div>

    <div class="status-banner {banner_cls}">{banner_icon} {banner_text}</div>

    <div class="section">
      <h2><span class="icon">📊</span> Summary</h2>
      <div class="stat-grid">
        {_stat_card(str(total_configs), 'Compared', '')}
        {_stat_card(str(int(total_regressions)), 'Regressions', 'err' if total_regressions > 0 else 'ok')}
        {_stat_card(str(int(improvements)), 'Improvements', 'ok')}
      </div>
    </div>

    <div class="section">
      <h2><span class="icon">🔍</span> Detailed Comparison</h2>
      <p class="desc">
        Cells in <span style="color:var(--c-red);font-weight:600">red</span> indicate regressions;
        <span style="color:var(--c-green);font-weight:600">green</span> indicates improvements.
      </p>
      {table_html}
    </div>

    <div class="report-footer">
      <p>Generated by <a href="https://github.com/tempobench/tempobench">TempoBench</a> · {sysinfo['timestamp']}</p>
    </div>

  </div>

{_THEME_TOGGLE_HTML}
</body>
</html>"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html
