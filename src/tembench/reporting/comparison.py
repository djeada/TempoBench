"""Comparison helpers and HTML comparison report."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .formatting import _col_label, _stat_card
from .resources import render_head_assets, render_theme_toggle
from .system import get_system_info


def compare_summaries(
    current_csv: Path,
    baseline_csv: Path,
    threshold_pct: float = 5.0,
) -> pd.DataFrame:
    """Compare current results against a baseline."""
    current = pd.read_csv(current_csv)
    baseline = pd.read_csv(baseline_csv)

    group_cols = [
        c
        for c in ["bench", "impl", "n"]
        if c in current.columns and c in baseline.columns
    ]
    if not group_cols:
        group_cols = ["bench"] if "bench" in current.columns else []
    if not group_cols:
        return pd.DataFrame()

    merged = current.merge(
        baseline, on=group_cols, suffixes=("_current", "_baseline"), how="outer"
    )
    result_cols = list(group_cols)

    for metric in ["wall_ms_median", "wall_ms_mean"]:
        curr_col, base_col = f"{metric}_current", f"{metric}_baseline"
        if curr_col in merged.columns and base_col in merged.columns:
            merged[f"{metric}_delta"] = merged[curr_col] - merged[base_col]
            merged[f"{metric}_delta_pct"] = (
                (merged[curr_col] - merged[base_col]) / merged[base_col] * 100
            ).round(2)
            merged[f"{metric}_regression"] = (
                merged[f"{metric}_delta_pct"] > threshold_pct
            )
            result_cols.extend(
                [
                    curr_col,
                    base_col,
                    f"{metric}_delta",
                    f"{metric}_delta_pct",
                    f"{metric}_regression",
                ]
            )

    for metric in ["peak_rss_mb_median", "peak_rss_mb_mean"]:
        curr_col, base_col = f"{metric}_current", f"{metric}_baseline"
        if curr_col in merged.columns and base_col in merged.columns:
            merged[f"{metric}_delta"] = merged[curr_col] - merged[base_col]
            merged[f"{metric}_delta_pct"] = (
                (merged[curr_col] - merged[base_col]) / merged[base_col] * 100
            ).round(2)
            result_cols.extend(
                [curr_col, base_col, f"{metric}_delta", f"{metric}_delta_pct"]
            )

    result_cols = [c for c in result_cols if c in merged.columns]
    return merged[result_cols]


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
    improvements = sum(
        (comparison_df[col] < -threshold_pct).sum() for col in delta_pct_cols
    )

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
    banner_text = (
        "No regressions detected"
        if total_regressions == 0
        else f"{int(total_regressions)} regression{'s' if total_regressions != 1 else ''} detected"
    )
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

{theme_toggle_html}
</body>
</html>"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html
