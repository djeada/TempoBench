"""Comparison helpers and HTML comparison report."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import PROJECT_URL
from ..summarize import TIME_COLUMN_PREFERENCE, TIME_SOURCE_COL, grid_columns
from ..system import get_system_info
from .formatting import _col_label, _stat_card
from .resources import render_head_assets, render_theme_toggle


def _key_columns(current: pd.DataFrame, baseline: pd.DataFrame) -> list[str]:
    """Infer the columns that identify a grid point in both summaries.

    Assuming fixed names would join unrelated rows into a cartesian product for
    any sweep named differently, and silently report the resulting nonsense as
    regressions.
    """
    shared = [c for c in current.columns if c in baseline.columns]
    return grid_columns(shared)


#: Why a row's verdict could not come from its timings.  Each one fails the
#: comparison: CI must not pass a grid point it could not actually check.
MISSING_CURRENT = "missing in current"
NO_TIMING = "no successful trial"
SOURCE_CHANGED = "timing source changed"


def compare_summaries(
    current_csv: Path,
    baseline_csv: Path,
    threshold_pct: float = 5.0,
) -> pd.DataFrame:
    """Compare current results against a baseline."""
    current = pd.read_csv(current_csv)
    baseline = pd.read_csv(baseline_csv)

    group_cols = _key_columns(current, baseline)
    if not group_cols:
        return pd.DataFrame()

    merged = current.merge(
        baseline,
        on=group_cols,
        suffixes=("_current", "_baseline"),
        how="outer",
        indicator=True,
    )
    result_cols = list(group_cols)

    # Only one duration decides pass/fail.  A summary can carry both a
    # self-reported and a wall-clock family, and wall clock includes process
    # startup — letting it raise regressions too would both double-count and
    # turn ordinary startup jitter into failures.
    comparable = [
        m
        for m in TIME_COLUMN_PREFERENCE
        if f"{m}_current" in merged.columns and f"{m}_baseline" in merged.columns
    ]
    if not comparable:
        return pd.DataFrame()
    decisive = comparable[0]

    # A row whose canonical duration came from different sources on the two
    # sides (one self-reported, one wall clock) cannot be compared on it: the
    # gap would be process startup, not a change in the code.  Such rows fall
    # back to wall clock, which both sides always have.
    source_cols = (f"{TIME_SOURCE_COL}_current", f"{TIME_SOURCE_COL}_baseline")
    switched = pd.Series(False, index=merged.index)
    if all(c in merged.columns for c in source_cols):
        switched = (
            merged[source_cols[0]].notna()
            & merged[source_cols[1]].notna()
            & (merged[source_cols[0]] != merged[source_cols[1]])
        )

    for metric in comparable:
        curr_col, base_col = f"{metric}_current", f"{metric}_baseline"
        delta = merged[curr_col] - merged[base_col]
        delta_pct = (delta / merged[base_col] * 100).round(2)
        if metric == decisive and metric.startswith("time_ms"):
            delta = delta.mask(switched)
            delta_pct = delta_pct.mask(switched)
        merged[f"{metric}_delta"] = delta
        merged[f"{metric}_delta_pct"] = delta_pct
        result_cols.extend([curr_col, base_col, f"{metric}_delta", f"{metric}_delta_pct"])

    verdict_pct = merged[f"{decisive}_delta_pct"]
    wall_pct = "wall_ms_median_delta_pct"
    if switched.any() and wall_pct in merged.columns:
        verdict_pct = verdict_pct.where(~switched, merged[wall_pct])
        merged["compared_on"] = decisive.rsplit("_", 1)[0]
        merged.loc[switched, "compared_on"] = "wall_ms"
        result_cols.append("compared_on")

    # A grid point the baseline measured but the current run did not is a
    # failure, not a pass: NaN > threshold is False, so without this a point
    # that stopped working entirely would sail through.  A point the baseline
    # never measured has nothing to regress against.
    measured_before = merged[f"{decisive}_baseline"].notna()
    problem = pd.Series("", index=merged.index)
    problem[merged["_merge"] == "right_only"] = MISSING_CURRENT
    problem[(merged["_merge"] == "both") & merged[f"{decisive}_current"].isna()] = NO_TIMING
    if "compared_on" not in merged.columns:
        # The timing source switched and there is no wall clock to fall back on.
        problem[switched & (problem == "")] = SOURCE_CHANGED
    problem[~measured_before] = ""
    merged["problem"] = problem
    merged[f"{decisive}_regression"] = measured_before & (
        (verdict_pct > threshold_pct) | (problem != "")
    )
    result_cols.extend([f"{decisive}_regression", "problem"])

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
    return merged[result_cols].reset_index(drop=True)


def comparison_tally(comparison_df: pd.DataFrame, threshold_pct: float) -> dict[str, int]:
    """Count compared configurations, regressions, and improvements.

    Everything is counted on the single decisive metric.  Summing over every
    `_delta_pct` column would tally one faster configuration once per metric it
    happens to carry, letting a run report more wins than it compared.
    """
    regression_cols = [c for c in comparison_df.columns if c.endswith("_regression")]
    decisive_delta = next(
        (c.replace("_regression", "_delta_pct") for c in regression_cols), None
    )
    improvements = 0
    compared = len(comparison_df)
    if decisive_delta and decisive_delta in comparison_df.columns:
        verdict = comparison_df[decisive_delta]
        if "compared_on" in comparison_df.columns and "wall_ms_median_delta_pct" in comparison_df.columns:
            verdict = verdict.where(
                comparison_df["compared_on"] != "wall_ms",
                comparison_df["wall_ms_median_delta_pct"],
            )
        improvements = int((verdict < -threshold_pct).sum())
        # Rows present on one side only were matched against nothing.
        compared = int(verdict.notna().sum())
    problems = 0
    if "problem" in comparison_df.columns:
        problems = int((comparison_df["problem"].fillna("") != "").sum())

    return {
        "compared": compared,
        "regressions": int(sum(comparison_df[c].sum() for c in regression_cols)),
        "improvements": improvements,
        "unmeasured": problems,
    }


def generate_comparison_report(
    comparison_df: pd.DataFrame,
    title: str = "TempoBench Comparison Report",
    threshold_pct: float = 5.0,
    output_path: Optional[Path] = None,
) -> str:
    """Generate an HTML comparison report."""
    sysinfo = get_system_info()

    tally = comparison_tally(comparison_df, threshold_pct)
    total_regressions = tally["regressions"]
    total_configs = tally["compared"]
    improvements = tally["improvements"]

    tbl = ['<div class="table-wrap"><table class="data-table">']
    tbl.append("<thead><tr>")
    for col in comparison_df.columns:
        tbl.append(f"<th>{html.escape(_col_label(col))}</th>")
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
                val = "—" if pd.isna(val) else f"{val:.3f}"
            tbl.append(f"<td{css}>{html.escape(str(val))}</td>")
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

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
{head_assets}
</head>
<body>
  <div class="container">

    <div class="report-header" style="background:linear-gradient(135deg,#7c3aed,#5b21b6)">
      <h1>{html.escape(title)}</h1>
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
      <p>Generated by <a href="{PROJECT_URL}">TempoBench</a> · {sysinfo['timestamp']}</p>
    </div>

  </div>

{theme_toggle_html}
</body>
</html>"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(page, encoding="utf-8")

    return page
