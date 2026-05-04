"""Column labels, value formatting, and HTML building blocks for tables."""

from __future__ import annotations

import pandas as pd

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
    "display_model": "Displayed Complexity",
    "C": "C",
    "C_ols": "C (OLS)",
    "baseline": "Baseline",
    "offset": "Offset",
    "formula": "Upper Bound",
    "rss": "RSS",
    "nobs": "Obs",
    "empirical_exponent": "Empirical Exp",
    "exponent_ci_low": "Exp CI Low",
    "exponent_ci_high": "Exp CI High",
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


def _table_html(
    df: pd.DataFrame, cls: str = "data-table", highlight_col: str | None = None
) -> str:
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
