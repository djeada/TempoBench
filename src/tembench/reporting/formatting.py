"""Column labels, value formatting, and HTML building blocks for tables."""

from __future__ import annotations

import html
import math

import pandas as pd

# ---------------------------------------------------------------------------
# Column display names for tables
# ---------------------------------------------------------------------------

_COL_LABELS = {
    "bench": "Benchmark",
    "_series": "Series",
    "impl": "Impl",
    "n": "n",
    "time_ms_median": "Time Med (ms)",
    "time_ms_mean": "Time Mean (ms)",
    "time_ms_count": "Runs",
    "time_ms_p10": "Time P10 (ms)",
    "time_ms_p90": "Time P90 (ms)",
    "time_source": "Time From",
    "wall_ms_median": "Wall Med (ms)",
    "wall_ms_mean": "Wall Mean (ms)",
    "wall_ms_count": "Runs",
    "wall_ms_p10": "Wall P10 (ms)",
    "wall_ms_p90": "Wall P90 (ms)",
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
    "runner_up": "Runner-up",
    "model_margin": "Margin",
    "confidence": "Confidence",
    "confidence_notes": "Caveats",
}


def _col_label(col: str) -> str:
    return _COL_LABELS.get(col, col)


def format_number(val: float, digits: int = 3) -> str:
    """Round `val` to `digits` significant figures in plain notation.

    Benchmarks span microseconds to minutes: fixed decimals turn a 0.0019 ms
    timing into "0.0" and scientific notation makes a column hard to scan, so
    only magnitudes too small to write plainly fall back to an exponent.
    """
    if not math.isfinite(val):
        return str(val)
    if val == 0:
        return "0"
    magnitude = abs(val)
    if magnitude >= 10**digits:
        return f"{val:,.0f}"
    if magnitude < 1e-4:
        return f"{val:.{digits - 1}e}"
    decimals = digits - 1 - math.floor(math.log10(magnitude))
    return f"{val:,.{max(decimals, 0)}f}"


def _fmt_val(val, col: str) -> str:
    """Format a cell value as escaped HTML, by its type and column name."""
    if pd.isna(val):
        return '<span class="na">—</span>'
    if isinstance(val, float):
        if "pct" in col:
            return f"{val:.1f}%"
        # A count column turns float once a failed grid point leaves it empty.
        if val.is_integer() and abs(val) < 1e15:
            return f"{int(val):,}"
        return format_number(val)
    # Cell text comes from user-defined grid values and command output.
    return html.escape(str(val))


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
        h.append(f"<th>{html.escape(_col_label(str(col)))}</th>")
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
    """Stat tile; `value` and `label` are inserted as HTML, so escape user text."""
    cls = f"stat-card {variant}".strip()
    return f'<div class="{cls}"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>'
