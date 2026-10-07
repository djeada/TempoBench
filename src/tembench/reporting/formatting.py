"""Column labels, value formatting, and HTML building blocks for tables."""

from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column display names — shared by chart axes, tooltips and tables
# ---------------------------------------------------------------------------

COLUMN_LABELS = {
    "bench": "Benchmark",
    "_series": "Series",
    "impl": "Implementation",
    "n": "Input size (n)",
    "time_ms": "Time (ms)",
    "time_ms_median": "Median time (ms)",
    "time_ms_mean": "Mean time (ms)",
    "time_ms_p10": "P10 time (ms)",
    "time_ms_p90": "P90 time (ms)",
    "time_ms_count": "Trials",
    "time_source": "Time source",
    "wall_ms": "Wall time (ms)",
    "wall_ms_median": "Median wall time (ms)",
    "wall_ms_mean": "Mean wall time (ms)",
    "wall_ms_p10": "P10 wall time (ms)",
    "wall_ms_p90": "P90 wall time (ms)",
    "wall_ms_count": "Trials",
    "reported_ms": "Reported time (ms)",
    "peak_rss_mb": "Peak memory (MB)",
    "peak_rss_mb_median": "Median peak memory (MB)",
    "peak_rss_mb_mean": "Mean peak memory (MB)",
    "ok": "OK",
    "failed": "Failed",
    "timeout": "Timeout",
    "error": "Error",
    "skipped": "Skipped",
    "retried": "Retried",
    # Complexity fits
    "model": "Fitted class",
    "display_model": "Complexity",
    "C": "C",
    "baseline": "Baseline",
    "offset": "Offset",
    "formula": "Upper bound",
    # Residual sum of squares, spelled out so it is not read as peak RSS memory.
    "rss": "Resid. SS",
    "nobs": "Points",
    "empirical_exponent": "Exponent",
    "exponent_ci_low": "Exponent CI low",
    "exponent_ci_high": "Exponent CI high",
    "runner_up": "Runner-up",
    "model_margin": "Margin",
    "confidence": "Confidence",
    "confidence_notes": "Caveats",
    "caveats": "Caveat codes",
    # Comparisons
    "compared_on": "Compared on",
    "problem": "Problem",
}

#: Comparison column suffixes and how each one renames its metric.
_COMPARISON_SUFFIXES = (
    ("_delta_pct", "{name} Δ%"),
    ("_delta", "{name} Δ{unit}"),
    ("_current", "{name}, current{unit}"),
    ("_baseline", "{name}, baseline{unit}"),
    ("_regression", "Regression"),
)


def _split_unit(text: str) -> tuple[str, str]:
    """Split "Median time (ms)" into ("Median time", " (ms)")."""
    if text.endswith(")") and " (" in text:
        cut = text.rindex(" (")
        return text[:cut], text[cut:]
    return text, ""


def column_label(col: str) -> str:
    """Human-readable name of a summary, fit or comparison column.

    Grid axes are user-defined and have no entry, so they keep their own name.
    """
    if col in COLUMN_LABELS:
        return COLUMN_LABELS[col]
    for suffix, template in _COMPARISON_SUFFIXES:
        if col.endswith(suffix) and col[: -len(suffix)] in COLUMN_LABELS:
            name, unit = _split_unit(COLUMN_LABELS[col[: -len(suffix)]])
            return template.format(name=name, unit=unit)
    return col


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


def format_pct(val: float) -> str:
    """A signed percentage change: "+12.3%", "-4.0%"."""
    return f"{val:+.1f}%"


def _fmt_val(val, col: str) -> str:
    """Format a cell value as escaped HTML, by its type and column name."""
    if pd.isna(val):
        return '<span class="na">—</span>'
    if isinstance(val, (bool, np.bool_)):
        return "yes" if val else "no"
    if isinstance(val, (int, np.integer)):
        return f"{int(val):,}"
    if isinstance(val, float):
        if col.endswith("_pct"):
            return format_pct(val)
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
    df: pd.DataFrame,
    cls: str = "data-table",
    highlight_col: str | None = None,
    cell_class=None,
) -> str:
    """Render a DataFrame as a styled HTML table.

    `cell_class(col, value)` may return a CSS class for a cell, e.g. to colour
    the one column a verdict rests on.
    """
    if df.empty:
        return '<p class="empty-msg">No data available.</p>'

    h = [f'<div class="table-wrap"><table class="{cls}">']
    h.append("<thead><tr>")
    for col in df.columns:
        h.append(f"<th>{html.escape(column_label(str(col)))}</th>")
    h.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        h.append("<tr>")
        for col in df.columns:
            val = row[col]
            classes = []
            if highlight_col and col == highlight_col:
                classes.append("highlight")
            if cell_class is not None:
                extra = cell_class(col, val)
                if extra:
                    classes.append(extra)
            if isinstance(val, (int, float, np.number)) and not isinstance(val, (bool, np.bool_)):
                classes.append("num")
            elif isinstance(val, str) and len(val) > 40:
                classes.append("wrap")  # prose, such as caveats, wraps

            css = f' class="{" ".join(classes)}"' if classes else ""
            h.append(f"<td{css}>{_fmt_val(val, str(col))}</td>")
        h.append("</tr>")

    h.append("</tbody></table></div>")
    return "\n".join(h)


def _stat_card(value: str, label: str, variant: str = "", detail: str = "") -> str:
    """Stat tile; every argument is inserted as HTML, so escape user text."""
    cls = f"stat-card {variant}".strip()
    extra = f'<div class="stat-detail">{detail}</div>' if detail else ""
    return (
        f'<div class="{cls}"><div class="stat-value">{value}</div>'
        f'<div class="stat-label">{label}</div>{extra}</div>'
    )
