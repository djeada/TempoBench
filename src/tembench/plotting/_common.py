from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import altair as alt
import pandas as pd

from ..summarize import grid_columns

PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be185d",
    "#65a30d",
]

NICE_LABELS = {
    "time_ms_median": "Time – Median (ms)",
    "time_ms_mean": "Time – Mean (ms)",
    "time_ms_p10": "Time – P10 (ms)",
    "time_ms_p90": "Time – P90 (ms)",
    "time_ms": "Time (ms)",
    "wall_ms_median": "Wall Time – Median (ms)",
    "wall_ms_mean": "Wall Time – Mean (ms)",
    "wall_ms_p10": "Wall Time – P10 (ms)",
    "wall_ms_p90": "Wall Time – P90 (ms)",
    "peak_rss_mb_median": "Peak RSS – Median (MB)",
    "peak_rss_mb_mean": "Peak RSS – Mean (MB)",
    "n": "Input Size (n)",
    "impl": "Implementation",
    "wall_ms": "Wall Time (ms)",
    "reported_ms": "Reported Time (ms)",
    "peak_rss_mb": "Peak RSS (MB)",
}


def apply_theme() -> None:
    """Register and enable a clean TempoBench theme."""
    theme_config = {
        "config": {
            "background": "#ffffff",
            "font": "Inter, system-ui, -apple-system, sans-serif",
            "title": {
                "fontSize": 16,
                "fontWeight": 600,
                "anchor": "start",
                "offset": 12,
            },
            "axis": {
                "labelFontSize": 12,
                "titleFontSize": 13,
                "titleFontWeight": 500,
                "titlePadding": 12,
                "gridColor": "#f1f5f9",
                "domainColor": "#cbd5e1",
                "tickColor": "#cbd5e1",
                "labelColor": "#475569",
                "titleColor": "#334155",
            },
            "legend": {
                "labelFontSize": 12,
                "titleFontSize": 13,
                "titleFontWeight": 500,
                "symbolSize": 120,
                "orient": "bottom",
                "direction": "horizontal",
                "padding": 12,
            },
            "view": {"stroke": None, "continuousWidth": 640, "continuousHeight": 400},
            "range": {"category": PALETTE},
            "line": {"strokeWidth": 2.5},
            "point": {"size": 60, "filled": True},
        }
    }

    if hasattr(alt, "theme") and hasattr(alt.theme, "register"):

        @alt.theme.register("tempobench", enable=True)
        def _tb_theme():
            return cast(Any, theme_config)

    else:
        alt.themes.register("tempobench", lambda: theme_config)
        alt.themes.enable("tempobench")


def label(col: str | None) -> str:
    """Human-readable axis title for a column, falling back to its own name."""
    if col is None:
        return ""
    return NICE_LABELS.get(col, col)


#: Tooltip format: four significant figures, so a 0.0019 ms timing is not
#: rounded to zero and a 17 000 ms one keeps its magnitude.
NUMBER_FORMAT = ",.4~r"

#: Per-status trial counts in a summary.  "retried" counts superseded attempts
#: of a trial that was run again; none of these identify a grid point.
STATUS_COLUMNS = ("ok", "failed", "timeout", "error", "skipped", "retried")

#: Column holding a combined series name when more than one grid axis varies.
SERIES_LABEL = "_series_label"
#: Column naming the single series of a summary that has no series axis.
SINGLE_SERIES = "_series"


def extra_series_columns(df: pd.DataFrame, x: str, color: str | None) -> list[str]:
    """Grid axes, besides `x`, `color` and `bench`, that vary across the data.

    Each distinct combination is its own curve; ignoring one (say `dtype`)
    would join int and float timings into one zigzag line with one fit.
    """
    taken = {x, color, "bench", SINGLE_SERIES, SERIES_LABEL, *STATUS_COLUMNS}
    return [
        c
        for c in grid_columns(df.columns)
        if c not in taken and df[c].nunique(dropna=False) > 1
    ]


def series_columns(df: pd.DataFrame, x: str, color: str | None) -> list[str]:
    """Columns that together identify one series (one curve, one fit)."""
    lead = [c for c in ("bench", color) if c is not None and c in df.columns]
    return lead + extra_series_columns(df, x, color)


def fit_frame(
    df: pd.DataFrame, x: str, color: str | None
) -> tuple[pd.DataFrame, list[str]]:
    """The frame and grouping a complexity fit uses — shared by chart and CLI.

    A summary with a single unnamed series still deserves a fit, so it gets a
    constant group rather than none.
    """
    by = series_columns(df, x, color)
    if not by:
        return df.assign(**{SINGLE_SERIES: "all"}), [SINGLE_SERIES]
    return df, by


def with_series_label(
    df: pd.DataFrame, x: str, color: str | None
) -> tuple[pd.DataFrame, str | None, str]:
    """Pick the field and legend title that tell the series apart by colour.

    With no other varying grid axis that is simply `color`; otherwise each
    combination gets its own name (``quick · dtype=int``) so every curve has a
    distinct colour and legend entry.
    """
    extras = extra_series_columns(df, x, color)
    lead = color if color is not None and color in df.columns else None
    if not extras:
        return df, lead, label(lead) if lead else ""

    def name(row: pd.Series) -> str:
        parts = [str(row[lead])] if lead else []
        parts.extend(f"{c}={row[c]}" for c in extras)
        return " · ".join(parts)

    labelled = df.assign(**{SERIES_LABEL: df.apply(name, axis=1)})
    title =" · ".join(([label(lead)] if lead else []) + extras)
    return labelled, SERIES_LABEL, title


def plottable_rows(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Rows that can be drawn, and a note for each kind that could not.

    A grid point where no trial succeeded has no value; drawing it would put a
    fake zero on the chart.  A log axis cannot place zero or a negative value,
    and Vega draws the whole axis without ticks if asked to.  Either omission
    is stated on the chart rather than made silently.
    """
    notes: list[str] = []
    if x not in df.columns or y not in df.columns:
        return df, notes
    xs = pd.to_numeric(df[x], errors="coerce")
    ys = pd.to_numeric(df[y], errors="coerce")
    missing = ys.isna() | xs.isna()
    if missing.any():
        notes.append(
            f"{int(missing.sum())} grid point(s) with no successful trial not shown"
        )
    keep = ~missing
    nonpositive = pd.Series(False, index=df.index)
    if log_x:
        nonpositive |= xs <= 0
    if log_y:
        nonpositive |= ys <= 0
    nonpositive &= keep
    if nonpositive.any():
        notes.append(
            f"{int(nonpositive.sum())} point(s) ≤ 0 omitted from the log scale"
        )
    return df[keep & ~nonpositive].copy(), notes


def metric_name(y: str) -> str:
    """What the y column measures, for titles: "Runtime", "Memory", or its label."""
    if y.startswith(("time_ms", "wall_ms", "reported_ms")):
        return "Runtime"
    if y.startswith("peak_rss"):
        return "Memory"
    return label(y)


def metric_unit(y: str) -> str:
    """The unit in the y column's label, e.g. "ms" from "Time – Median (ms)"."""
    title = label(y)
    if title.endswith(")") and "(" in title:
        return title[title.rindex("(") + 1 : -1]
    return ""


def x_title(x: str) -> str:
    """Short name of the x axis for chart titles."""
    return "Input Size" if x == "n" else label(x)


def message_chart(text: str, title: str | None = None) -> alt.Chart:
    """A chart that only states why there is nothing to draw."""
    chart = (
        alt.Chart(pd.DataFrame({"_message": [text]}))
        .mark_text(fontSize=14, color="#64748b")
        .encode(text="_message:N")
        .properties(width=640, height=120)
    )
    return chart.properties(title=title) if title else chart


def chart_title(text: str, subtitle: list[str]) -> alt.TitleParams:
    """Title with the fit formulas and omission notes as subtitle lines."""
    if not subtitle:
        return alt.TitleParams(text=text)
    return alt.TitleParams(
        text=text,
        subtitle=subtitle,
        subtitleFontSize=10,
        subtitleColor="#64748b",
        subtitlePadding=4,
    )


def resolve_y(df: pd.DataFrame, y: str, fallbacks: list[str]) -> str:
    if y in df.columns:
        return y
    for candidate in fallbacks:
        if candidate in df.columns:
            return candidate
    return y


def axis_scale(log_enabled: bool) -> alt.Scale:
    return alt.Scale(type="log") if log_enabled else alt.Scale(zero=True)


def shared_color_scale(df: pd.DataFrame, field: str | None) -> alt.Scale | None:
    """Fix each series' colour across every chart, so a series keeps its hue."""
    if field is None or field not in df.columns:
        return None
    values = sorted(df[field].dropna().unique().tolist())
    if not values:
        return None
    return alt.Scale(domain=values, range=PALETTE[: len(values)])


def legend_toggle(name: str, field: str | None, enabled: bool) -> Any | None:
    if not enabled or field is None:
        return None
    return alt.selection_point(name=name, fields=[field], bind="legend")


def legend_opacity(
    selection: Any | None,
    *,
    shown: float = 1.0,
    hidden: float = 0.08,
) -> Any:
    if selection is None:
        return alt.value(shown)
    return alt.condition(selection, alt.value(shown), alt.value(hidden))


def categorical_color(
    field: str | None,
    *,
    enabled: bool,
    title: str,
    fallback_color: str,
    scale: alt.Scale | None = None,
    legend: alt.Legend | None = None,
) -> Any:
    if not enabled or field is None:
        return alt.value(fallback_color)
    if scale is not None and legend is not None:
        return alt.Color(field, title=title, scale=scale, legend=legend)
    if scale is not None:
        return alt.Color(field, title=title, scale=scale)
    if legend is not None:
        return alt.Color(field, title=title, legend=legend)
    return alt.Color(field, title=title)


def build_tooltips(
    items: Sequence[tuple[str, str, str | None]],
) -> list[alt.Tooltip]:
    tooltips: list[alt.Tooltip] = []
    for field, title, fmt in items:
        if fmt is None:
            tooltips.append(alt.Tooltip(field, title=title))
        else:
            tooltips.append(alt.Tooltip(field, title=title, format=fmt))
    return tooltips


apply_theme()
