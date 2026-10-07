from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, cast

import altair as alt
import pandas as pd

from ..reporting.formatting import column_label
from ..summarize import grid_columns

#: Categorical palette, in fixed order: slot k always goes to the k-th series
#: (sorted by name), so a series keeps its colour across every chart.  The order
#: keeps neighbouring slots apart under colour-vision deficiencies.
PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
#: The same hues stepped for a dark surface.
PALETTE_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

# Ink and rule colours for each theme.  Charts are drawn on a transparent
# background so they take the colour of whatever page section holds them.
_LIGHT = {"text": "#1c1c1a", "muted": "#5f5e5a", "grid": "#ecebe7", "domain": "#c9c8c2"}
_DARK = {"text": "#f2f1ec", "muted": "#b4b3aa", "grid": "#2f2f2c", "domain": "#55544f"}

#: Sequential scheme for heatmap cells — one hue, light to dark.
SEQUENTIAL_SCHEME: Final = "blues"


def _theme_colors(ink: dict[str, str], palette: list[str]) -> dict[str, Any]:
    """The colour-bearing part of the chart config for one theme."""
    return {
        "title": {"color": ink["text"], "subtitleColor": ink["muted"]},
        "axis": {
            "gridColor": ink["grid"],
            "domainColor": ink["domain"],
            "tickColor": ink["domain"],
            "labelColor": ink["muted"],
            "titleColor": ink["muted"],
        },
        "legend": {"labelColor": ink["muted"], "titleColor": ink["muted"]},
        "header": {"labelColor": ink["text"], "titleColor": ink["muted"]},
        "text": {"color": ink["muted"]},
        "rule": {"color": ink["domain"]},
        "range": {"category": palette},
    }


#: Overrides the HTML pages merge into every chart's config in dark mode.
DARK_CONFIG = _theme_colors(_DARK, PALETTE_DARK)


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def apply_theme() -> None:
    """Register and enable a clean TempoBench theme."""
    layout = {
        "background": "transparent",
        "font": FONT,
        "padding": 8,
        "title": {
            "fontSize": 15,
            "fontWeight": 600,
            "anchor": "start",
            "offset": 14,
            "subtitleFontSize": 11,
            "subtitlePadding": 4,
        },
        "axis": {
            "labelFontSize": 11,
            "titleFontSize": 12,
            "titleFontWeight": 500,
            "titlePadding": 10,
            "labelPadding": 4,
            "tickSize": 4,
            "labelFlush": False,
        },
        "legend": {
            "labelFontSize": 12,
            "titleFontSize": 12,
            "titleFontWeight": 500,
            "symbolSize": 120,
            "orient": "bottom",
            "direction": "horizontal",
            "padding": 8,
            "columnPadding": 16,
            # Wrap rather than run off the side of a phone screen.
            "columns": {"expr": "width < 480 ? 2 : 4"},
            "titleLimit": 480,
            "labelLimit": 320,
        },
        "header": {"labelFontSize": 13, "labelFontWeight": 600, "titleFontSize": 12},
        "view": {"stroke": None, "continuousWidth": 640, "continuousHeight": 360},
        "line": {"strokeWidth": 2},
        "point": {"size": 64, "filled": True},
    }
    theme_config = {"config": _merge(layout, _theme_colors(_LIGHT, PALETTE))}

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
    return column_label(col)


#: Tooltip format: four significant figures, so a 0.0019 ms timing is not
#: rounded to zero and a 17 000 ms one keeps its magnitude.
NUMBER_FORMAT = ",.4~r"
#: Axis tick format: SI prefixes for big counts, plain decimals otherwise
#: (``~s`` alone labels 0.5 as "500m" and 0 on a million-wide axis as "0M").
AXIS_FORMAT_EXPR = (
    "abs(datum.value) >= 10000 ? format(datum.value, '~s') : format(datum.value, ',~g')"
)

#: Per-status trial counts in a summary.  "retried" counts superseded attempts
#: of a trial that was run again; none of these identify a grid point.
STATUS_COLUMNS = ("ok", "failed", "timeout", "error", "skipped", "retried")

#: Column holding a combined series name when more than one grid axis varies.
SERIES_LABEL = "_series_label"
#: Column naming the single series of a summary that has no series axis.
SINGLE_SERIES = "_series"

#: Positive data spanning more than this ratio gets a log axis by default:
#: on a linear one everything below the top decade is squashed onto the floor.
LOG_SPAN = 30.0


def read_summary(summary: Path | pd.DataFrame, bench: str | None = None) -> pd.DataFrame:
    """The summary as a frame, narrowed to one benchmark when `bench` is set."""
    df = summary.copy() if isinstance(summary, pd.DataFrame) else pd.read_csv(summary)
    if bench is None:
        return df
    if "bench" not in df.columns:
        raise ValueError(
            "Summary does not contain a 'bench' column; cannot filter by --bench."
        )
    df = df[df["bench"].astype(str) == bench].copy()
    if df.empty:
        raise ValueError(f"No rows found for bench='{bench}'.")
    return df


def multi_bench(df: pd.DataFrame) -> bool:
    return "bench" in df.columns and df["bench"].nunique(dropna=False) > 1


def default_series(df: pd.DataFrame, color: str | None) -> str | None:
    """The series column, falling back to `bench` when nothing else varies.

    Several benchmarks with no series axis of their own — the same algorithm in
    C++, Rust and Python, say — are what the chart is there to compare, so they
    share one panel, one colour each, rather than a panel each.
    """
    if color is None and multi_bench(df):
        return "bench"
    return color


def wants_log(values: Sequence[float] | pd.Series) -> bool:
    """Whether data spanning `values` reads better on a log axis."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty or (numeric <= 0).any():
        return False
    return float(numeric.max() / numeric.min()) > LOG_SPAN


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
    lead = [c for c in dict.fromkeys(("bench", color)) if c is not None and c in df.columns]
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
    title = " · ".join(([label(lead)] if lead else []) + extras)
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
    """The unit in the y column's label, e.g. "ms" from "Median time (ms)"."""
    title = label(y)
    if title.endswith(")") and "(" in title:
        return title[title.rindex("(") + 1 : -1]
    return ""


def x_title(x: str) -> str:
    """Short name of the x axis for chart titles."""
    return "input size" if x == "n" else label(x)


def message_chart(text: str, title: str | None = None) -> alt.Chart:
    """A chart that only states why there is nothing to draw."""
    chart = (
        alt.Chart(pd.DataFrame({"_message": [text]}))
        .mark_text(fontSize=14)
        .encode(text="_message:N")
        .properties(width=640, height=120)
    )
    return chart.properties(title=title) if title else chart


def chart_title(text: str, subtitle: list[str]) -> alt.TitleParams:
    """Title with omission notes as subtitle lines."""
    if not subtitle:
        return alt.TitleParams(text=text)
    return alt.TitleParams(text=text, subtitle=subtitle)


def resolve_y(df: pd.DataFrame, y: str, fallbacks: Sequence[str]) -> str:
    if y in df.columns:
        return y
    for candidate in fallbacks:
        if candidate in df.columns:
            return candidate
    return y


def axis_scale(log_enabled: bool) -> alt.Scale:
    # A "nice" log domain rounds out to whole decades, which can leave most
    # of the plot empty when the data covers only part of one.
    return alt.Scale(type="log", nice=False) if log_enabled else alt.Scale(zero=True)


def log_ticks(values: Sequence[float] | pd.Series) -> list[float]:
    """Tick values for a log axis over `values`: 1-2-5 steps, or whole decades.

    Vega's own log ticks label every integer multiple, so a three-decade axis
    gets thirty crowded labels.
    """
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    positive = numeric[numeric > 0]
    if positive.empty:
        return []
    low = math.floor(math.log10(float(positive.min())))
    high = math.ceil(math.log10(float(positive.max())))
    steps = (1, 2, 5) if high - low <= 3 else (1,)
    return [m * 10.0**e for e in range(low, high + 1) for m in steps]


def number_axis(log_enabled: bool = False, values: Sequence[float] | pd.Series = ()) -> alt.Axis:
    """A quantitative axis whose labels never read "0M" or "500m".

    On a log axis, ticks are placed by `log_ticks` over `values` — the data the
    axis spans.
    """
    ticks = log_ticks(values) if log_enabled else []
    if ticks:
        return alt.Axis(labelExpr=AXIS_FORMAT_EXPR, values=ticks, labelOverlap="greedy", labelSeparation=6)
    return alt.Axis(labelExpr=AXIS_FORMAT_EXPR, labelOverlap="greedy", labelSeparation=6)


def shared_color_scale(df: pd.DataFrame, field: str | None) -> alt.Scale | None:
    """Fix each series' colour across every chart, so a series keeps its hue.

    Only the domain is fixed; the colours come from the theme's categorical
    range, which the HTML pages swap for its dark-mode steps.  Pass the whole
    summary, not one benchmark's rows, so a chart narrowed by `--bench` colours
    its series as the full chart does.
    """
    if field is None or field not in df.columns:
        return None
    values = sorted(df[field].dropna().unique().tolist(), key=str)
    if not values:
        return None
    return alt.Scale(domain=values)


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
    kwargs: dict[str, Any] = {"title": title}
    if scale is not None:
        kwargs["scale"] = scale
    if legend is not None:
        kwargs["legend"] = legend
    return alt.Color(f"{field}:N", **kwargs)


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
