from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import altair as alt
import pandas as pd

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
    "wall_ms_median": "Wall Time – Median (ms)",
    "wall_ms_mean": "Wall Time – Mean (ms)",
    "wall_ms_p10": "Wall Time – P10 (ms)",
    "wall_ms_p90": "Wall Time – P90 (ms)",
    "peak_rss_mb_median": "Peak RSS – Median (MB)",
    "peak_rss_mb_mean": "Peak RSS – Mean (MB)",
    "n": "Input Size (n)",
    "impl": "Implementation",
    "wall_ms": "Wall Time (ms)",
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


def label(col: str) -> str:
    return NICE_LABELS.get(col, col)


def resolve_y(df: pd.DataFrame, y: str, fallbacks: list[str]) -> str:
    if y in df.columns:
        return y
    for candidate in fallbacks:
        if candidate in df.columns:
            return candidate
    return y


def axis_scale(log_enabled: bool) -> alt.Scale:
    return alt.Scale(type="log") if log_enabled else alt.Scale(zero=True)


def shared_color_scale(df: pd.DataFrame, field: str) -> alt.Scale | None:
    if field not in df.columns:
        return None
    values = sorted(df[field].dropna().unique().tolist())
    if not values:
        return None
    return alt.Scale(domain=values, range=PALETTE[: len(values)])


def legend_toggle(name: str, field: str, enabled: bool) -> Any | None:
    if not enabled:
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
    field: str,
    *,
    enabled: bool,
    title: str,
    fallback_color: str,
    scale: alt.Scale | None = None,
    legend: alt.Legend | None = None,
) -> Any:
    if not enabled:
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
