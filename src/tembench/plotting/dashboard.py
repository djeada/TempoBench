from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import altair as alt
import pandas as pd

from ._common import default_series, read_summary
from .distribution import plot_boxplot
from .runtime import plot_runtime
from .summary import MEMORY_COLUMNS, plot_heatmap, plot_memory


def dashboard_charts(
    summary: Path | pd.DataFrame,
    runs_jsonl: Path | None = None,
    x: str = "n",
    color: str | None = "impl",
    bench: str | None = None,
    complexity_strategy: str = "heuristic",
    log_x: bool | None = None,
    log_y: bool | None = None,
) -> list[alt.TopLevelMixin]:
    """The dashboard's charts, in order: runtime, memory, heatmap, spread.

    Kept as separate charts so a page can lay each out at its own width.
    """
    df = read_summary(summary, bench)
    charts: list[alt.TopLevelMixin] = [
        plot_runtime(
            df,
            x=x,
            color=color,
            show_fit=True,
            complexity_strategy=complexity_strategy,
            log_x=log_x,
            log_y=log_y,
        )
    ]

    if any(c in df.columns for c in MEMORY_COLUMNS):
        charts.append(plot_memory(df, x=x, color=color, log_x=log_x, log_y=log_y))

    series = default_series(df, color)
    if series in df.columns and x in df.columns:
        charts.append(plot_heatmap(df, x=x, y=series))

    if runs_jsonl and runs_jsonl.exists():
        charts.append(plot_boxplot(runs_jsonl, x=color, size=x, bench=bench, log_y=log_y))
    return charts


def create_dashboard(
    summary: Path | pd.DataFrame,
    runs_jsonl: Path | None = None,
    x: str = "n",
    color: str | None = "impl",
    title: str = "TempoBench Dashboard",
    log_x: bool | None = None,
    log_y: bool | None = None,
) -> alt.TopLevelMixin:
    """The dashboard as a single Vega-Lite spec, its charts stacked vertically."""
    charts = dashboard_charts(
        summary, runs_jsonl, x=x, color=color, log_x=log_x, log_y=log_y
    )
    if len(charts) == 1:
        return charts[0]
    return (
        alt.vconcat(*cast(list[Any], charts))
        .properties(title=alt.TitleParams(text=title, fontSize=20, anchor="start"))
        .configure_concat(spacing=40)
    )
