from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd

from .distribution import plot_boxplot
from .runtime import plot_runtime
from .summary import plot_heatmap, plot_memory


def create_dashboard(
    summary_csv: Path,
    runs_jsonl: Path | None = None,
    x: str = "n",
    color: str = "impl",
    title: str = "TempoBench Dashboard",
    log_x: bool = False,
    log_y: bool = False,
) -> alt.TopLevelMixin:
    """Create a multi-chart dashboard."""
    df = pd.read_csv(summary_csv)
    charts: list[Any] = [
        plot_runtime(
            summary_csv,
            x=x,
            color=color,
            show_fit=True,
            log_x=log_x,
            log_y=log_y,
        )
    ]

    if "peak_rss_mb_median" in df.columns or "peak_rss_mb_mean" in df.columns:
        charts.append(plot_memory(summary_csv, x=x, color=color, log_x=log_x, log_y=log_y))

    if color in df.columns and x in df.columns:
        charts.append(plot_heatmap(summary_csv, x=x, y=color))

    if runs_jsonl and runs_jsonl.exists():
        charts.append(plot_boxplot(runs_jsonl, x=color))

    if len(charts) == 1:
        return charts[0]

    return (
        alt.vconcat(*charts)
        .properties(
            title=alt.TitleParams(
                text=title, fontSize=20, fontWeight=700, anchor="middle", dy=-10
            )
        )
        .configure_concat(spacing=40)
    )
