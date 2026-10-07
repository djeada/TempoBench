from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd

from ._common import (
    axis_scale,
    label,
    message_chart,
    multi_bench,
    number_axis,
    shared_color_scale,
    wants_log,
)


def plot_boxplot(
    runs_jsonl: Path,
    x: str | None = "impl",
    y: str | None = None,
    size: str | None = "n",
    bench: str | None = None,
    log_y: bool | None = None,
) -> alt.TopLevelMixin:
    """Create a boxplot of trial durations from raw JSONL runs.

    One box per grid point: trials are grouped by `size` (the input-size axis)
    and coloured by `x`, because pooling every input size into one box shows
    the spread of the sizes, not of the measurement.  With several benchmarks
    each gets its own row.

    With `y` unset the duration is chosen per grid point the same way
    `summarize` chooses it — self-reported when every successful trial of that
    point supplied one, otherwise wall clock — so the spread shown here is the
    spread of the numbers that were actually fitted.

    With `x` unset, each benchmark is a series.  `log_y`
    left as None picks a log axis when the durations span more than
    `LOG_SPAN`.
    """
    rows = []
    param_keys: list[str] = []
    with runs_jsonl.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") != "ok":
                continue
            if bench is not None and str(row.get("bench")) != bench:
                continue
            for key, value in (row.pop("params", None) or {}).items():
                row[key] = value
                if key not in param_keys:
                    param_keys.append(key)
            rows.append(row)

    title = "Spread of trial durations"
    if not rows:
        return message_chart("No successful trials to show", title)

    df = pd.DataFrame(rows)
    if x is None and "bench" in df.columns:
        x = "bench"
    if y is None:
        y = _pick_duration(df, [c for c in ["bench", *param_keys] if c in df.columns])

    if x not in df.columns or y not in df.columns:
        return message_chart(f"The runs have no '{x}' or '{y}' field to plot", title)

    df = df[pd.to_numeric(df[y], errors="coerce").notna()]
    if log_y is None:
        log_y = wants_log(df[y])
    elif log_y:
        df = df[pd.to_numeric(df[y], errors="coerce") > 0]
    by_size = size is not None and size in df.columns and size != x
    color = alt.Color(
        f"{x}:N", title=label(x), scale=shared_color_scale(df, x) or alt.Undefined
    )
    y_enc = alt.Y(f"{y}:Q", title=label(y), scale=axis_scale(log_y), axis=number_axis(log_y, df[y]))
    # Selections are not supported on boxplots; a legend toggle here makes
    # Vega fail with an unknown signal, so the box is left non-interactive.
    box = alt.Chart(df).mark_boxplot(
        extent="min-max",
        size=12 if by_size else 36,
        median={"color": "white", "strokeWidth": 2},
        ticks=False,
    )
    if by_size:
        chart = box.encode(
            x=alt.X(f"{size}:O", title=label(size), axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset(f"{x}:N"),
            y=y_enc,
            color=color,
        )
    else:
        chart = box.encode(
            x=alt.X(f"{x}:O", title=label(x), axis=alt.Axis(labelAngle=0)),
            y=y_enc,
            color=color,
        )
    chart = chart.properties(width=640, height=320)
    if multi_bench(df) and x != "bench":
        return (
            chart.facet(
                row=alt.Row(
                    "bench:N",
                    title=None,
                    header=alt.Header(labelAngle=0, labelOrient="top", labelAnchor="start"),
                )
            )
            .resolve_scale(y="independent")
            .properties(title=title)
        )
    return chart.properties(title=title)


def _pick_duration(df: pd.DataFrame, group_cols: list[str]) -> str:
    """Name of the per-trial duration to show, mirroring `summarize`.

    Returns ``reported_ms`` or ``wall_ms`` when every grid point agrees, and
    otherwise adds a ``time_ms`` column holding each point's own choice.
    """
    if "reported_ms" not in df.columns:
        return "wall_ms"
    reported = pd.to_numeric(df["reported_ms"], errors="coerce")
    if not group_cols:
        return "reported_ms" if reported.notna().all() else "wall_ms"
    complete = reported.notna().groupby(
        [df[c].astype(str) for c in group_cols]
    ).transform("all")
    if complete.all():
        return "reported_ms"
    if not complete.any() or "wall_ms" not in df.columns:
        return "wall_ms"
    df["time_ms"] = reported.where(complete, pd.to_numeric(df["wall_ms"], errors="coerce"))
    return "time_ms"
