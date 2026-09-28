from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd

from ._common import PALETTE, label


def _message(text: str) -> alt.Chart:
    return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value(text))


def plot_boxplot(
    runs_jsonl: Path,
    x: str | None = "impl",
    y: str | None = None,
    size: str | None = "n",
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
            for key, value in (row.pop("params", None) or {}).items():
                row[key] = value
                if key not in param_keys:
                    param_keys.append(key)
            rows.append(row)

    if not rows:
        return _message("No successful runs for boxplot")

    df = pd.DataFrame(rows)
    if y is None:
        y = _pick_duration(df, [c for c in ["bench", *param_keys] if c in df.columns])

    if x not in df.columns or y not in df.columns:
        return _message("Required columns not found")

    df = df[pd.to_numeric(df[y], errors="coerce").notna()]
    by_size = size is not None and size in df.columns and size != x
    color = alt.Color(f"{x}:N", title=label(x), scale=alt.Scale(range=PALETTE))
    y_enc = alt.Y(f"{y}:Q", title=label(y), scale=alt.Scale(zero=True))
    # Selections are not supported on boxplots; a legend toggle here makes
    # Vega fail with an unknown signal, so the box is left non-interactive.
    box = alt.Chart(df).mark_boxplot(
        extent="min-max", size=14 if by_size else 40, median={"color": "white", "strokeWidth": 2}
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
    chart = chart.properties(width=640, height=360)
    title = f"Distribution: {label(y)}"
    if "bench" in df.columns and df["bench"].nunique() > 1:
        return (
            chart.facet(row=alt.Row("bench:N", title="Benchmark"))
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
