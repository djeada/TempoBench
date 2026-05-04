from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd

from ._common import PALETTE, label


def plot_boxplot(
    runs_jsonl: Path,
    x: str = "impl",
    y: str = "wall_ms",
) -> alt.Chart:
    """Create a boxplot from raw JSONL runs."""
    rows = []
    with runs_jsonl.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("status") == "ok":
                    if "params" in row:
                        for key, value in row["params"].items():
                            row[key] = value
                    rows.append(row)
            except json.JSONDecodeError:
                continue

    if not rows:
        return (
            alt.Chart(pd.DataFrame())
            .mark_text()
            .encode(text=alt.value("No successful runs for boxplot"))
        )

    df = pd.DataFrame(rows)
    if x not in df.columns or y not in df.columns:
        return (
            alt.Chart(pd.DataFrame())
            .mark_text()
            .encode(text=alt.value("Required columns not found"))
        )

    highlight = alt.selection_point(name="box_highlight", fields=[x], bind="legend")

    return (
        alt.Chart(df)
        .mark_boxplot(
            extent="min-max", size=40, median={"color": "white", "strokeWidth": 2}
        )
        .encode(
            x=alt.X(f"{x}:O", title=label(x), axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y}:Q", title=label(y), scale=alt.Scale(zero=True)),
            color=alt.Color(
                f"{x}:N",
                title=label(x) + "  (click to toggle)",
                scale=alt.Scale(range=PALETTE),
            ),
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.12)),
        )
        .add_params(highlight)
        .properties(width=640, height=360, title=f"Distribution: {label(y)}")
    )
