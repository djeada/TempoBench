from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd

from .complexity import fit_models, predict_series

def plot_runtime(
    summary_csv: Path,
    x: str = "n",
    y: str = "wall_ms_median",
    color: str = "impl",
    show_fit: bool = True,
    by: list[str] | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> alt.Chart:
    df = pd.read_csv(summary_csv)
    # Support flexible column names based on summarize
    y_col = y
    if y_col not in df.columns:
        # try common alt names
        for cand in ["wall_ms_median", "wall_ms_mean", "wall_ms_med"]:
            if cand in df.columns:
                y_col = cand
                break
    x_enc = alt.X(x, title=x, scale=alt.Scale(type="log")) if log_x else alt.X(x, title=x)
    y_enc = alt.Y(y_col, title=y_col, scale=alt.Scale(type="log")) if log_y else alt.Y(y_col, title=y_col)
    base = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=x_enc,
            y=y_enc,
            color=color if color in df.columns else alt.value("steelblue"),
            tooltip=list(df.columns),
        )
        .properties(width=600, height=400)
    )
    if not show_fit:
        return base
    # determine grouping for model fitting
    by_cols = by or [c for c in [color] if c in df.columns]
    if not by_cols:
        by_cols = []
    if df.empty or x not in df.columns or y_col not in df.columns:
        return base
    fits = fit_models(df, x_col=x, y_col=y_col, by=by_cols or [x])
    if fits.empty:
        return base
    preds = predict_series(df, fits, x_col=x, by=by_cols) if by_cols else pd.DataFrame()
    if preds.empty:
        return base
    fit_layer = (
        alt.Chart(preds)
        .mark_line(strokeDash=[4, 4])
        .encode(
            x=x_enc,
            y=alt.Y("yhat", title=y_col, scale=alt.Scale(type="log") if log_y else alt.Undefined),
            color=color if color in preds.columns else alt.value("steelblue"),
            detail=by_cols,
            tooltip=by_cols + ["model", x, "yhat"],
        )
    )
    # pick the largest x per group for label placement without deprecated GroupBy.apply behavior
    if by_cols:
        label_points = (
            preds.sort_values(x)
            .groupby(by_cols, dropna=False, group_keys=False)
            .tail(1)
            .reset_index(drop=True)
        )
    else:
        label_points = preds
    label_layer = (
        alt.Chart(label_points)
        .mark_text(align="left", dx=5)
        .encode(
            x=x,
            y="yhat",
            color=color if color in preds.columns else alt.value("steelblue"),
            text="model",
            detail=by_cols,
        )
    )
    return base + fit_layer + label_layer
