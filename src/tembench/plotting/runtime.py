from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd

from ..complexity import fit_models, predict_series
from ._common import (
    PALETTE,
    axis_scale,
    build_tooltips,
    categorical_color,
    label,
    legend_opacity,
    legend_toggle,
    resolve_y,
    shared_color_scale,
)


def plot_runtime(
    summary_csv: Path,
    x: str = "n",
    y: str = "wall_ms_median",
    color: str = "impl",
    bench: str | None = None,
    show_fit: bool = True,
    by: list[str] | None = None,
    complexity_strategy: str = "heuristic",
    log_x: bool = False,
    log_y: bool = False,
) -> alt.Chart:
    df = pd.read_csv(summary_csv)
    if bench is not None:
        if "bench" not in df.columns:
            raise ValueError(
                "Summary does not contain a 'bench' column; cannot filter by --bench."
            )
        df = df[df["bench"] == bench].copy()
        if df.empty:
            raise ValueError(f"No rows found for bench='{bench}'.")

    has_multi_bench = "bench" in df.columns and df["bench"].nunique(dropna=False) > 1
    y_col = resolve_y(df, y, ["wall_ms_median", "wall_ms_mean"])

    x_scale = axis_scale(log_x)
    y_scale = axis_scale(log_y)
    x_enc = alt.X(x, title=label(x), scale=x_scale, axis=alt.Axis(format="~s"))
    y_enc = alt.Y(y_col, title=label(y_col), scale=y_scale)

    color_enabled = color in df.columns
    color_scale = shared_color_scale(df, color)
    legend_sel = legend_toggle("rt_legend", color, color_enabled)
    color_enc = categorical_color(
        color,
        enabled=color_enabled,
        title=label(color),
        fallback_color=PALETTE[0],
        scale=color_scale,
        legend=alt.Legend(
            title="Implementation  (click to toggle)",
            symbolType="cross",
            symbolSize=150,
            symbolStrokeWidth=2.5,
        )
        if color_enabled
        else None,
    )

    nearest = alt.selection_point(
        name="rt_nearest",
        nearest=True,
        on="pointerover",
        fields=[x],
        empty=False,
    )

    tooltips = build_tooltips(
        [
            (x, label(x), ","),
            (y_col, label(y_col), ".2f"),
        ]
    )
    tooltips.append(
        alt.Tooltip(color, title=label(color)) if color_enabled else alt.Tooltip(y_col)
    )
    if "bench" in df.columns:
        tooltips.insert(0, alt.Tooltip("bench", title="Benchmark"))

    base = (
        alt.Chart(df)
        .mark_point(shape="cross", size=200, strokeWidth=3, filled=False)
        .encode(
            x=x_enc,
            y=y_enc,
            color=color_enc,
            opacity=legend_opacity(legend_sel),
            tooltip=tooltips,
        )
        .properties(width=640, height=400, title="Runtime vs Input Size")
    )
    if legend_sel is not None:
        base = base.add_params(legend_sel)

    voronoi = (
        alt.Chart(df)
        .mark_point(size=0, opacity=0)
        .encode(x=x_enc, y=y_enc, tooltip=tooltips)
        .add_params(nearest)
    )

    rule = (
        alt.Chart(df)
        .mark_rule(color="#94a3b8", strokeWidth=1, strokeDash=[4, 3])
        .encode(x=x_enc, opacity=alt.condition(nearest, alt.value(0.6), alt.value(0)))
        .transform_filter(nearest)
    )

    highlight_dots = (
        alt.Chart(df)
        .mark_point(
            shape="circle", size=100, filled=True, strokeWidth=2, stroke="white"
        )
        .encode(
            x=x_enc,
            y=y_enc,
            color=categorical_color(
                color,
                enabled=color_enabled,
                title=label(color),
                fallback_color=PALETTE[0],
                scale=color_scale,
            ),
            opacity=legend_opacity(legend_sel, hidden=0.0),
        )
        .transform_filter(nearest)
    )

    if not show_fit:
        chart = base + voronoi + rule + highlight_dots
        if has_multi_bench:
            return chart.facet(
                row=alt.Row("bench:N", title="Benchmark"),
            ).resolve_scale(y="independent")
        return chart

    by_cols = by or [c for c in ["bench", color] if c in df.columns]
    if not by_cols:
        by_cols = [c for c in [color] if c in df.columns]
    if not by_cols or df.empty or x not in df.columns or y_col not in df.columns:
        return base + voronoi + rule + highlight_dots

    fits = fit_models(df, x_col=x, y_col=y_col, by=by_cols, strategy=complexity_strategy)
    if fits.empty:
        return base + voronoi + rule + highlight_dots
    preds = predict_series(df, fits, x_col=x, by=by_cols)
    if preds.empty:
        return base + voronoi + rule + highlight_dots

    label_field = "display_model" if "display_model" in preds.columns else "model"
    fit_tooltips = build_tooltips(
        [
            (label_field, "Complexity", None),
            ("formula", "Bound", None),
            (x, label(x), ","),
            ("yhat", "Upper Bound (ms)", ".2f"),
        ]
    )
    if "empirical_exponent" in preds.columns:
        fit_tooltips.extend(
            build_tooltips(
                [
                    ("empirical_exponent", "Exponent", ".2f"),
                    ("exponent_ci_low", "Exp CI Low", ".2f"),
                    ("exponent_ci_high", "Exp CI High", ".2f"),
                ]
            )
        )

    fit_layer = (
        alt.Chart(preds)
        .mark_line(strokeDash=[8, 4], strokeWidth=2)
        .encode(
            x=alt.X(x, title=label(x), scale=x_scale, axis=alt.Axis(format="~s")),
            y=alt.Y("yhat", title=label(y_col), scale=y_scale),
            color=categorical_color(
                color,
                enabled=color in preds.columns,
                title=label(color),
                fallback_color=PALETTE[0],
                scale=color_scale,
            ),
            opacity=legend_opacity(legend_sel, shown=0.7, hidden=0.05),
            detail=by_cols,
            tooltip=fit_tooltips,
        )
    )

    preds_sorted = preds.sort_values(x)
    label_rows = []
    label_positions = [0.45, 0.65, 0.80, 0.55, 0.70, 0.85]
    for i, (_key, group_df) in enumerate(preds_sorted.groupby(by_cols, dropna=False)):
        pos = label_positions[i % len(label_positions)]
        idx = int(len(group_df) * pos)
        idx = max(0, min(idx, len(group_df) - 1))
        row = group_df.iloc[idx].copy()
        impl_name = row.get(color, "") if color in preds.columns else ""
        display_model = row.get("display_model", row["model"])
        row["_label"] = f"{impl_name}: {display_model}" if impl_name else display_model
        label_rows.append(row)
    label_points = pd.DataFrame(label_rows).reset_index(drop=True)

    label_points = label_points.sort_values("yhat", ascending=False).reset_index(
        drop=True
    )
    label_points["_dy"] = [-14 - i * 16 for i in range(len(label_points))]

    label_layer = (
        alt.Chart(label_points)
        .mark_text(align="left", dx=6, fontSize=12, fontWeight=700)
        .encode(
            x=x,
            y=alt.Y("yhat:Q", scale=y_scale),
            text="_label:N",
            color=categorical_color(
                color,
                enabled=color in label_points.columns,
                title=label(color),
                fallback_color=PALETTE[0],
                scale=color_scale,
            ),
            opacity=legend_opacity(legend_sel, hidden=0.05),
        )
    )

    layered = base + voronoi + rule + highlight_dots + fit_layer + label_layer

    if has_multi_bench:
        return layered.facet(
            row=alt.Row("bench:N", title="Benchmark"),
        ).resolve_scale(y="independent")

    fit_info_parts = []
    if color in fits.columns:
        for _, row in fits.iterrows():
            fit_info_parts.append(f"{row[color]}: {row['formula']}")
    else:
        for _, row in fits.iterrows():
            fit_info_parts.append(row["formula"])

    return layered.properties(
        title=alt.TitleParams(
            text="Runtime vs Input Size",
            subtitle=" │ ".join(fit_info_parts),
            subtitleFontSize=10,
            subtitleColor="#64748b",
            subtitlePadding=4,
        )
    )
