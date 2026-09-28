from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd

from ..complexity import fit_models, predict_series
from ..summarize import (
    TIME_COLUMN_PREFERENCE,
    count_column_for,
    spread_columns_for,
)
from ._common import (
    NUMBER_FORMAT,
    PALETTE,
    axis_scale,
    build_tooltips,
    categorical_color,
    chart_title,
    fit_frame,
    label,
    legend_opacity,
    legend_toggle,
    message_chart,
    metric_name,
    metric_unit,
    plottable_rows,
    resolve_y,
    shared_color_scale,
    with_series_label,
    x_title,
)


def _formula_text(formula: str, y_col: str) -> str:
    """Bound formula named for what it bounds: T(n) is time, not memory."""
    if metric_name(y_col) == "Runtime":
        return formula
    return formula.replace("T(n)", "f(n)", 1)


def _subtitle_lines(parts: list[str], per_line: int = 3) -> list[str]:
    return [" │ ".join(parts[i : i + per_line]) for i in range(0, len(parts), per_line)]


def plot_runtime(
    summary_csv: Path,
    x: str = "n",
    y: str = "time_ms_median",
    color: str | None = "impl",
    bench: str | None = None,
    show_fit: bool = True,
    by: list[str] | None = None,
    complexity_strategy: str = "heuristic",
    log_x: bool = False,
    log_y: bool = False,
) -> alt.TopLevelMixin:
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
    y_col = resolve_y(df, y, list(TIME_COLUMN_PREFERENCE))
    metric = metric_name(y_col)
    unit = metric_unit(y_col)
    title = f"{metric} vs {x_title(x)}"

    # The fit sees every measured point; the chart only what its axes can show.
    fit_df, by_cols = fit_frame(df, x, color)
    if by:
        by_cols = by
    df, series, series_title = with_series_label(df, x, color)
    shown, notes = plottable_rows(df, x, y_col, log_x=log_x, log_y=log_y)
    if shown.empty:
        return message_chart("; ".join(["No measured points to plot", *notes]), title)

    x_scale = axis_scale(log_x)
    y_scale = axis_scale(log_y)
    x_enc = alt.X(x, title=label(x), scale=x_scale, axis=alt.Axis(format="~s"))
    y_enc = alt.Y(y_col, title=label(y_col), scale=y_scale)

    color_enabled = series is not None
    color_scale = shared_color_scale(df, series)
    legend_sel = legend_toggle("rt_legend", series, color_enabled)
    # Every layer carries the same colour encoding: layers whose legend titles
    # differ get merged into one legend titled "A, A (click to toggle), …".
    color_enc = categorical_color(
        series,
        enabled=color_enabled,
        title=f"{series_title}  (click to toggle)",
        fallback_color=PALETTE[0],
        scale=color_scale,
        legend=alt.Legend(
            symbolType="cross", symbolSize=150, symbolStrokeWidth=2.5, titleLimit=480
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
    hover = nearest & legend_sel if legend_sel is not None else nearest

    tooltips = build_tooltips(
        [
            (x, label(x), ","),
            (y_col, label(y_col), NUMBER_FORMAT),
        ]
    )
    if series is not None:
        tooltips.append(alt.Tooltip(series, title=series_title))
    if "bench" in df.columns:
        tooltips.insert(0, alt.Tooltip("bench", title="Benchmark"))

    # Both selections live on this one view.  Spread over two layers, Altair
    # lists each selection under both views once the chart sits in a
    # dashboard's vconcat, and Vega rejects the duplicated signal.
    base = (
        alt.Chart(shown)
        .mark_point(shape="cross", size=200, strokeWidth=3, filled=False)
        .encode(
            x=x_enc,
            y=y_enc,
            color=color_enc,
            opacity=legend_opacity(legend_sel),
            tooltip=tooltips,
        )
        .properties(width=640, height=400)
        .add_params(*[p for p in (legend_sel, nearest) if p is not None])
    )

    # Hover marks stay in the data and are hidden by opacity: a layer filtered
    # down to the (initially empty) selection has no extent to scale from.
    rule = (
        alt.Chart(shown)
        .mark_rule(color="#94a3b8", strokeWidth=1, strokeDash=[4, 3])
        .encode(x=x_enc, opacity=alt.condition(nearest, alt.value(0.6), alt.value(0)))
    )

    highlight_dots = (
        alt.Chart(shown)
        .mark_point(
            shape="circle", size=100, filled=True, strokeWidth=2, stroke="white"
        )
        .encode(
            x=x_enc,
            y=y_enc,
            color=color_enc,
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.0)),
        )
    )

    layered = base + rule + highlight_dots
    fit_parts: list[str] = []

    if show_fit:
        fits = fit_models(
            fit_df,
            x_col=x,
            y_col=y_col,
            by=by_cols,
            strategy=complexity_strategy,
            count_col=count_column_for(y_col),
            spread_cols=spread_columns_for(y_col),
        )
        if not fits.empty:
            layered, fit_parts = _add_fit_layers(
                layered,
                shown,
                fits,
                x=x,
                y_col=y_col,
                by_cols=by_cols,
                series=series,
                color_enc=color_enc,
                legend_sel=legend_sel,
                x_enc=x_enc,
                y_scale=y_scale,
                unit=unit,
                log_x=log_x,
                log_y=log_y,
            )

    title_params = chart_title(title, _subtitle_lines(fit_parts) + notes)
    if has_multi_bench:
        return (
            layered.facet(row=alt.Row("bench:N", title="Benchmark"))
            .resolve_scale(y="independent")
            .properties(title=title_params)
        )
    return layered.properties(title=title_params)


def _add_fit_layers(
    layered: alt.LayerChart,
    shown: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    x: str,
    y_col: str,
    by_cols: list[str],
    series: str | None,
    color_enc,
    legend_sel,
    x_enc: alt.X,
    y_scale: alt.Scale,
    unit: str,
    log_x: bool,
    log_y: bool,
) -> tuple[alt.LayerChart, list[str]]:
    """Overlay each series' upper-bound curve and its class label."""
    # Predictions span the drawn points only, so a failed largest size does
    # not stretch the curve into a region nothing measured.
    pred_src = shown
    missing_by = [c for c in by_cols if c not in pred_src.columns]
    if missing_by:  # the constant single-series group
        pred_src = pred_src.assign(**{c: "all" for c in missing_by})
    preds = predict_series(pred_src, fits, x_col=x, by=by_cols)
    if preds.empty:
        return layered, []
    if series is not None and series not in preds.columns:
        # The combined series name is derived, so carry it over from the data.
        names = pred_src[by_cols + [series]].drop_duplicates(by_cols)
        preds = preds.merge(names, on=by_cols, how="left")
    if log_x:
        preds = preds[preds[x] > 0]
    if log_y:
        preds = preds[preds["yhat"] > 0]
    if preds.empty:
        return layered, []
    preds = preds.assign(formula=preds["formula"].map(lambda f: _formula_text(f, y_col)))

    label_field = "display_model" if "display_model" in preds.columns else "model"
    bound_title = f"Upper Bound ({unit})" if unit else "Upper Bound"
    fit_tooltips = build_tooltips(
        [
            (label_field, "Complexity", None),
            ("formula", "Bound", None),
            (x, label(x), ","),
            ("yhat", bound_title, NUMBER_FORMAT),
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
    if "confidence" in preds.columns:
        fit_tooltips.extend(
            build_tooltips(
                [
                    ("confidence", "Confidence", None),
                    ("confidence_notes", "Caveats", None),
                ]
            )
        )

    fit_layer = (
        alt.Chart(preds)
        .mark_line(strokeDash=[8, 4], strokeWidth=2)
        .encode(
            x=x_enc,
            y=alt.Y("yhat", title=label(y_col), scale=y_scale),
            color=color_enc,
            opacity=legend_opacity(legend_sel, shown=0.7, hidden=0.05),
            detail=by_cols,
            tooltip=fit_tooltips,
        )
    )

    preds_sorted = preds.sort_values(x)
    label_rows = []
    fit_parts = []
    # Labels sit at staggered fractions of each curve so neighbours do not overlap.
    label_positions = [0.45, 0.65, 0.80, 0.55, 0.70, 0.85]
    for i, (_key, group_df) in enumerate(preds_sorted.groupby(by_cols, dropna=False)):
        pos = label_positions[i % len(label_positions)]
        idx = max(0, min(int(len(group_df) * pos), len(group_df) - 1))
        row = group_df.iloc[idx].copy()
        name = str(row[series]) if series is not None else ""
        display_model = row.get("display_model", row["model"])
        # A class the data cannot support must not read like one it can.
        if row.get("confidence") == "low":
            display_model = f"{display_model} (low confidence)"
        row["_label"] = f"{name}: {display_model}" if name else display_model
        label_rows.append(row)
        prefix = f"{row['bench']} / " if "bench" in by_cols and shown["bench"].nunique() > 1 else ""
        fit_parts.append(
            f"{prefix}{name}: {row['formula']}" if name else f"{prefix}{row['formula']}"
        )
    label_points = pd.DataFrame(label_rows).reset_index(drop=True)

    label_layer = (
        alt.Chart(label_points)
        .mark_text(align="left", dx=6, dy=-10, fontSize=12, fontWeight=700)
        .encode(
            x=x_enc,
            y=alt.Y("yhat:Q", title=label(y_col), scale=y_scale),
            text="_label:N",
            color=color_enc,
            opacity=legend_opacity(legend_sel, hidden=0.05),
        )
    )
    return layered + fit_layer + label_layer, fit_parts
