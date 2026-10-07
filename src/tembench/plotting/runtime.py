from __future__ import annotations

import math
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
    default_series,
    fit_frame,
    label,
    legend_opacity,
    legend_toggle,
    message_chart,
    metric_name,
    metric_unit,
    multi_bench,
    number_axis,
    plottable_rows,
    read_summary,
    resolve_y,
    shared_color_scale,
    wants_log,
    with_series_label,
    x_title,
)

#: Height of one runtime panel, in pixels; label spacing is worked out from it.
HEIGHT = 360
#: Plot width, in pixels, below which end-of-line labels drop the series name.
_NARROW = 520
#: Vertical room one end-of-line label needs, in pixels.
_LABEL_GAP = 15

#: Which part of the chart a row of the combined frame belongs to.
_KIND = "_kind"


def _formula_text(formula: str, y_col: str) -> str:
    """Bound formula named for what it bounds: T(n) is time, not memory."""
    if metric_name(y_col) == "Runtime":
        return formula
    return formula.replace("T(n)", "f(n)", 1)


def fit_runtime(
    summary: Path | pd.DataFrame,
    x: str = "n",
    y: str = "time_ms_median",
    color: str | None = "impl",
    by: list[str] | None = None,
    complexity_strategy: str = "heuristic",
) -> tuple[pd.DataFrame, list[str]]:
    """Fit a complexity class to each series, grouped exactly as the chart draws it.

    Returns the fits and the columns they are grouped by.  The CLI prints and
    exports this frame and hands it to `plot_runtime`, so what is drawn, shown
    and saved is one fit, computed once.
    """
    df = read_summary(summary)
    y_col = resolve_y(df, y, TIME_COLUMN_PREFERENCE)
    fit_df, by_cols = fit_frame(df, x, default_series(df, color))
    if by:
        by_cols = by
    fits = fit_models(
        fit_df,
        x_col=x,
        y_col=y_col,
        by=by_cols,
        strategy=complexity_strategy,
        count_col=count_column_for(y_col),
        spread_cols=spread_columns_for(y_col),
    )
    return fits, by_cols


def plot_runtime(
    summary: Path | pd.DataFrame,
    x: str = "n",
    y: str = "time_ms_median",
    color: str | None = "impl",
    bench: str | None = None,
    show_fit: bool = True,
    by: list[str] | None = None,
    complexity_strategy: str = "heuristic",
    log_x: bool | None = None,
    log_y: bool | None = None,
    fits: pd.DataFrame | None = None,
    title: str | None = None,
) -> alt.TopLevelMixin:
    """Runtime against input size, with each series' fitted upper bound.

    Several benchmarks with series of their own get a panel each; without, the
    benchmarks themselves are the series.  `log_x`/`log_y` left as None pick a
    log axis when the data spans more than `LOG_SPAN`.  Pass `fits` (from
    `fit_runtime`) to draw fits already computed, and `title` to replace the
    default "Runtime vs input size".
    """
    everything = read_summary(summary)
    df = read_summary(everything, bench)
    color = default_series(df, color)
    facet = multi_bench(df) and color != "bench"
    y_col = resolve_y(df, y, TIME_COLUMN_PREFERENCE)
    title = title or f"{metric_name(y_col)} vs {x_title(x)}"

    if show_fit and fits is None:
        # The fit sees every measured point; the chart only what its axes show.
        fits, _ = fit_runtime(
            df, x=x, y=y_col, color=color, by=by, complexity_strategy=complexity_strategy
        )
    by_cols = by or fit_frame(df, x, color)[1]
    df, series, series_title = with_series_label(df, x, color)
    measured, _ = plottable_rows(df, x, y_col)
    if log_x is None:
        log_x = x in measured.columns and wants_log(measured[x])
    if log_y is None:
        log_y = y_col in measured.columns and wants_log(measured[y_col])
    shown, notes = plottable_rows(df, x, y_col, log_x=log_x, log_y=log_y)
    if shown.empty:
        return message_chart("; ".join(["No measured points to plot", *notes]), title)

    parts = [shown.assign(**{_KIND: "point"})]
    if show_fit and fits is not None and not fits.empty:
        parts.extend(
            _fit_rows(
                shown, fits, x=x, y_col=y_col, by_cols=by_cols, series=series,
                log_x=log_x, log_y=log_y, facet=facet,
            )
        )
    frame = pd.concat(parts, ignore_index=True)

    x_enc = alt.X(
        f"{x}:Q", title=label(x), scale=axis_scale(log_x), axis=number_axis(log_x, frame[x])
    )
    y_scale = axis_scale(log_y)
    spanned = pd.concat([frame[y_col], frame.get("yhat", pd.Series(dtype=float))])
    y_axis = number_axis(log_y, spanned)

    color_enabled = series is not None
    legend_sel = legend_toggle("rt_legend", series, color_enabled)
    # Every layer carries the same colour encoding: layers whose legend titles
    # differ get merged into one legend titled "A, A (click to toggle), …".
    color_enc = categorical_color(
        series,
        enabled=color_enabled,
        title=f"{series_title}  (click to toggle)",
        fallback_color=PALETTE[0],
        scale=shared_color_scale(everything if series in everything else df, series),
        legend=alt.Legend(symbolType="circle") if color_enabled else None,
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
            (y_col, label(y_col), NUMBER_FORMAT),
        ]
    )
    if series is not None and series != "bench":
        tooltips.append(alt.Tooltip(series, title=series_title))
    if "bench" in df.columns:
        tooltips.insert(0, alt.Tooltip("bench", title="Benchmark"))

    def only(kind: str) -> str:
        return f"datum.{_KIND} === '{kind}'"

    # Both selections live on this one view.  Spread over two layers, Altair
    # lists each selection under both views once the chart sits in a
    # dashboard's vconcat, and Vega rejects the duplicated signal.
    points = (
        alt.Chart()
        .transform_filter(only("point"))
        .mark_point(shape="circle", size=64, filled=True)
        .encode(
            x=x_enc,
            y=alt.Y(f"{y_col}:Q", title=label(y_col), scale=y_scale, axis=y_axis),
            color=color_enc,
            opacity=legend_opacity(legend_sel),
            tooltip=tooltips,
        )
        .add_params(*[p for p in (legend_sel, nearest) if p is not None])
    )
    # The hover rule stays in the data and is hidden by opacity: a layer
    # filtered down to the (initially empty) selection has no extent.
    rule = (
        alt.Chart()
        .transform_filter(only("point"))
        .mark_rule(strokeWidth=1, strokeDash=[4, 3])
        .encode(x=x_enc, opacity=alt.condition(nearest, alt.value(0.8), alt.value(0)))
    )
    layers: list[alt.Chart] = [points, rule]

    if (frame[_KIND] == "fit").any():
        layers.extend(
            _fit_layers(
                frame,
                x=x,
                y_col=y_col,
                by_cols=by_cols,
                color_enc=color_enc,
                legend_sel=legend_sel,
                x_enc=x_enc,
                y_scale=y_scale,
                only=only,
                responsive=not facet,
            )
        )

    chart = alt.layer(*layers, data=frame)
    title_params = chart_title(title, notes)
    if facet:
        return (
            chart.properties(width=640, height=HEIGHT)
            .facet(
                row=alt.Row(
                    "bench:N",
                    title=None,
                    header=alt.Header(labelAngle=0, labelOrient="top", labelAnchor="start"),
                )
            )
            # Benchmarks rarely share a size range or a speed.
            .resolve_scale(x="independent", y="independent")
            .properties(title=title_params)
        )
    return chart.properties(width=640, height=HEIGHT, title=title_params)


def _fit_rows(
    shown: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    x: str,
    y_col: str,
    by_cols: list[str],
    series: str | None,
    log_x: bool,
    log_y: bool,
    facet: bool,
) -> list[pd.DataFrame]:
    """Each series' upper-bound curve, and one class label at its far end.

    They join the measured points in one frame, so a faceted chart splits all
    three by benchmark alike — a layer with data of its own would be drawn,
    whole, in every panel.
    """
    # Predictions span the drawn points only, so a failed largest size does
    # not stretch the curve into a region nothing measured.
    pred_src = shown
    missing_by = [c for c in by_cols if c not in pred_src.columns]
    if missing_by:  # the constant single-series group
        pred_src = pred_src.assign(**{c: "all" for c in missing_by})
    preds = predict_series(pred_src, fits, x_col=x, by=by_cols)
    if preds.empty:
        return []
    if series is not None and series not in preds.columns:
        # The combined series name is derived, so carry it over from the data.
        names = pred_src[by_cols + [series]].drop_duplicates(by_cols)
        preds = preds.merge(names, on=by_cols, how="left")
    if log_x:
        preds = preds[preds[x] > 0]
    if log_y:
        preds = preds[preds["yhat"] > 0]
    if preds.empty:
        return []
    preds = preds.assign(formula=preds["formula"].map(lambda f: _formula_text(f, y_col)))

    ends = preds.sort_values(x).groupby(by_cols, dropna=False, sort=False).tail(1).copy()
    ends["_label"] = [_class_label(row, series) for _, row in ends.iterrows()]
    ends["_short_label"] = [_class_label(row, None, short=True) for _, row in ends.iterrows()]
    panels = ends.groupby("bench", dropna=False) if facet and "bench" in ends else [(None, ends)]
    label_y = ends["yhat"].astype(float)
    for key, group in panels:
        panel = shown if key is None else shown[shown["bench"] == key]
        panel_preds = preds if key is None else preds[preds["bench"] == key]
        values = pd.concat([panel[y_col], panel_preds["yhat"]])
        label_y.update(_spread_labels(group["yhat"], values, log_y))
    ends["_label_y"] = label_y
    return [preds.assign(**{_KIND: "fit"}), ends.assign(**{_KIND: "label"})]


def _class_label(row: pd.Series, series: str | None, short: bool = False) -> str:
    klass = str(row.get("display_model", row["model"]))
    # A class the data cannot support must not read like one it can.
    if row.get("confidence") == "low":
        klass = f"{klass}?" if short else f"{klass} (low confidence)"
    return f"{row[series]}: {klass}" if series is not None else klass


def _spread_labels(ends: pd.Series, values: pd.Series, log_y: bool) -> pd.Series:
    """Heights for end-of-line labels that keep them from overlapping.

    Curves that finish close together would stack their labels on top of each
    other, so each label is lifted just far enough to clear the one below.
    The spacing is worked out on the axis as drawn — in log space on a log
    axis — over the data range `values`, assuming a `HEIGHT`-pixel panel.
    """
    def pos(v: float) -> float:
        return math.log10(v) if log_y else v

    finite = [pos(v) for v in values if pd.notna(v) and (v > 0 or not log_y)]
    if not finite:
        return ends.astype(float)
    low = min(finite) if log_y else min(0.0, *finite)
    span = (max(finite) - low) or 1.0
    gap = _LABEL_GAP / HEIGHT
    placed: dict[object, float] = {}
    previous = -math.inf
    for i in sorted(ends.index, key=lambda i: ends[i]):
        target = max((pos(ends[i]) - low) / span, previous + gap)
        value = low + target * span
        placed[i] = 10**value if log_y else value
        previous = target
    return pd.Series(placed, dtype=float)


def _fit_layers(
    frame: pd.DataFrame,
    *,
    x: str,
    y_col: str,
    by_cols: list[str],
    color_enc,
    legend_sel,
    x_enc: alt.X,
    y_scale: alt.Scale,
    only,
    responsive: bool,
) -> list[alt.Chart]:
    """The dashed upper-bound curves and their class labels."""
    unit = metric_unit(y_col)
    label_field = "display_model" if "display_model" in frame.columns else "model"
    bound_title = f"Upper bound ({unit})" if unit else "Upper bound"
    fit_tooltips = build_tooltips(
        [
            (label_field, "Complexity", None),
            ("formula", "Bound", None),
            (x, label(x), ","),
            ("yhat", bound_title, NUMBER_FORMAT),
        ]
    )
    if "bench" in frame.columns:
        fit_tooltips.insert(0, alt.Tooltip("bench", title="Benchmark"))
    if "empirical_exponent" in frame.columns:
        fit_tooltips.extend(
            build_tooltips(
                [
                    ("empirical_exponent", "Exponent", ".2f"),
                    ("exponent_ci_low", "Exponent CI low", ".2f"),
                    ("exponent_ci_high", "Exponent CI high", ".2f"),
                ]
            )
        )
    if "confidence" in frame.columns:
        fit_tooltips.extend(
            build_tooltips(
                [
                    ("confidence", "Confidence", None),
                    ("confidence_notes", "Caveats", None),
                ]
            )
        )
    y_fit = alt.Y("yhat:Q", title=label(y_col), scale=y_scale)

    curve = (
        alt.Chart()
        .transform_filter(only("fit"))
        .mark_line(strokeDash=[6, 4], strokeWidth=1.75)
        .encode(
            x=x_enc,
            y=y_fit,
            color=color_enc,
            opacity=legend_opacity(legend_sel, shown=0.85, hidden=0.05),
            detail=[f"{c}:N" for c in by_cols],
            tooltip=fit_tooltips,
        )
    )
    # Labels sit just past the end of their curve, in text ink rather than the
    # series colour: several palette colours are too light to read as text.
    # On a narrow chart (a phone) the full label would squeeze the plot, so
    # only the class is shown and the legend and tooltip name the series.  A
    # faceted chart has a fixed width, so it always has room for the full one.
    shorten = f"width < {_NARROW} ? datum._short_label : datum._label" if responsive else "datum._label"
    text = (
        alt.Chart()
        .transform_filter(only("label"))
        .transform_calculate(_text=shorten)
        .mark_text(align="left", baseline="middle", dx=8, fontSize=11, fontWeight=600)
        .encode(
            x=x_enc,
            y=alt.Y("_label_y:Q", title=label(y_col), scale=y_scale),
            text="_text:N",
            opacity=legend_opacity(legend_sel, hidden=0.05),
            tooltip=fit_tooltips,
        )
    )
    return [curve, text]
