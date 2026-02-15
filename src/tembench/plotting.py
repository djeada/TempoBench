from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd

from .complexity import fit_models, predict_series

# ---------------------------------------------------------------------------
# Shared theme
# ---------------------------------------------------------------------------

_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be185d",
    "#65a30d",
]


def _apply_theme() -> None:
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
            "range": {"category": _PALETTE},
            "line": {"strokeWidth": 2.5},
            "point": {"size": 60, "filled": True},
        }
    }

    # Use new API (altair ≥5.5) when available, fall back to legacy
    if hasattr(alt, "theme") and hasattr(alt.theme, "register"):

        @alt.theme.register("tempobench", enable=True)
        def _tb_theme():
            return alt.theme.ThemeConfig(theme_config)

    else:
        alt.themes.register("tempobench", lambda: theme_config)
        alt.themes.enable("tempobench")


_apply_theme()

# ---------------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------------

_NICE_LABELS = {
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


def _label(col: str) -> str:
    return _NICE_LABELS.get(col, col)


def _resolve_y(df: pd.DataFrame, y: str, fallbacks: list[str]) -> str:
    if y in df.columns:
        return y
    for c in fallbacks:
        if c in df.columns:
            return c
    return y


# ---------------------------------------------------------------------------
# Runtime plot
# ---------------------------------------------------------------------------


def plot_runtime(
    summary_csv: Path,
    x: str = "n",
    y: str = "wall_ms_median",
    color: str = "impl",
    bench: str | None = None,
    show_fit: bool = True,
    by: list[str] | None = None,
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
    y_col = _resolve_y(df, y, ["wall_ms_median", "wall_ms_mean"])

    x_scale = alt.Scale(type="log") if log_x else alt.Scale(zero=True)
    y_scale = alt.Scale(type="log") if log_y else alt.Scale(zero=True)
    x_enc = alt.X(x, title=_label(x), scale=x_scale, axis=alt.Axis(format="~s"))
    y_enc = alt.Y(y_col, title=_label(y_col), scale=y_scale)

    # Shared color scale — same color for each impl across data + fit
    impl_values = sorted(df[color].unique()) if color in df.columns else []
    shared_color_scale = (
        alt.Scale(domain=impl_values, range=_PALETTE[: len(impl_values)])
        if impl_values
        else None
    )

    # Interactive legend selection — click to toggle series visibility
    legend_sel = (
        alt.selection_point(name="rt_legend", fields=[color], bind="legend")
        if color in df.columns
        else None
    )

    color_enc = (
        alt.Color(
            color,
            title=_label(color),
            scale=shared_color_scale,
            legend=alt.Legend(
                title="Implementation  (click to toggle)",
                symbolType="cross",
                symbolSize=150,
                symbolStrokeWidth=2.5,
            ),
        )
        if color in df.columns
        else alt.value(_PALETTE[0])
    )

    # Nearest-point selection for interactive crosshair tooltip
    nearest = alt.selection_point(
        name="rt_nearest",
        nearest=True,
        on="pointerover",
        fields=[x],
        empty=False,
    )

    tooltips = [
        alt.Tooltip(x, title=_label(x), format=","),
        alt.Tooltip(y_col, title=_label(y_col), format=".2f"),
        (
            alt.Tooltip(color, title=_label(color))
            if color in df.columns
            else alt.Tooltip(y_col)
        ),
    ]
    if "bench" in df.columns:
        tooltips.insert(0, alt.Tooltip("bench", title="Benchmark"))

    # Measured data — discrete sample points (crosses), NOT connected lines
    base = (
        alt.Chart(df)
        .mark_point(shape="cross", size=200, strokeWidth=3, filled=False)
        .encode(
            x=x_enc,
            y=y_enc,
            color=color_enc,
            opacity=(
                alt.condition(legend_sel, alt.value(1.0), alt.value(0.08))
                if legend_sel
                else alt.value(1.0)
            ),
            tooltip=tooltips,
        )
        .properties(width=640, height=400, title="Runtime vs Input Size")
    )
    if legend_sel:
        base = base.add_params(legend_sel)

    # Invisible voronoi layer for nearest-point snapping
    voronoi = (
        alt.Chart(df)
        .mark_point(size=0, opacity=0)
        .encode(x=x_enc, y=y_enc, tooltip=tooltips)
        .add_params(nearest)
    )

    # Vertical rule at nearest x for crosshair effect
    rule = (
        alt.Chart(df)
        .mark_rule(color="#94a3b8", strokeWidth=1, strokeDash=[4, 3])
        .encode(x=x_enc, opacity=alt.condition(nearest, alt.value(0.6), alt.value(0)))
        .transform_filter(nearest)
    )

    # Highlighted dots at nearest x
    highlight_dots = (
        alt.Chart(df)
        .mark_point(
            shape="circle", size=100, filled=True, strokeWidth=2, stroke="white"
        )
        .encode(
            x=x_enc,
            y=y_enc,
            color=(
                alt.Color(color, scale=shared_color_scale, legend=None)
                if color in df.columns
                else alt.value(_PALETTE[0])
            ),
            opacity=(
                alt.condition(legend_sel, alt.value(1.0), alt.value(0.0))
                if legend_sel
                else alt.value(1.0)
            ),
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

    fits = fit_models(df, x_col=x, y_col=y_col, by=by_cols)
    if fits.empty:
        return base + voronoi + rule + highlight_dots
    preds = predict_series(df, fits, x_col=x, by=by_cols)
    if preds.empty:
        return base + voronoi + rule + highlight_dots

    # Fit color uses the same scale as data points (same impl → same color)
    fit_color_enc = (
        alt.Color(color, scale=shared_color_scale, legend=None)
        if color in preds.columns
        else alt.value(_PALETTE[0])
    )

    # Fit overlay — smooth dashed upper-bound curve, toggles with legend
    fit_layer = (
        alt.Chart(preds)
        .mark_line(strokeDash=[8, 4], strokeWidth=2)
        .encode(
            x=alt.X(x, title=_label(x), scale=x_scale, axis=alt.Axis(format="~s")),
            y=alt.Y("yhat", title=_label(y_col), scale=y_scale),
            color=fit_color_enc,
            opacity=(
                alt.condition(legend_sel, alt.value(0.7), alt.value(0.05))
                if legend_sel
                else alt.value(0.7)
            ),
            detail=by_cols,
            tooltip=[
                alt.Tooltip("model", title="Complexity"),
                alt.Tooltip("formula", title="Bound"),
                alt.Tooltip(x, title=_label(x), format=","),
                alt.Tooltip("yhat", title="Upper Bound (ms)", format=".2f"),
            ],
        )
    )

    # Labels: "O(…)" placed at different x positions per series to avoid overlap
    preds_sorted = preds.sort_values(x)
    label_rows = []
    label_positions = [0.45, 0.65, 0.80, 0.55, 0.70, 0.85]
    for i, (_key, g) in enumerate(preds_sorted.groupby(by_cols, dropna=False)):
        pos = label_positions[i % len(label_positions)]
        idx = int(len(g) * pos)
        idx = max(0, min(idx, len(g) - 1))
        row = g.iloc[idx].copy()
        impl_name = row.get(color, "") if color in preds.columns else ""
        row["_label"] = f"{impl_name}: {row['model']}" if impl_name else row["model"]
        label_rows.append(row)
    label_points = pd.DataFrame(label_rows).reset_index(drop=True)

    label_points = label_points.sort_values("yhat", ascending=False).reset_index(
        drop=True
    )
    n_labels = len(label_points)
    dy_offsets = [-14 - i * 16 for i in range(n_labels)]
    label_points["_dy"] = dy_offsets

    label_layer = (
        alt.Chart(label_points)
        .mark_text(align="left", dx=6, fontSize=12, fontWeight=700)
        .encode(
            x=x,
            y=alt.Y("yhat:Q", scale=y_scale),
            text="_label:N",
            color=(
                alt.Color(color, scale=shared_color_scale, legend=None)
                if color in label_points.columns
                else alt.value(_PALETTE[0])
            ),
            opacity=(
                alt.condition(legend_sel, alt.value(1.0), alt.value(0.05))
                if legend_sel
                else alt.value(1.0)
            ),
        )
    )

    # Combine all layers
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
    subtitle_text = " │ ".join(fit_info_parts)

    return layered.properties(
        title=alt.TitleParams(
            text="Runtime vs Input Size",
            subtitle=subtitle_text,
            subtitleFontSize=10,
            subtitleColor="#64748b",
            subtitlePadding=4,
        )
    )


# ---------------------------------------------------------------------------
# Memory plot
# ---------------------------------------------------------------------------


def plot_memory(
    summary_csv: Path,
    x: str = "n",
    y: str = "peak_rss_mb_median",
    color: str = "impl",
    log_x: bool = False,
    log_y: bool = False,
) -> alt.Chart:
    """Create a memory usage line chart from the summary CSV."""
    df = pd.read_csv(summary_csv)
    y_col = _resolve_y(df, y, ["peak_rss_mb_median", "peak_rss_mb_mean", "peak_rss_mb"])

    if y_col not in df.columns:
        return (
            alt.Chart(pd.DataFrame())
            .mark_text()
            .encode(text=alt.value("No memory data available"))
        )

    x_scale = alt.Scale(type="log") if log_x else alt.Scale(zero=True)
    y_scale = alt.Scale(type="log") if log_y else alt.Scale(zero=True)

    legend_sel = (
        alt.selection_point(name="mem_legend", fields=[color], bind="legend")
        if color in df.columns
        else None
    )

    color_enc = (
        alt.Color(
            color,
            title=_label(color),
            scale=alt.Scale(range=_PALETTE),
            legend=alt.Legend(
                title="Implementation  (click to toggle)",
            ),
        )
        if color in df.columns
        else alt.value(_PALETTE[2])
    )

    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=50))
        .encode(
            x=alt.X(x, title=_label(x), scale=x_scale, axis=alt.Axis(format="~s")),
            y=alt.Y(y_col, title=_label(y_col), scale=y_scale),
            color=color_enc,
            opacity=(
                alt.condition(legend_sel, alt.value(1.0), alt.value(0.08))
                if legend_sel
                else alt.value(1.0)
            ),
            tooltip=[
                alt.Tooltip(x, title=_label(x), format=","),
                alt.Tooltip(y_col, title=_label(y_col), format=".2f"),
            ],
        )
        .properties(width=640, height=360, title="Memory Usage vs Input Size")
    )
    if legend_sel:
        chart = chart.add_params(legend_sel)
    return chart


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


def plot_heatmap(
    summary_csv: Path,
    x: str = "n",
    y: str = "impl",
    value: str = "wall_ms_median",
) -> alt.Chart:
    """Create a performance heatmap."""
    df = pd.read_csv(summary_csv)

    if x not in df.columns or y not in df.columns:
        return (
            alt.Chart(pd.DataFrame())
            .mark_text()
            .encode(text=alt.value("Insufficient data for heatmap"))
        )

    value_col = _resolve_y(df, value, ["wall_ms_median", "wall_ms_mean"])

    rect = (
        alt.Chart(df)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X(f"{x}:O", title=_label(x), axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y}:O", title=_label(y)),
            color=alt.Color(
                f"{value_col}:Q",
                title=_label(value_col),
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(
                    direction="horizontal", orient="bottom", gradientLength=300
                ),
            ),
            tooltip=[
                alt.Tooltip(x, title=_label(x)),
                alt.Tooltip(y, title=_label(y)),
                alt.Tooltip(value_col, title=_label(value_col), format=".1f"),
            ],
        )
        .properties(width=640, height=300, title=f"Heatmap: {_label(value_col)}")
    )

    text = (
        alt.Chart(df)
        .mark_text(fontSize=13, fontWeight=500)
        .encode(
            x=alt.X(f"{x}:O"),
            y=alt.Y(f"{y}:O"),
            text=alt.Text(f"{value_col}:Q", format=".1f"),
            color=alt.condition(
                alt.datum[value_col] > df[value_col].median(),
                alt.value("white"),
                alt.value("#1e293b"),
            ),
        )
    )

    return rect + text


# ---------------------------------------------------------------------------
# Boxplot
# ---------------------------------------------------------------------------


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
                        for k, v in row["params"].items():
                            row[k] = v
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

    chart = (
        alt.Chart(df)
        .mark_boxplot(
            extent="min-max", size=40, median={"color": "white", "strokeWidth": 2}
        )
        .encode(
            x=alt.X(f"{x}:O", title=_label(x), axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y}:Q", title=_label(y), scale=alt.Scale(zero=True)),
            color=alt.Color(
                f"{x}:N",
                title=_label(x) + "  (click to toggle)",
                scale=alt.Scale(range=_PALETTE),
            ),
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.12)),
        )
        .add_params(highlight)
        .properties(width=640, height=360, title=f"Distribution: {_label(y)}")
    )
    return chart


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def plot_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "wall_ms_median",
    group: str = "impl",
) -> alt.Chart:
    """Create a grouped bar chart comparing current vs baseline."""
    curr_col = f"{metric}_current"
    base_col = f"{metric}_baseline"

    if curr_col not in comparison_df.columns or base_col not in comparison_df.columns:
        return (
            alt.Chart(pd.DataFrame())
            .mark_text()
            .encode(text=alt.value("Comparison data not available"))
        )

    suffixes = ("_current", "_baseline", "_delta", "_delta_pct", "_regression")
    id_vars = [c for c in comparison_df.columns if not c.endswith(suffixes)]
    df_melted = comparison_df.melt(
        id_vars=id_vars,
        value_vars=[curr_col, base_col],
        var_name="version",
        value_name="value",
    )
    df_melted["version"] = df_melted["version"].map(
        {curr_col: "Current", base_col: "Baseline"}
    )

    x_col = (
        group if group in df_melted.columns else (id_vars[0] if id_vars else "index")
    )

    ver_sel = alt.selection_point(name="cmp_version", fields=["version"], bind="legend")

    chart = (
        alt.Chart(df_melted)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{x_col}:O", title=_label(x_col), axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title=_label(metric), scale=alt.Scale(zero=True)),
            color=alt.Color(
                "version:N",
                title="Version  (click to toggle)",
                scale=alt.Scale(
                    domain=["Baseline", "Current"], range=["#94a3b8", "#2563eb"]
                ),
            ),
            opacity=alt.condition(ver_sel, alt.value(1.0), alt.value(0.12)),
            xOffset="version:N",
            tooltip=[
                alt.Tooltip(x_col, title=_label(x_col)),
                alt.Tooltip("version", title="Version"),
                alt.Tooltip("value:Q", title=_label(metric), format=".2f"),
            ],
        )
        .add_params(ver_sel)
        .properties(width=640, height=400, title=f"Comparison: {_label(metric)}")
    )
    return chart


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def create_dashboard(
    summary_csv: Path,
    runs_jsonl: Optional[Path] = None,
    x: str = "n",
    color: str = "impl",
    title: str = "TempoBench Dashboard",
    log_x: bool = False,
    log_y: bool = False,
) -> alt.VConcatChart:
    """Create a multi-chart dashboard."""
    df = pd.read_csv(summary_csv)
    charts: list[alt.Chart] = []

    charts.append(
        plot_runtime(
            summary_csv, x=x, color=color, show_fit=True, log_x=log_x, log_y=log_y
        )
    )

    if "peak_rss_mb_median" in df.columns or "peak_rss_mb_mean" in df.columns:
        charts.append(
            plot_memory(summary_csv, x=x, color=color, log_x=log_x, log_y=log_y)
        )

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


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_chart(
    chart: alt.Chart,
    output_path: Path,
    fmt: str = "html",
) -> str:
    """Save chart to file. Supports html, json, png, svg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "html":
        chart.save(output_path)
        return str(output_path)
    elif fmt == "json":
        with open(output_path, "w") as f:
            f.write(chart.to_json())
        return str(output_path)
    elif fmt in ("png", "svg"):
        try:
            chart.save(output_path, format=fmt)
            return str(output_path)
        except Exception:
            html_path = output_path.with_suffix(".html")
            chart.save(html_path)
            return str(html_path)
    else:
        raise ValueError(
            f"Unsupported format: {fmt}. Use 'html', 'json', 'png', or 'svg'."
        )
