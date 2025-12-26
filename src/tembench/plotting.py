from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

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


def plot_memory(
    summary_csv: Path,
    x: str = "n",
    y: str = "peak_rss_mb_median",
    color: str = "impl",
    log_x: bool = False,
    log_y: bool = False,
) -> alt.Chart:
    """Create a memory usage plot from the summary CSV."""
    df = pd.read_csv(summary_csv)
    
    y_col = y
    if y_col not in df.columns:
        for cand in ["peak_rss_mb_median", "peak_rss_mb_mean", "peak_rss_mb"]:
            if cand in df.columns:
                y_col = cand
                break
    
    if y_col not in df.columns:
        # Return empty chart if no memory data
        return alt.Chart(pd.DataFrame()).mark_text().encode(
            text=alt.value("No memory data available")
        )
    
    x_enc = alt.X(x, title=x, scale=alt.Scale(type="log")) if log_x else alt.X(x, title=x)
    y_enc = alt.Y(y_col, title="Memory (MB)", scale=alt.Scale(type="log")) if log_y else alt.Y(y_col, title="Memory (MB)")
    
    chart = (
        alt.Chart(df)
        .mark_area(opacity=0.6, line=True)
        .encode(
            x=x_enc,
            y=y_enc,
            color=color if color in df.columns else alt.value("#16a34a"),
            tooltip=list(df.columns),
        )
        .properties(width=600, height=400, title="Memory Usage vs Input Size")
    )
    return chart


def plot_heatmap(
    summary_csv: Path,
    x: str = "n",
    y: str = "impl",
    value: str = "wall_ms_median",
) -> alt.Chart:
    """Create a heatmap of performance metrics."""
    df = pd.read_csv(summary_csv)
    
    if x not in df.columns or y not in df.columns:
        return alt.Chart(pd.DataFrame()).mark_text().encode(
            text=alt.value("Insufficient data for heatmap")
        )
    
    value_col = value
    if value_col not in df.columns:
        for cand in ["wall_ms_median", "wall_ms_mean"]:
            if cand in df.columns:
                value_col = cand
                break
    
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(f"{x}:O", title=x),
            y=alt.Y(f"{y}:O", title=y),
            color=alt.Color(
                f"{value_col}:Q",
                title=value_col,
                scale=alt.Scale(scheme="blues"),
            ),
            tooltip=[x, y, value_col],
        )
        .properties(width=600, height=400, title=f"Performance Heatmap: {value_col}")
    )
    
    # Add text labels
    text = chart.mark_text(baseline="middle").encode(
        text=alt.Text(f"{value_col}:Q", format=".1f"),
        color=alt.condition(
            alt.datum[value_col] > df[value_col].median(),
            alt.value("white"),
            alt.value("black"),
        ),
    )
    
    return chart + text


def plot_boxplot(
    runs_jsonl: Path,
    x: str = "impl",
    y: str = "wall_ms",
) -> alt.Chart:
    """Create a boxplot from raw JSONL runs to show distribution."""
    rows = []
    with runs_jsonl.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("status") == "ok":
                    # Flatten params
                    if "params" in row:
                        for k, v in row["params"].items():
                            row[k] = v
                    rows.append(row)
            except json.JSONDecodeError:
                continue
    
    if not rows:
        return alt.Chart(pd.DataFrame()).mark_text().encode(
            text=alt.value("No successful runs for boxplot")
        )
    
    df = pd.DataFrame(rows)
    
    if x not in df.columns or y not in df.columns:
        return alt.Chart(pd.DataFrame()).mark_text().encode(
            text=alt.value("Required columns not found")
        )
    
    chart = (
        alt.Chart(df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X(f"{x}:O", title=x),
            y=alt.Y(f"{y}:Q", title=f"{y} (ms)"),
            color=alt.Color(f"{x}:N"),
        )
        .properties(width=600, height=400, title=f"Distribution of {y} by {x}")
    )
    return chart


def plot_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "wall_ms_median",
    group: str = "impl",
) -> alt.Chart:
    """Create a comparison chart showing current vs baseline."""
    curr_col = f"{metric}_current"
    base_col = f"{metric}_baseline"
    
    if curr_col not in comparison_df.columns or base_col not in comparison_df.columns:
        return alt.Chart(pd.DataFrame()).mark_text().encode(
            text=alt.value("Comparison data not available")
        )
    
    # Melt the data for grouped bar chart
    id_vars = [c for c in comparison_df.columns if not c.endswith(('_current', '_baseline', '_delta', '_delta_pct', '_regression'))]
    df_melted = comparison_df.melt(
        id_vars=id_vars,
        value_vars=[curr_col, base_col],
        var_name="version",
        value_name="value",
    )
    df_melted["version"] = df_melted["version"].map({
        curr_col: "Current",
        base_col: "Baseline",
    })
    
    x_col = group if group in df_melted.columns else (id_vars[0] if id_vars else "index")
    
    chart = (
        alt.Chart(df_melted)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:O", title=x_col),
            y=alt.Y("value:Q", title=metric),
            color=alt.Color("version:N", scale=alt.Scale(domain=["Baseline", "Current"], range=["#94a3b8", "#2563eb"])),
            xOffset="version:N",
            tooltip=[x_col, "version", "value"],
        )
        .properties(width=600, height=400, title=f"Comparison: {metric}")
    )
    return chart


def create_dashboard(
    summary_csv: Path,
    runs_jsonl: Optional[Path] = None,
    x: str = "n",
    color: str = "impl",
    title: str = "TempoBench Dashboard",
) -> alt.VConcatChart:
    """Create a comprehensive dashboard with multiple charts."""
    df = pd.read_csv(summary_csv)
    
    charts = []
    
    # Runtime chart
    runtime_chart = plot_runtime(summary_csv, x=x, color=color, show_fit=True)
    runtime_chart = runtime_chart.properties(title="Runtime vs Input Size")
    charts.append(runtime_chart)
    
    # Memory chart (if available)
    if "peak_rss_mb_median" in df.columns or "peak_rss_mb_mean" in df.columns:
        memory_chart = plot_memory(summary_csv, x=x, color=color)
        charts.append(memory_chart)
    
    # Heatmap
    if color in df.columns and x in df.columns:
        heatmap = plot_heatmap(summary_csv, x=x, y=color)
        charts.append(heatmap)
    
    # Boxplot if raw runs available
    if runs_jsonl and runs_jsonl.exists():
        boxplot = plot_boxplot(runs_jsonl, x=color)
        charts.append(boxplot)
    
    # Combine charts vertically
    if len(charts) == 1:
        return charts[0]
    
    dashboard = alt.vconcat(*charts).properties(
        title=alt.TitleParams(text=title, fontSize=20, anchor="middle")
    )
    
    return dashboard


def save_chart(
    chart: alt.Chart,
    output_path: Path,
    format: str = "html",
) -> str:
    """Save chart to various formats.
    
    Args:
        chart: Altair chart to save
        output_path: Path to save the chart
        format: Output format - 'html', 'json', 'png', 'svg'
        
    Returns:
        Path to the saved file (may differ from output_path if fallback was used)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "html":
        chart.save(output_path)
        return str(output_path)
    elif format == "json":
        with open(output_path, "w") as f:
            f.write(chart.to_json())
        return str(output_path)
    elif format in ("png", "svg"):
        # These require additional dependencies (altair_saver or vl-convert)
        try:
            chart.save(output_path, format=format)
            return str(output_path)
        except Exception:
            # Fall back to HTML if PNG/SVG export fails
            html_path = output_path.with_suffix(".html")
            chart.save(html_path)
            return str(html_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'html', 'json', 'png', or 'svg'.")
