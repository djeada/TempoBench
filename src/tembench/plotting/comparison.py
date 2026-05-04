from __future__ import annotations

import altair as alt
import pandas as pd

from ._common import label


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

    x_col = group if group in df_melted.columns else (id_vars[0] if id_vars else "index")
    ver_sel = alt.selection_point(name="cmp_version", fields=["version"], bind="legend")

    return (
        alt.Chart(df_melted)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{x_col}:O", title=label(x_col), axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title=label(metric), scale=alt.Scale(zero=True)),
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
                alt.Tooltip(x_col, title=label(x_col)),
                alt.Tooltip("version", title="Version"),
                alt.Tooltip("value:Q", title=label(metric), format=".2f"),
            ],
        )
        .add_params(ver_sel)
        .properties(width=640, height=400, title=f"Comparison: {label(metric)}")
    )
