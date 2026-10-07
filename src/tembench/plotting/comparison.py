from __future__ import annotations

import altair as alt
import pandas as pd

from ..reporting.comparison import VERDICT_TEXT
from ._common import NUMBER_FORMAT, chart_title, message_chart

#: Status colours, kept apart from the categorical palette: they mean
#: "worse" and "better", never "series 4".
_VERDICT_COLORS = {"Slower": "#d03b3b", "Faster": "#1f8a4c", "Unchanged": "#9a9993", "New": "#9a9993"}


def plot_deltas(
    comparison_df: pd.DataFrame,
    verdict: pd.DataFrame,
    keys: list[str],
    threshold_pct: float,
) -> alt.TopLevelMixin:
    """Each configuration's change against the baseline, largest slowdown first.

    `verdict` is `reporting.comparison.verdicts(comparison_df, threshold_pct)`.
    The threshold band is drawn, so it is plain which changes decided anything.
    """
    title = "Change against the baseline"
    # A grid column with one value (a lone benchmark, say) only lengthens labels.
    named = [k for k in keys if comparison_df[k].nunique(dropna=False) > 1] or keys
    rows = verdict.assign(
        _config=comparison_df[named].astype(str).agg(" · ".join, axis=1) if named else "all",
        _verdict=verdict["verdict"].map(VERDICT_TEXT).fillna(verdict["verdict"]),
    )
    drawn = rows[rows["delta_pct"].notna()]
    notes = []
    if len(drawn) < len(rows):
        notes.append(f"{len(rows) - len(drawn)} configuration(s) without a change to show")
    if drawn.empty:
        return message_chart("; ".join(["Nothing to compare", *notes]), title)

    order = drawn.sort_values("delta_pct", ascending=False)["_config"].tolist()
    present = [v for v in _VERDICT_COLORS if v in set(drawn["_verdict"])]
    y = alt.Y("_config:N", title=None, sort=order, axis=alt.Axis(labelLimit=260))
    x = alt.X("delta_pct:Q", title="Change (%)", axis=alt.Axis(format="+~f", tickCount=8))
    color = alt.Color(
        "_verdict:N",
        title=None,
        scale=alt.Scale(domain=present, range=[_VERDICT_COLORS[v] for v in present]),
    )
    tooltip = [
        alt.Tooltip("_config:N", title="Configuration"),
        alt.Tooltip("_verdict:N", title="Verdict"),
        alt.Tooltip("delta_pct:Q", title="Change (%)", format="+.1f"),
        alt.Tooltip("current:Q", title="Current", format=NUMBER_FORMAT),
        alt.Tooltip("baseline:Q", title="Baseline", format=NUMBER_FORMAT),
    ]
    band = (
        alt.Chart(pd.DataFrame({"lo": [-threshold_pct], "hi": [threshold_pct]}))
        .mark_rect(opacity=0.12, color="#9a9993")
        .encode(x=alt.X("lo:Q", title="Change (%)"), x2="hi:Q")
    )
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(strokeWidth=1).encode(x=alt.X("zero:Q", title="Change (%)"))
    stems = (
        alt.Chart(drawn)
        .mark_rule(strokeWidth=2)
        .encode(y=y, x=alt.X("zero_:Q", title="Change (%)"), x2="delta_pct:Q", color=color, tooltip=tooltip)
        .transform_calculate(zero_="0")
    )
    dots = alt.Chart(drawn).mark_point(filled=True, size=70).encode(
        y=y, x=x, color=color, tooltip=tooltip
    )
    return alt.layer(band, zero, stems, dots).properties(
        width=640,
        height=alt.Step(22),
        title=chart_title(title, [f"Shaded: within ±{threshold_pct:g}%", *notes]),
    )
