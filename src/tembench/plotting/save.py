from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path

import altair as alt

from ..reporting.resources import chart_slot, render_page


def is_wide(chart: alt.TopLevelMixin) -> bool:
    """Whether `chart` has to keep a fixed width (facets and concatenations)."""
    return not isinstance(chart, (alt.Chart, alt.LayerChart))


def responsive(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Let a single-view chart fill the width of whatever holds it.

    Vega-Lite can only size single and layered views to their container; a
    faceted or stacked chart keeps its fixed width.
    """
    if is_wide(chart):
        return chart
    return chart.properties(
        width="container",
        autosize=alt.AutoSizeParams(type="fit-x", contains="padding"),
    )


def chart_sections(charts: Sequence[alt.TopLevelMixin], start: int = 0) -> tuple[str, list[dict]]:
    """Page sections holding `charts`, and the specs to render into them.

    `start` numbers the slots, for a page that already holds other charts.
    """
    sections = []
    specs = []
    for i, chart in enumerate(charts, start=start):
        sections.append(
            f'<section class="section chart-section">{chart_slot(i, is_wide(chart))}</section>'
        )
        specs.append(responsive(chart).to_dict())
    return "\n".join(sections), specs


def chart_page(
    charts: Sequence[alt.TopLevelMixin], title: str, kind: str = "Chart", meta: str = ""
) -> str:
    """A standalone HTML page rendering `charts`, safe for any data they hold.

    `meta` is plain text shown under the title.
    """
    body, specs = chart_sections(charts)
    return render_page(title=title, kind=kind, meta=html.escape(meta), body=body, specs=specs)


def chart_html(chart: alt.TopLevelMixin, title: str = "TempoBench chart") -> str:
    """A standalone HTML page rendering `chart`."""
    return chart_page([chart], title)


def save_chart(
    chart: alt.TopLevelMixin | Sequence[alt.TopLevelMixin],
    output_path: Path,
    title: str = "TempoBench chart",
    kind: str = "Chart",
    meta: str = "",
) -> str:
    """Write `chart` (or several, one per section) as a standalone HTML page."""
    charts = [chart] if isinstance(chart, alt.TopLevelMixin) else list(chart)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(chart_page(charts, title, kind, meta), encoding="utf-8")
    return str(output_path)
