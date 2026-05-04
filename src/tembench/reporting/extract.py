"""Extract Vega-Lite specs from Altair-generated HTML."""

from __future__ import annotations

import re


def _extract_vega_spec(chart_html_text: str) -> str | None:
    """Extract the Vega-Lite JSON spec from an Altair-generated HTML file.

    Returns the spec as a string, or None if not found.
    """
    m = re.search(r"var\s+spec\s*=\s*(\{.*?\});\s*\n", chart_html_text, re.DOTALL)
    if m:
        return m.group(1)
    return None
