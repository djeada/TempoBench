from __future__ import annotations

from pathlib import Path
from typing import Literal

import altair as alt


def save_chart(
    chart: alt.TopLevelMixin,
    output_path: Path,
    fmt: Literal["html", "json", "png", "svg"] = "html",
) -> str:
    """Save chart to file. Supports html, json, png, svg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "html":
        chart.save(output_path)
        return str(output_path)
    if fmt == "json":
        with output_path.open("w") as f:
            f.write(chart.to_json())
        return str(output_path)
    if fmt in ("png", "svg"):
        try:
            chart.save(output_path, format=fmt)
            return str(output_path)
        except Exception:
            html_path = output_path.with_suffix(".html")
            chart.save(html_path)
            return str(html_path)
    raise ValueError(
        f"Unsupported format: {fmt}. Use 'html', 'json', 'png', or 'svg'."
    )
