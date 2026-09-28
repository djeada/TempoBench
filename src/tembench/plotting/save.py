from __future__ import annotations

from pathlib import Path
from typing import Literal

import altair as alt

from ..reporting.resources import json_for_script, vega_script_tags

# Altair's own HTML export embeds the spec unescaped, so a benchmark named
# "</script>…" would break out of the page's script; this page escapes it.
_PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    #vis.vega-embed {{ width: 100%; display: flex; }}
    #vis.vega-embed details, #vis.vega-embed details summary {{ position: relative; }}
  </style>
{scripts}
</head>
<body>
  <div id="vis"></div>
  <script>
    (function(vegaEmbed) {{
      var spec = {spec};
      var embedOpt = {{"mode": "vega-lite"}};
      vegaEmbed("#vis", spec, embedOpt).catch(function(err) {{
        var pre = document.createElement("pre");
        pre.textContent = "Error rendering chart: " + err;
        document.getElementById("vis").appendChild(pre);
      }});
    }})(vegaEmbed);
  </script>
</body>
</html>
"""


def chart_html(chart: alt.TopLevelMixin) -> str:
    """A standalone HTML page rendering `chart`, safe for any data it holds."""
    spec = json_for_script(chart.to_json(indent=None))
    return _PAGE.format(scripts=vega_script_tags(), spec=spec)


def save_chart(
    chart: alt.TopLevelMixin,
    output_path: Path,
    fmt: Literal["html", "json", "png", "svg"] = "html",
) -> str:
    """Save chart to file. Supports html, json, png, svg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "html":
        output_path.write_text(chart_html(chart), encoding="utf-8")
        return str(output_path)
    if fmt == "json":
        with output_path.open("w", encoding="utf-8") as f:
            f.write(chart.to_json())
        return str(output_path)
    if fmt in ("png", "svg"):
        try:
            chart.save(output_path, format=fmt)
            return str(output_path)
        except Exception:
            html_path = output_path.with_suffix(".html")
            html_path.write_text(chart_html(chart), encoding="utf-8")
            return str(html_path)
    raise ValueError(
        f"Unsupported format: {fmt}. Use 'html', 'json', 'png', or 'svg'."
    )
