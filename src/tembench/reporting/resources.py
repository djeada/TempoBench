"""Packaged HTML/CSS/JS assets used by HTML reports."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import altair as alt

_ASSET_PACKAGE = "tembench.reporting.assets"
_FONTS_HTML = "\n".join(
    [
        '  <link rel="preconnect" href="https://fonts.googleapis.com">',
        "  <link",
        '    href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap"',
        '    rel="stylesheet">',
    ]
)

_THEME_TOGGLE_BUTTON = "\n".join(
    [
        '  <button class="theme-toggle" id="themeToggle" aria-label="Toggle theme">',
        '    <span class="theme-toggle-icon" id="themeIcon">🌙</span>',
        '    <span id="themeLabel">Dark</span>',
        "  </button>",
    ]
)


@lru_cache(maxsize=1)
def load_report_css() -> str:
    return files(_ASSET_PACKAGE).joinpath("report.css").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_theme_toggle_js() -> str:
    return files(_ASSET_PACKAGE).joinpath("theme-toggle.js").read_text(encoding="utf-8")


def render_head_assets() -> str:
    return f"{_FONTS_HTML}\n  <style>{load_report_css()}</style>"


def render_theme_toggle() -> str:
    return f"{_THEME_TOGGLE_BUTTON}\n\n  <script>\n{load_theme_toggle_js()}\n  </script>"


def vega_script_tags(indent: str = "  ") -> str:
    """CDN script tags for the Vega libraries matching the installed Altair.

    Altair writes specs against its own Vega-Lite schema; loading an older
    major version to render them drops or misreads newer properties.
    """
    # Altair 5 and 6 both export these; the fallbacks are Altair 5's values.
    vega = getattr(alt, "VEGA_VERSION", "5")
    vega_lite = getattr(alt, "VEGALITE_VERSION", alt.SCHEMA_VERSION.lstrip("v"))
    embed = getattr(alt, "VEGAEMBED_VERSION", "6")
    return "\n".join(
        f'{indent}<script src="https://cdn.jsdelivr.net/npm/{name}@{version}"></script>'
        for name, version in (
            ("vega", vega),
            ("vega-lite", vega_lite),
            ("vega-embed", embed),
        )
    )


def json_for_script(text: str) -> str:
    """Make JSON text safe to place inside an HTML ``<script>`` element.

    Chart data carries user-defined names: a literal ``</script>`` in one
    would end the element and run whatever follows.  The escapes are still
    the same JSON, since ``<``, ``>`` and ``&`` only occur inside strings.
    """
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
