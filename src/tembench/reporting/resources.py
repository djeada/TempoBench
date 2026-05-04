"""Packaged HTML/CSS/JS assets used by HTML reports."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

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
