"""TempoBench CLI package.

The Typer `app` is built in `app.py`. Importing the `commands` subpackage
triggers each module's `@app.command()` decorator so the CLI surface is
populated as a side effect of `import tembench.cli`.
"""

from .app import app, console

# Import every command module so its @app.command() registers with `app`.
from .commands import (  # noqa: F401  (import-for-side-effects)
    compare,
    dashboard,
    heatmap,
    inspect,
    memory,
    plot,
    report,
    run,
    summarize,
    sysinfo,
)

__all__ = ["app", "console"]
