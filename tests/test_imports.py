"""Every module must import on its own.

A circular import between two modules is invisible to the rest of the suite
whenever some earlier test happens to import them in the working order, so it
is checked explicitly, in a fresh interpreter, one module at a time.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

MODULES = [
    "tembench",
    "tembench.cli",
    "tembench.command",
    "tembench.complexity",
    "tembench.config",
    "tembench.placeholders",
    "tembench.plotting",
    "tembench.reporting",
    "tembench.runner",
    "tembench.summarize",
    "tembench.system",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_in_isolation(module: str):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{module} failed to import:\n{result.stderr}"
