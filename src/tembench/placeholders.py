"""Built-in command-template placeholders.

Grid keys supply the sweep parameters, but a few values are properties of the
machine running the benchmark rather than of the sweep.  Hardcoding them makes
configs non-portable: ``python`` is not a command on most Linux installs, and
even where it is, it need not be the interpreter TempoBench was installed into.

These placeholders are always available and are shadowed by a grid key of the
same name, so a config can still sweep over e.g. several interpreters.
"""

from __future__ import annotations

import sys
from typing import Any, Dict

from .command import quote_argument


def builtin_placeholders() -> Dict[str, Any]:
    """Return the placeholder values that do not come from the grid."""
    return {"python": quote_argument(sys.executable)}


BUILTIN_PLACEHOLDER_NAMES = frozenset(builtin_placeholders())
