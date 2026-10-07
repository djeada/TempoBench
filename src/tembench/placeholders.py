"""Command templates: grid values and built-in placeholders.

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
    return {"python": sys.executable}


BUILTIN_PLACEHOLDER_NAMES = frozenset(builtin_placeholders())


class _TemplateValue:
    """A value that quotes itself when substituted into a command.

    Commands are split into arguments and launched directly, so an unquoted
    ``two words`` would become two arguments and ``it's`` would not parse at
    all.  A value is therefore always exactly one argument.  Format specs still
    apply (``{n:05d}``), and ``{key:raw}`` splices the value in verbatim for a
    grid that holds whole command lines.
    """

    __slots__ = ("value",)

    def __init__(self, value: object):
        self.value = value

    def __format__(self, spec: str) -> str:
        if spec == "raw":
            return str(self.value)
        return quote_argument(format(self.value, spec))

    def __getitem__(self, key: Any) -> _TemplateValue:
        return _TemplateValue(self.value[key])  # type: ignore[index]

    def __getattr__(self, name: str) -> _TemplateValue:
        return _TemplateValue(getattr(self.value, name))


def format_cmd(template: str, params: dict[str, object]) -> str:
    """Expand a command template from the grid point plus built-in placeholders.

    Grid keys take precedence, so a sweep may shadow a built-in.
    """
    values = {**builtin_placeholders(), **params}
    return template.format(**{key: _TemplateValue(value) for key, value in values.items()})
