"""Parsing of self-reported timings emitted by the benchmarked command.

Wall-clock time of a whole process includes interpreter/runtime startup, image
loading, and argument parsing.  For small inputs that overhead routinely dwarfs
the work being measured, which makes the observed curve flat and any Big-O fit
derived from it meaningless.

To measure the workload rather than the process, a benchmarked command may time
its own hot section and print the result on stdout::

    TEMPOBENCH_MS: 12.345

The marker is deliberately trivial so that any language can emit it.  It is
recognised case-sensitively, may use ``:``, ``=`` or whitespace as a separator,
and must be alone on its line.  When a command prints it more than once, the
last occurrence wins, so a program may report intermediate progress.
"""

from __future__ import annotations

import re

MARKER_NAME = "TEMPOBENCH_MS"

_MARKER_RE = re.compile(
    rf"^[^\S\n]*{MARKER_NAME}[^\S\n]*[:=]?[^\S\n]*"
    r"([0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?|\.[0-9]+(?:[eE][-+]?[0-9]+)?)"
    r"[^\S\n]*$",
    re.MULTILINE,
)


def parse_reported_ms(stdout: str | None) -> float | None:
    """Return the last self-reported duration in milliseconds, if any.

    Returns None when the marker is absent or its value is not a finite,
    non-negative number.
    """
    if not stdout or MARKER_NAME not in stdout:
        return None

    for match in reversed(_MARKER_RE.findall(stdout)):
        try:
            value = float(match)
        except ValueError:
            continue
        if value == value and value not in (float("inf"), float("-inf")) and value >= 0:
            return value
    return None
