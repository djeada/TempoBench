"""Turning a configured command string into something the OS can launch.

The two platforms disagree about what a command line even is.  POSIX shells
tokenise before `exec`, so a list of arguments is the honest representation.
Windows passes a single string to `CreateProcess` and lets the callee parse it,
and its path separator is the character POSIX quoting uses for escapes — so
splitting a Windows command with POSIX rules silently eats the backslashes and
turns ``C:\\tools\\prog.exe`` into ``C:toolsprog.exe``.
"""

from __future__ import annotations

import os
import shlex

WINDOWS = os.name == "nt"


def split_command(cmd: str) -> str | list[str]:
    """Return the command in the form `subprocess.Popen` expects on this OS."""
    if WINDOWS:
        # Popen accepts the raw command line on Windows and hands it to
        # CreateProcess, which applies the platform's own quoting rules.
        return cmd
    return shlex.split(cmd)


def quote_argument(value: str) -> str:
    """Quote a single argument so it survives substitution into a command."""
    if not WINDOWS:
        return shlex.quote(value)
    # POSIX single-quoting means nothing to CreateProcess; it wants double
    # quotes, and only where whitespace would otherwise split the argument.
    if value and not any(ch.isspace() or ch == '"' for ch in value):
        return value
    return '"' + value.replace('"', r"\"") + '"'
