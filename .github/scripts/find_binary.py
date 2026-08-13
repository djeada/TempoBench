"""Locate the executable Nuitka just built, on any of the three platforms.

Nuitka names the output *directory* after the entry module and the output
*file* after --output-filename, so `--output-filename=tembench` on
build/tembench_entry.py lands in `dist/tembench_entry.dist/tembench.exe`.
Guessing that layout is what broke the v0.1.0 release, so search for the file
instead and print the whole tree when it is missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def on_windows() -> bool:
    """Resolved per call, not at import: CI sets RUNNER_OS in the environment."""
    return os.environ.get("RUNNER_OS", "") == "Windows" or os.name == "nt"


def binary_name(windows: bool | None = None) -> str:
    return "tembench.exe" if (on_windows() if windows is None else windows) else "tembench.bin"


def find_binary(dist: Path = Path("dist"), windows: bool | None = None) -> Path:
    """Return the built executable, or exit non-zero describing what is there."""
    wanted = binary_name(windows)
    # An onefile build leaves the intermediate standalone tree behind, so
    # `dist/tembench.bin` and `dist/tembench_entry.dist/tembench.bin` can both
    # exist. Only the shallow one is self-contained; shipping the deep one
    # would produce a binary that dies without the directory it was cut from.
    # Prefer the shallowest match rather than relying on how the names sort.
    found = sorted(
        (p for p in dist.rglob(wanted) if p.is_file()),
        key=lambda p: (len(p.parts), p),
    )
    if not found:
        print(f"No {wanted} found under {dist}/. Contents:", flush=True)
        for item in sorted(dist.rglob("*")):
            print(f"  {item}", flush=True)
        sys.exit(1)
    if len(found) > 1:
        print(f"Multiple candidates for {wanted}, using the first:", flush=True)
        for item in found:
            print(f"  {item}", flush=True)
    return found[0]


if __name__ == "__main__":
    # Run as its own workflow step so "the binary was never found" is
    # distinguishable from "the binary was found and misbehaved" using only the
    # public jobs API, which reports step names but not log contents.
    print(f"Found: {find_binary()}", flush=True)
