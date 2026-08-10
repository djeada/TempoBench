"""Provenance snapshot writer.

Timings are only meaningful next to the machine that produced them, and that
machine is usually not the one someone later reads the report on — benchmarks
run in CI, results get analysed on a laptop.  The snapshot is therefore written
at run time and carries the hardware alongside the invocation, so downstream
commands can describe the run rather than themselves.
"""

from __future__ import annotations

import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..system import get_system_info

PROVENANCE_FILENAME = "provenance.json"


def write_provenance(
    out_dir: Path, seed: int, workers: int, append: bool = False
) -> None:
    """Record how and where this run was produced.

    Appending adds trials to an existing results file, so the file then spans
    several invocations — possibly on different machines.  Overwriting the
    snapshot would leave the results claiming to come from whichever run
    happened to finish last, so earlier snapshots are kept under ``previous``.
    """
    path = out_dir / PROVENANCE_FILENAME
    prov: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "cmdline": " ".join(shlex.quote(a) for a in sys.argv),
        "workers": workers,
        "system": get_system_info(),
    }

    if append:
        earlier = read_provenance(path)
        if earlier is not None:
            history = earlier.pop("previous", [])
            if not isinstance(history, list):
                history = []
            prov["previous"] = [*history, earlier]

    path.write_text(json.dumps(prov, indent=2))


def read_provenance(path: Path) -> dict | None:
    """Load a provenance snapshot, or None when it is absent or unreadable."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
