"""Provenance snapshot writer."""

from __future__ import annotations

import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path


def write_provenance(out_dir: Path, seed: int, workers: int) -> None:
    prov = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "cmdline": " ".join(shlex.quote(a) for a in sys.argv),
        "workers": workers,
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
