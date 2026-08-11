"""Lay the built binary out under release/ the way the release attaches it.

Run by build.yml as well as release.yml: packaging that only ever runs on a tag
is packaging nobody has tested.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from find_binary import find_binary  # noqa: E402


def main() -> int:
    runner_os = os.environ.get("RUNNER_OS", "Linux")
    exe = find_binary(windows=runner_os == "Windows")
    release_dir = Path("release")
    release_dir.mkdir(exist_ok=True)

    # Windows builds standalone: the executable is useless without the DLLs and
    # data directory beside it, so the whole directory ships as a zip.
    if runner_os == "Windows":
        standalone_dir = exe.parent
        archive = shutil.make_archive(
            str(release_dir / "tembench-windows"),
            "zip",
            root_dir=str(standalone_dir.parent),
            base_dir=standalone_dir.name,
        )
        print(f"Packaged artifact: {archive} ({Path(archive).stat().st_size} bytes)")
        return 0

    # Linux and macOS build onefile: one self-contained executable, renamed here
    # from the build-time name it needed to avoid colliding with its data dir.
    dst = release_dir / f"tembench-{runner_os.lower()}"
    shutil.copy2(exe, dst)
    dst.chmod(0o755)
    print(f"Packaged artifact: {dst} ({dst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
