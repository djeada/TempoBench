"""Prove the built binary actually works, not just that it starts.

`--help` cannot tell a working binary from one that lost its data files: it
never touches the packaged CSS/JS or altair's bundled schemas. Rendering a
chart and a report does, which is how the missing --include-package-data was
caught. Both build.yml and release.yml run this same file so the release path
is exercised on every pull request instead of first running on a tag.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from find_binary import find_binary  # noqa: E402

SUMMARY = (
    "bench,impl,n,time_ms_median,time_ms_count,time_ms_p10,time_ms_p90\n"
    "b,a,100,1.0,5,0.9,1.1\n"
    "b,a,200,2.1,5,2.0,2.2\n"
    "b,a,400,4.0,5,3.9,4.1\n"
    "b,a,800,8.2,5,8.0,8.4\n"
)


def main() -> int:
    exe = find_binary()
    print(f"Testing {exe}", flush=True)

    # Windows runners default to a legacy code page, and Rich's console
    # detection is unreliable under captured pipes; pin both so a failure here
    # means the binary is broken rather than the terminal is.
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "NO_COLOR": "1",
    }

    def run(*args: str) -> int:
        print(f"\n$ {exe} {' '.join(args)}", flush=True)
        result = subprocess.run(
            [str(exe), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            env=env,
        )
        # Print both streams whatever happens: when this runs on a tag there is
        # no second chance to reproduce it, and a bare exit code says nothing.
        print(f"--- exit {result.returncode} ---", flush=True)
        print("--- stdout ---", flush=True)
        print(result.stdout[-6000:], flush=True)
        print("--- stderr ---", flush=True)
        print(result.stderr[-6000:], flush=True)
        return result.returncode

    if run("--help") != 0:
        print("FAIL: the binary does not start.", flush=True)
        return 1

    work = Path("smoke")
    work.mkdir(exist_ok=True)
    summary = work / "summary.csv"
    summary.write_text(SUMMARY, encoding="utf-8")

    chart = work / "runtime.html"
    if run("plot", "--summary", str(summary), "--out-html", str(chart)) != 0:
        print("FAIL: chart generation — altair data files are likely missing.", flush=True)
        return 1
    if not chart.is_file() or chart.stat().st_size == 0:
        print(f"FAIL: {chart} was not written.", flush=True)
        return 1

    report = work / "report.html"
    if run("report", "--summary", str(summary), "--output", str(report)) != 0:
        print("FAIL: report generation — packaged assets are likely missing.", flush=True)
        return 1

    html = report.read_text(encoding="utf-8")
    missing = [m for m in ("sysinfo-grid", "theme-toggle") if m not in html]
    if missing:
        print(f"FAIL: report is missing packaged assets: {missing}", flush=True)
        return 1

    print(f"\nOK: report rendered, {len(html)} bytes, assets present.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
