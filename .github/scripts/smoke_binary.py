"""Prove the built binary actually works, not just that it starts.

`--help` cannot tell a working binary from one that lost its data files: it
never touches the packaged CSS/JS or altair's bundled schemas. Rendering a
chart and a report does, which is how the missing --include-package-data was
caught. Both build.yml and release.yml run this same file so the release path
is exercised on every pull request instead of first running on a tag.

Each stage runs as its own workflow step. Job steps are readable through the
public API while job *logs* need a token, so one named step per stage means the
failing stage is identifiable from outside the repo. Debugging a Windows-only
failure with no log access is otherwise guesswork.

Usage: smoke_binary.py {help|plot|report}
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

WORK = Path("smoke")
# Job logs need a token to read, so a failure on a platform the maintainer
# cannot reproduce is otherwise invisible from outside the repo. Everything
# printed here is also appended to this file, which the workflow posts as a
# commit comment on failure -- commit comments are readable without auth.
DIAGNOSTICS = Path("smoke-diagnostics.txt")


def _log(text: str = "") -> None:
    print(text, flush=True)
    with DIAGNOSTICS.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def _env() -> dict[str, str]:
    # Windows runners default to a legacy code page, and Rich's console
    # detection is unreliable under captured pipes; pin both so a failure means
    # the binary is broken rather than the terminal is.
    return {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "NO_COLOR": "1",
        "TERM": "dumb",
        "COLUMNS": "120",
    }


def _run(exe: Path, *args: str) -> int:
    _log(f"$ {exe} {' '.join(args)}")
    try:
        result = subprocess.run(
            [str(exe), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            env=_env(),
        )
    except OSError as exc:
        # The binary exists but the OS refused to run it at all: a missing
        # runtime DLL on Windows shows up here, not as a Python traceback.
        _log(f"--- could not execute: {exc!r} ---")
        return 1
    # Log both streams whatever happens: on a tag there is no second chance to
    # reproduce it, and a bare exit code says nothing about why.
    _log(f"--- exit {result.returncode} ---")
    _log("--- stdout ---")
    _log(result.stdout[-6000:])
    _log("--- stderr ---")
    _log(result.stderr[-6000:])
    return result.returncode


def _summary() -> Path:
    WORK.mkdir(exist_ok=True)
    path = WORK / "summary.csv"
    path.write_text(SUMMARY, encoding="utf-8")
    return path


def stage_help(exe: Path) -> int:
    if _run(exe, "--help") == 0:
        return 0
    # Narrow the failure down in the same run rather than over several. These
    # separate whether the embedded runtime is broken (nothing works) from
    # whether it is specifically Rich's help rendering, which draws box
    # characters a legacy Windows code page cannot encode.
    _log("\n=== narrowing down: does anything else run? ===")
    _log("\n[probe] sysinfo — starts the runtime and prints a Rich table")
    _run(exe, "sysinfo")
    _log("\n[probe] plot --help — subcommand help, a smaller Rich panel")
    _run(exe, "plot", "--help")
    _log("\n[probe] no arguments — Typer prints usage and exits non-zero by design")
    _run(exe)
    _log("FAIL: the binary does not start.")
    return 1


def stage_plot(exe: Path) -> int:
    chart = WORK / "runtime.html"
    if _run(exe, "plot", "--summary", str(_summary()), "--out-html", str(chart)) != 0:
        _log("FAIL: chart generation — altair data files are likely missing.")
        return 1
    if not chart.is_file() or chart.stat().st_size == 0:
        _log(f"FAIL: {chart} was not written.")
        return 1
    _log(f"OK: chart rendered, {chart.stat().st_size} bytes.")
    return 0


def stage_report(exe: Path) -> int:
    report = WORK / "report.html"
    if _run(exe, "report", "--summary", str(_summary()), "--output", str(report)) != 0:
        _log("FAIL: report generation — packaged assets are likely missing.")
        return 1
    html = report.read_text(encoding="utf-8")
    missing = [m for m in ("sysinfo-grid", "theme-toggle") if m not in html]
    if missing:
        _log(f"FAIL: report is missing packaged assets: {missing}")
        return 1
    _log(f"OK: report rendered, {len(html)} bytes, assets present.")
    return 0


STAGES = {"help": stage_help, "plot": stage_plot, "report": stage_report}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in STAGES:
        print(f"usage: {argv[0]} {{{'|'.join(STAGES)}}}", flush=True)
        return 2
    return STAGES[argv[1]](find_binary())


if __name__ == "__main__":
    sys.exit(main(sys.argv))
