"""Benchmark seven algorithms in C++, Rust and Python and check they agree.

For every algorithm config in this directory this runs the full TempoBench
pipeline (run, summarize, plot with fits, report) and then checks two things:

* every language printed the same CHECKSUM at every input size, so all three
  really ran the same algorithm on the same input;
* every language was fitted the same complexity class, and it is the class the
  algorithm is known to have.

A language is tens of times slower or faster than another; its complexity
class must not change.  Run from the repository root:

    python examples/cross_language/run_all.py                 # all seven
    python examples/cross_language/run_all.py merge_sort      # just one
    python examples/cross_language/run_all.py --out-dir artifacts/xl
    python examples/cross_language/run_all.py merge_sort --reels  # + a video

Exits non-zero when a checksum or a class disagrees.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from tembench.reel import pretty_model

HERE = Path(__file__).resolve().parent
EXPECTED = {
    "binary_search": "O(log n)",
    "divisor_count": "O(√n)",
    "max_subarray": "O(n)",
    "merge_sort": "O(n log n)",
    "insertion_sort": "O(n²)",
    "matrix_multiply": "O(n³)",
    "held_karp": "O(n² 2^n)",
}
TITLES = {
    "binary_search": "Binary search",
    "divisor_count": "Counting divisors",
    "max_subarray": "Maximum subarray",
    "merge_sort": "Merge sort",
    "insertion_sort": "Insertion sort",
    "matrix_multiply": "Matrix multiplication",
    "held_karp": "Travelling salesman",
}
LANGUAGES = ("cpp", "rust", "python")
LANGUAGE_NAMES = {"cpp": ("C++", "cpp"), "rust": ("Rust", "rs"), "python": ("Python", "py")}
#: Where the implementations are browsable once this branch is on main.
SOURCE_URL = "https://github.com/djeada/TempoBench/blob/main/examples/cross_language"
CHECKSUM_RE = re.compile(r"^CHECKSUM: (\S+)$", re.MULTILINE)
CONFIDENCE_STYLE = {"high": "green", "medium": "yellow", "low": "red"}

console = Console()


def tembench(*args: str | Path) -> None:
    """Run one tembench command, failing loudly with its output if it fails."""
    cmd = [sys.executable, "-m", "tembench", *map(str, args)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        console.print(proc.stdout, proc.stderr, markup=False)
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def checksum_disagreements(runs_path: Path) -> list[str]:
    """Return one line per input size where the languages printed different checksums."""
    seen: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    with runs_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            match = CHECKSUM_RE.search(record.get("stdout") or "")
            if record.get("status") == "ok" and match:
                seen[int(record["params"]["n"])][record["bench"]].add(match.group(1))
    problems = []
    for n, by_language in sorted(seen.items()):
        values = {value for checksums in by_language.values() for value in checksums}
        if len(values) != 1 or set(by_language) != set(LANGUAGES):
            detail = ", ".join(f"{lang}={sorted(by_language.get(lang, set()))}" for lang in LANGUAGES)
            problems.append(f"n={n}: {detail}")
    return problems


def run_algorithm(algo: str, out_dir: Path, reel: bool = False) -> dict[str, dict[str, str]]:
    """Run the pipeline for one algorithm; return its fits keyed by language."""
    out = out_dir / algo
    tembench("run", "--config", HERE / algo / "benchmark.yaml", "--out-dir", out, "--quiet")
    tembench("summarize", "--runs", out / "runs.jsonl", "--out-csv", out / "summary.csv")
    tembench(
        "plot", "--summary", out / "summary.csv", "--out-html", out / "runtime.html",
        "--export-fits", out / "fits.csv",
    )
    tembench("report", "--summary", out / "summary.csv", "--output", out / "report.html")
    with (out / "fits.csv").open(encoding="utf-8") as handle:
        fits = {row["bench"]: row for row in csv.DictReader(handle)}
    if reel:
        tembench(
            "reel", "--summary", out / "summary.csv", "--title", TITLES[algo],
            "--output", out / "reel.mp4", "--poster", out / "reel.png",
        )
        (out / "caption.txt").write_text(caption(algo, fits), encoding="utf-8")
    return fits


def caption(algo: str, fits: dict[str, dict[str, str]]) -> str:
    """A description to post with the reel: the result, and the code behind it."""
    classes = {pretty_model(fits[lang]["model"]) for lang in LANGUAGES if lang in fits}
    names = ", ".join(LANGUAGE_NAMES[lang][0] for lang in LANGUAGES[:-1]) + f" and {LANGUAGE_NAMES[LANGUAGES[-1]][0]}"
    if len(classes) == 1:
        result = f"{TITLES[algo]} in {names}: all three grow as {classes.pop()}."
    else:
        result = f"{TITLES[algo]} in {names}: " + ", ".join(
            f"{LANGUAGE_NAMES[lang][0]} {pretty_model(fits[lang]['model'])}" for lang in LANGUAGES if lang in fits
        ) + "."
    links = [f"{LANGUAGE_NAMES[lang][0]}: {SOURCE_URL}/{algo}/{algo}.{LANGUAGE_NAMES[lang][1]}" for lang in LANGUAGES]
    return "\n".join([result, "", *links, "", "Measured and fitted with TempoBench: https://github.com/djeada/TempoBench", ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("algorithms", nargs="*", choices=list(EXPECTED), metavar="ALGO",
                        help=f"algorithms to run (default: all of {', '.join(EXPECTED)})")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/cross_language"))
    parser.add_argument("--reels", action="store_true",
                        help="also render a short video per algorithm (needs matplotlib and ffmpeg)")
    args = parser.parse_args()

    table = Table(title="Cross-language complexity", title_style="bold")
    table.add_column("Algorithm")
    table.add_column("Expected")
    for language in LANGUAGES:
        table.add_column(language)
    table.add_column("Checksums")

    failures: list[str] = []
    for algo in args.algorithms or list(EXPECTED):
        with console.status(f"Benchmarking {algo} in {', '.join(LANGUAGES)}"):
            fits = run_algorithm(algo, args.out_dir, args.reels)
        expected = EXPECTED[algo]
        cells = []
        for language in LANGUAGES:
            fit = fits.get(language)
            if fit is None:
                cells.append("[red]no fit[/red]")
                failures.append(f"{algo}: no fit for {language}")
                continue
            style = CONFIDENCE_STYLE.get(fit["confidence"], "dim")
            mark = "" if fit["model"] == expected else " [red]✗[/red]"
            cells.append(f"{fit['model']}{mark} [{style}]{fit['confidence']}[/{style}]")
            if fit["model"] != expected:
                failures.append(f"{algo}: {language} fitted {fit['model']}, expected {expected} "
                                f"({fit['confidence_notes'] or 'no caveats'})")
        bad_checksums = checksum_disagreements(args.out_dir / algo / "runs.jsonl")
        failures += [f"{algo}: checksums differ at {problem}" for problem in bad_checksums]
        table.add_row(algo, expected, *cells, "[red]differ[/red]" if bad_checksums else "[green]match[/green]")

    console.print(table)
    console.print(f"Reports: {args.out_dir}/<algorithm>/report.html")
    if args.reels:
        console.print(f"Reels:   {args.out_dir}/<algorithm>/reel.mp4, with caption.txt to post alongside")
    if failures:
        console.print("\n[red]Disagreements:[/red]")
        for failure in failures:
            console.print(f"  • {failure}", markup=False)
        return 1
    console.print("\n[green]All languages agree on every checksum and every complexity class.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
