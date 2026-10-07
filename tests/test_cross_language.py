"""The cross-language example's three implementations must do identical work.

Fitting the same class in C++, Rust and Python only means something when the
three programs run the same algorithm on the same input, so every algorithm
must print the same checksum in every language.  Timings are not checked here:
`examples/cross_language/run_all.py` does that on a quiet machine.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tembench.config import load_config
from tembench.runner.reported import parse_reported_ms

EXAMPLE = Path(__file__).parents[1] / "examples" / "cross_language"
SIZES = {
    "binary_search": [1, 5, 64],
    "divisor_count": [1, 36, 97, 10**10],
    "max_subarray": [1, 2, 100],
    "merge_sort": [1, 2, 7, 100],
    "insertion_sort": [1, 2, 7, 100],
    "matrix_multiply": [1, 3, 16],
    "held_karp": [1, 2, 5, 9],
}
COMPILERS = {
    "cpp": ("g++", ["-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror"], "cpp"),
    "rust": ("rustc", ["-C", "opt-level=3", "-D", "warnings"], "rs"),
}


@pytest.fixture(scope="module")
def programs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, list[str]]]:
    """Every implementation, keyed by algorithm then language, ready to run."""
    for compiler, _, _ in COMPILERS.values():
        if shutil.which(compiler) is None:
            pytest.skip(f"{compiler} is not installed")
    out = tmp_path_factory.mktemp("build")

    def build(algo: str, language: str) -> list[str]:
        compiler, flags, ext = COMPILERS[language]
        binary = out / f"{algo}_{language}"
        source = EXAMPLE / algo / f"{algo}.{ext}"
        subprocess.run([compiler, *flags, "-o", str(binary), str(source)], check=True, capture_output=True)
        return [str(binary)]

    jobs = [(algo, language) for algo in SIZES for language in COMPILERS]
    with ThreadPoolExecutor() as pool:
        built = dict(zip(jobs, pool.map(lambda job: build(*job), jobs)))
    return {
        algo: {
            "cpp": built[(algo, "cpp")],
            "rust": built[(algo, "rust")],
            "python": [sys.executable, str(EXAMPLE / algo / f"{algo}.py")],
        }
        for algo in SIZES
    }


def _run(argv: list[str], n: int) -> tuple[str, float | None]:
    proc = subprocess.run(
        [*argv, "--n", str(n), "--min-ms", "0"], check=True, capture_output=True, text=True, timeout=60
    )
    match = re.search(r"^CHECKSUM: (\S+)$", proc.stdout, re.MULTILINE)
    assert match, proc.stdout
    return match.group(1), parse_reported_ms(proc.stdout)


@pytest.mark.parametrize("algo", sorted(SIZES))
def test_every_language_computes_the_same_result(programs, algo):
    for n in SIZES[algo]:
        results = {lang: _run(argv, n) for lang, argv in programs[algo].items()}
        checksums = {lang: checksum for lang, (checksum, _) in results.items()}
        assert len(set(checksums.values())) == 1, f"{algo} n={n}: {checksums}"
        assert all(ms is not None for _, ms in results.values())


def test_known_answers(programs):
    # Spot-check the shared checksum against values worked out by hand.
    assert _run(programs["divisor_count"]["cpp"], 36)[0] == "9"
    assert _run(programs["divisor_count"]["cpp"], 97)[0] == "2"
    assert _run(programs["held_karp"]["cpp"], 1)[0] == "0"


def test_each_algorithm_has_its_own_folder_and_config():
    folders = sorted(p for p in EXAMPLE.iterdir() if (p / "benchmark.yaml").exists())
    assert [p.name for p in folders] == sorted(SIZES)
    for folder in folders:
        algo = folder.name
        assert sorted(p.name for p in folder.iterdir() if p.suffix in (".cpp", ".rs", ".py")) == [
            f"{algo}.cpp", f"{algo}.py", f"{algo}.rs",
        ]
        cfg = load_config(folder / "benchmark.yaml")
        assert [bench.name for bench in cfg.benchmarks] == ["cpp", "rust", "python"]
        assert cfg.limits.metric == "reported"
        assert all(f"{algo}/{algo}." in (b.build or b.cmd) for b in cfg.benchmarks)
