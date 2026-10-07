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
from pathlib import Path

import pytest
import yaml

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


def _build(tmp_path_factory: pytest.TempPathFactory, compiler: str, args: list[str]) -> Path:
    if shutil.which(compiler) is None:
        pytest.skip(f"{compiler} is not installed")
    out = tmp_path_factory.mktemp("build") / compiler
    subprocess.run([compiler, *args, "-o", str(out)], check=True, capture_output=True)
    return out


@pytest.fixture(scope="module")
def programs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[str]]:
    cpp = _build(tmp_path_factory, "g++", ["-O2", "-std=c++17", str(EXAMPLE / "algorithms.cpp")])
    rust = _build(tmp_path_factory, "rustc", ["-C", "opt-level=3", str(EXAMPLE / "algorithms.rs")])
    return {
        "cpp": [str(cpp)],
        "rust": [str(rust)],
        "python": [sys.executable, str(EXAMPLE / "algorithms.py")],
    }


def _run(argv: list[str], algo: str, n: int) -> tuple[str, float | None]:
    proc = subprocess.run(
        [*argv, "--algo", algo, "--n", str(n), "--min-ms", "0"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    match = re.search(r"^CHECKSUM: (\S+)$", proc.stdout, re.MULTILINE)
    assert match, proc.stdout
    return match.group(1), parse_reported_ms(proc.stdout)


@pytest.mark.parametrize("algo", sorted(SIZES))
def test_every_language_computes_the_same_result(programs, algo):
    for n in SIZES[algo]:
        results = {lang: _run(argv, algo, n) for lang, argv in programs.items()}
        checksums = {lang: checksum for lang, (checksum, _) in results.items()}
        assert len(set(checksums.values())) == 1, f"{algo} n={n}: {checksums}"
        assert all(ms is not None for _, ms in results.values())


def test_known_answers(programs):
    # Spot-check the shared checksum against values worked out by hand.
    assert _run(programs["cpp"], "divisor_count", 36)[0] == "9"
    assert _run(programs["cpp"], "divisor_count", 97)[0] == "2"
    assert _run(programs["cpp"], "held_karp", 1)[0] == "0"


def test_configs_cover_every_algorithm_in_every_language():
    configs = sorted(EXAMPLE.glob("*.yaml"))
    assert {path.stem for path in configs} == set(SIZES)
    for path in configs:
        cfg = load_config(path)
        assert [bench.name for bench in cfg.benchmarks] == ["cpp", "rust", "python"]
        assert cfg.limits.metric == "reported"
        assert yaml.safe_load(path.read_text())["benchmarks"][0]["cmd"].endswith(f"--algo {path.stem} --n {{n}}")
