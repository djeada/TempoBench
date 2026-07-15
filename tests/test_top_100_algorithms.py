from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

import pandas as pd  # noqa: E402
from top_100_algorithms.catalog import ALGORITHMS  # noqa: E402
from top_100_algorithms.probes import count_python_steps  # noqa: E402
from top_100_algorithms.verify import SIZES, add_fits  # noqa: E402


def test_catalog_contains_exactly_100_named_algorithms():
    assert len(ALGORITHMS) == 100
    assert len(set(ALGORITHMS)) == 100


def test_every_demo_executes_and_has_declared_assumptions():
    for spec in ALGORITHMS.values():
        result, steps = count_python_steps(spec.run, 4)
        assert isinstance(result, int)
        assert steps > 0
        assert spec.assumption
        assert spec.expected in SIZES


def test_deterministic_step_curves_match_all_expected_classes():
    rows = []
    for spec in ALGORITHMS.values():
        for n in SIZES[spec.expected]:
            _, steps = count_python_steps(spec.run, n)
            rows.append(
                {
                    "algorithm": spec.name,
                    "category": spec.category,
                    "expected": spec.expected,
                    "assumption": spec.assumption,
                    "n": n,
                    "python_steps": steps,
                    # Keep this test deterministic; wall-clock probing is the
                    # responsibility of the end-to-end verifier.
                    "runtime_ns": steps,
                }
            )
    report = add_fits(pd.DataFrame(rows))
    assert report["steps_match"].all(), report.loc[~report["steps_match"]]
