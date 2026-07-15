from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

import pandas as pd  # noqa: E402
from top_100_algorithms.catalog import ALGORITHMS  # noqa: E402
from top_100_algorithms.implementations_advanced import (  # noqa: E402
    SELF_CHECKS as ADVANCED_CHECKS,
)
from top_100_algorithms.implementations_advanced import (
    run_all_named_self_checks as check_all_advanced,
)
from top_100_algorithms.implementations_advanced import (
    run_self_checks as check_advanced,
)
from top_100_algorithms.implementations_core import (  # noqa: E402
    SELF_CHECKS as CORE_CHECKS,
)
from top_100_algorithms.implementations_core import (
    run_all_named_self_checks as check_all_core,
)
from top_100_algorithms.implementations_core import (
    run_self_checks as check_core,
)
from top_100_algorithms.implementations_graph import (  # noqa: E402
    SELF_CHECKS as GRAPH_CHECKS,
)
from top_100_algorithms.implementations_graph import (
    run_all_named_self_checks as check_all_graph,
)
from top_100_algorithms.implementations_graph import (
    run_self_checks as check_graph,
)
from top_100_algorithms.probes import count_python_steps  # noqa: E402
from top_100_algorithms.verify import SIZES, add_fits  # noqa: E402


def test_catalog_contains_exactly_100_named_algorithms():
    assert len(ALGORITHMS) == 100
    assert len(set(ALGORITHMS)) == 100
    assert len({id(spec.run) for spec in ALGORITHMS.values()}) == 100
    assert all(
        spec.run.__module__.endswith(("implementations_core", "implementations_graph", "implementations_advanced"))
        for spec in ALGORITHMS.values()
    )


def test_canonical_correctness_checks_cover_all_implementation_families():
    check_core()
    check_graph()
    check_advanced()


def test_every_named_algorithm_has_and_passes_a_semantic_correctness_check():
    checks = {**CORE_CHECKS, **GRAPH_CHECKS, **ADVANCED_CHECKS}
    assert set(checks) == set(ALGORITHMS)
    assert len(checks) == 100
    check_all_core()
    check_all_graph()
    check_all_advanced()


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
