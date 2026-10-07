"""Candidate complexity classes and their basis functions."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

_MODEL_ORDER = ["O(1)", "O(log n)", "O(√n)", "O(n)", "O(n log n)", "O(n²)", "O(n³)", "O(2^n)", "O(n² 2^n)"]

#: Classes whose cost doubles with every unit of n.
EXPONENTIAL_MODELS = frozenset({"O(2^n)", "O(n² 2^n)"})
#: Largest input size at which an exponential class is considered.  Nothing
#: exponential finishes far beyond it, and over a sweep that reaches further
#: the exponential basis is flat at every size but the last.
MAX_EXPONENTIAL_N = 64.0


def _exp2(n: float) -> float:
    # Clamped so the basis stays finite; sizes this large are never fitted
    # with an exponential class (see MAX_EXPONENTIAL_N).
    return 2.0 ** min(max(float(n), 0.0), 400.0)

_ALL_MODELS: List[Tuple[str, Callable[[float], float]]] = [
    ("O(1)", lambda n: 1.0),
    # log(1) = 0 is the whole point of the log classes: clamping at 2 instead
    # lifts the n=1 point and makes exact log/n·log series look like √n or n.
    ("O(log n)", lambda n: math.log(max(n, 1))),
    ("O(√n)", lambda n: math.sqrt(max(n, 0))),
    ("O(n)", lambda n: float(n)),
    ("O(n log n)", lambda n: float(n) * math.log(max(n, 1))),
    ("O(n²)", lambda n: float(n) ** 2),
    ("O(n³)", lambda n: float(n) ** 3),
    ("O(2^n)", _exp2),
    ("O(n² 2^n)", lambda n: float(n) ** 2 * _exp2(n)),
]


def _basis_functions() -> Dict[str, Callable[[float], float]]:
    return dict(_ALL_MODELS)


_BASIS_STR = {
    "O(1)": "1",
    "O(log n)": "log(n)",
    "O(√n)": "√n",
    "O(n)": "n",
    "O(n log n)": "n·log(n)",
    "O(n²)": "n²",
    "O(n³)": "n³",
    "O(2^n)": "2^n",
    "O(n² 2^n)": "n²·2^n",
}
