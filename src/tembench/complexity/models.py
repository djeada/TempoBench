"""Candidate complexity classes and their basis functions."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

_MODEL_ORDER = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)", "O(n³)"]

_ALL_MODELS: List[Tuple[str, Callable[[float], float]]] = [
    ("O(1)", lambda n: 1.0),
    ("O(log n)", lambda n: math.log(max(n, 2))),
    ("O(n)", lambda n: float(n)),
    ("O(n log n)", lambda n: float(n) * math.log(max(n, 2))),
    ("O(n²)", lambda n: float(n) ** 2),
    ("O(n³)", lambda n: float(n) ** 3),
]


def _basis_functions() -> Dict[str, Callable[[float], float]]:
    return dict(_ALL_MODELS)


_BASIS_STR = {
    "O(1)": "1",
    "O(log n)": "log(n)",
    "O(n)": "n",
    "O(n log n)": "n·log(n)",
    "O(n²)": "n²",
    "O(n³)": "n³",
}
