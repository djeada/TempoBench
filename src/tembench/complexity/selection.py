"""Big-O model selection.

The selector now uses a scale-free workflow:
1. Sort data by input size and rule out effectively-constant series.
2. For <= 2 points, or for 3 points with low dynamic range, fall back to the
   empirical log-log slope because richer model comparison is underdetermined.
3. For larger series, fit every non-constant candidate in linear space, but
   compare them via log-space AIC-like scores so the largest-n point does not
   dominate purely by magnitude.
4. Apply a tail-ratio step-down guard between adjacent model pairs. This only
   overrides the log-AIC winner when the tail looks closer to the simpler
   class and that simpler class is either statistically competitive or matches
   the empirical slope hint.
"""

from __future__ import annotations

import math
from typing import Sequence

from .fitting import (
    _is_effectively_constant,
    _log_log_slope,
    _log_space_aic,
    _slope_to_model,
    _tail_ratio_favors_simpler,
)
from .models import _MODEL_ORDER, _basis_functions

_LOW_DYNAMIC_RANGE_MAX = 5.0
_STEP_DOWN_AIC_TOL = 2.0


def _select_model(x: Sequence[float], y: Sequence[float]) -> str:
    """Select the best Big-O complexity class."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        return "O(1)"

    pairs = sorted(zip(x, y))
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    if _is_effectively_constant(y, x):
        return "O(1)"

    positive_series = all(v > 0 for v in x) and all(v > 0 for v in y)
    slope_hint = (
        _slope_to_model(_log_log_slope(x, y)) if positive_series else "O(n)"
    )

    if len(x) <= 2:
        return slope_hint if positive_series else "O(n)"

    dynamic_range = max(y) / min(y) if positive_series else float("inf")
    if len(x) == 3 and dynamic_range < _LOW_DYNAMIC_RANGE_MAX:
        return slope_hint if positive_series else "O(n)"

    bases = _basis_functions()
    aic_by_model = {
        model: _log_space_aic(x, y, bases[model]) for model in _MODEL_ORDER[1:]
    }
    candidates = {
        model: score for model, score in aic_by_model.items() if math.isfinite(score)
    }
    if not candidates:
        return slope_hint if positive_series else "O(n)"

    selected = min(candidates, key=lambda model: candidates[model])
    selected_idx = _MODEL_ORDER.index(selected)
    slope_idx = _MODEL_ORDER.index(slope_hint)

    while selected_idx > 1:
        simpler = _MODEL_ORDER[selected_idx - 1]
        current = _MODEL_ORDER[selected_idx]
        if not _tail_ratio_favors_simpler(x, y, simpler, current):
            break

        simpler_score = aic_by_model.get(simpler, float("inf"))
        current_score = aic_by_model[current]
        if simpler_score <= current_score + _STEP_DOWN_AIC_TOL or (
            _MODEL_ORDER.index(simpler) == slope_idx
        ):
            selected_idx -= 1
            continue
        break

    return _MODEL_ORDER[selected_idx]
