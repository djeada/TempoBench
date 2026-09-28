"""Big-O model selection.

1. Sort data by input size and rule out effectively-constant series.
2. For <= 2 points, or for 3 points with low dynamic range, fall back to the
   empirical log-log slope because richer model comparison is underdetermined.
3. For larger series, fit every non-constant candidate by relative-error least
   squares and rank them by the AIC of those same residuals, so the fit and the
   score agree on what a good curve is.
4. Among classes within `_STEP_DOWN_AIC_TOL` of the winner — ones the data
   cannot tell apart — step down to the simpler class when the growth between
   the two largest inputs, net of the fitted overhead, is closer to it.  A class
   the data clearly rejects is never chosen.
"""

from __future__ import annotations

import math
from typing import Sequence

from .fitting import (
    _is_effectively_constant,
    _log_log_slope,
    _model_score,
    _slope_to_model,
    _tail_ratio_favors_simpler,
    _wls_fit,
)
from .models import _MODEL_ORDER, _basis_functions

_LOW_DYNAMIC_RANGE_MAX = 5.0
_STEP_DOWN_AIC_TOL = 2.0


def runner_up(selected: str, scores: dict[str, float]) -> tuple[str | None, float]:
    """Return the best rival class and how far behind the winner it scored.

    The margin is ``rival_score - selected_score`` in AIC units, so it is
    positive when the selected class really did score best.  It goes negative —
    by less than `_STEP_DOWN_AIC_TOL` — when the tail check preferred a simpler
    class the score could not separate from the winner.
    Returns ``(None, inf)`` when there was nothing to compare against.
    """
    rivals = {model: score for model, score in scores.items() if model != selected}
    if not rivals or selected not in scores:
        return None, float("inf")
    best = min(rivals, key=lambda model: rivals[model])
    return best, rivals[best] - scores[selected]


def _select_model(x: Sequence[float], y: Sequence[float]) -> str:
    """Select the best Big-O complexity class."""
    return _rank_models(x, y)[0]


def _rank_models(
    x: Sequence[float], y: Sequence[float]
) -> tuple[str, dict[str, float]]:
    """Select a class and return the scores every candidate received.

    The scores let callers see how much better the winner was than the next
    class.  When two classes score within noise of each other the choice is a
    coin flip that a single label would hide, so the margin is reported rather
    than discarded.  Returns ``(selected, {model: AIC score})``; the score map
    is empty for series too short for model comparison.  Scores are AIC values
    of relative-error fits (see `fitting._model_score`).
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        return "O(1)", {}

    pairs = sorted(zip(x, y))
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    if _is_effectively_constant(y, x):
        return "O(1)", {}

    positive_series = all(v > 0 for v in x) and all(v > 0 for v in y)
    slope_hint = (
        _slope_to_model(_log_log_slope(x, y)) if positive_series else "O(n)"
    )

    if len(x) <= 2:
        return (slope_hint if positive_series else "O(n)"), {}

    dynamic_range = max(y) / min(y) if positive_series else float("inf")
    if len(x) == 3 and dynamic_range < _LOW_DYNAMIC_RANGE_MAX:
        return (slope_hint if positive_series else "O(n)"), {}

    bases = _basis_functions()
    candidates = {}
    for model in _MODEL_ORDER[1:]:
        score = _model_score(x, y, bases[model])
        if math.isfinite(score):
            candidates[model] = score
    # Exponential bases can overfit short polynomial series. Only admit this
    # candidate when observed growth is already beyond the polynomial range.
    if positive_series and _log_log_slope(x, y) < 3.2:
        candidates.pop("O(n² 2^n)", None)
    if not candidates:
        return (slope_hint if positive_series else "O(n)"), {}

    selected = min(candidates, key=lambda model: candidates[model])
    best_score = candidates[selected]
    selected_idx = _MODEL_ORDER.index(selected)

    # Only a class the score cannot separate from the winner may replace it:
    # the tail check breaks ties, it does not overrule the evidence.
    while selected_idx > 1:
        simpler = _MODEL_ORDER[selected_idx - 1]
        current = _MODEL_ORDER[selected_idx]
        if candidates.get(simpler, math.inf) > best_score + _STEP_DOWN_AIC_TOL:
            break
        _, simpler_baseline, _ = _wls_fit(x, y, bases[simpler])
        if not _tail_ratio_favors_simpler(x, y, simpler, current, simpler_baseline):
            break
        selected_idx -= 1

    return _MODEL_ORDER[selected_idx], candidates
