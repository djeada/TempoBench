"""How much a fitted complexity class can actually be believed.

Fitting always returns a class.  Whether that class means anything depends on
the measurements it was derived from: four points spanning a factor of two in
``n``, or a series where most of every reading is fixed process overhead, will
produce a confident-looking label from data that cannot distinguish O(n) from
O(n²).  Reporting the class without that context is the difference between a
measurement and a guess, so every fit carries the caveats that apply to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

#: Fewer points than this cannot separate neighbouring complexity classes.
MIN_POINTS = 4
#: Input sizes must span at least this factor for growth to be observable.
MIN_N_RATIO = 8.0
#: Durations must span at least this factor, or nothing grew enough to fit.
MIN_Y_RATIO = 2.0
#: A bootstrap exponent interval wider than this spans whole classes.
MAX_EXPONENT_CI_WIDTH = 0.5
#: Fraction of the largest reading that may be constant overhead.
MAX_OVERHEAD_SHARE = 0.5
#: Widest p10-p90 gap, as a fraction of the median, that still counts as a
#: repeatable measurement.  Beyond it the machine was doing something else.
MAX_RELATIVE_SPREAD = 0.5
#: Trials per input size below which a "median" is not really a median.
MIN_SAMPLES = 3
#: Log-AIC lead the chosen class needs over the next one to count as decided.
#: Two is the conventional threshold for "substantially supported" in AIC model
#: comparison; below it, the runner-up explains the data about as well.
MIN_MODEL_MARGIN = 2.0

_NOTE_TEXT = {
    "few-points": f"fewer than {MIN_POINTS} input sizes",
    "narrow-n-range": f"input sizes span less than {MIN_N_RATIO:g}x",
    "flat-signal": f"durations span less than {MIN_Y_RATIO:g}x",
    "wide-exponent-ci": "empirical exponent interval is wider than "
    f"{MAX_EXPONENT_CI_WIDTH:g}",
    "overhead-dominated": "more than "
    f"{MAX_OVERHEAD_SHARE:.0%} of the largest reading is constant overhead",
    "ambiguous-class": "another class explains the data about as well",
    "thin-samples": f"fewer than {MIN_SAMPLES} trials per input size",
    "unstable-timings": "repeated trials disagree by more than "
    f"{MAX_RELATIVE_SPREAD:.0%} of the median — was the machine busy?",
}


@dataclass(frozen=True)
class FitQuality:
    """Confidence rating for one fitted series, with the reasons behind it."""

    confidence: str
    notes: tuple[str, ...]

    @property
    def summary(self) -> str:
        """Render the caveats as a single human-readable clause."""
        return "; ".join(_NOTE_TEXT.get(note, note) for note in self.notes)


def _ratio(values: Sequence[float]) -> float:
    """Return max/min for a strictly positive series, else infinity."""
    positive = [v for v in values if v > 0]
    if not positive or len(positive) != len(values):
        return float("inf")
    return max(positive) / min(positive)


def assess_fit(
    x: Sequence[float],
    y: Sequence[float],
    *,
    effective_baseline: float,
    exponent_ci_low: float,
    exponent_ci_high: float,
    model_margin: float = float("inf"),
    min_samples: float | None = None,
    max_relative_spread: float | None = None,
) -> FitQuality:
    """Rate how well the measurements support the class that was selected.

    Each independent weakness contributes one caveat; a single caveat downgrades
    the fit to ``medium`` and two or more to ``low``.  The rating is deliberately
    a count rather than a weighted score: the point is to say *what* is wrong
    with the measurement so it can be fixed, not to produce a number.
    """
    notes: list[str] = []

    if len(x) < MIN_POINTS:
        notes.append("few-points")
    if _ratio(x) < MIN_N_RATIO:
        notes.append("narrow-n-range")
    if _ratio(y) < MIN_Y_RATIO:
        notes.append("flat-signal")

    ci_width = exponent_ci_high - exponent_ci_low
    if math.isfinite(ci_width) and ci_width > MAX_EXPONENT_CI_WIDTH:
        notes.append("wide-exponent-ci")

    y_max = max(y) if y else 0.0
    if y_max > 0 and max(0.0, effective_baseline) / y_max > MAX_OVERHEAD_SHARE:
        notes.append("overhead-dominated")

    if model_margin < MIN_MODEL_MARGIN:
        notes.append("ambiguous-class")

    # Too few trials per point leaves noise that is systematic rather than
    # scattered — it bends the curve instead of widening the spread, so none of
    # the checks above can see it.
    if min_samples is not None and min_samples < MIN_SAMPLES:
        notes.append("thin-samples")

    # Trials of the same grid point that disagree wildly were not measuring the
    # same thing.  Nothing else here can see it: the medians can still trace a
    # smooth curve while every point behind them is unrepeatable.
    if max_relative_spread is not None and max_relative_spread > MAX_RELATIVE_SPREAD:
        notes.append("unstable-timings")

    if not notes:
        confidence = "high"
    elif len(notes) == 1:
        confidence = "medium"
    else:
        confidence = "low"
    return FitQuality(confidence=confidence, notes=tuple(notes))
