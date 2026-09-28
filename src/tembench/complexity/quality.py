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

from .fitting import _is_effectively_constant
from .formatting import _MODEL_SLOPE_INTERVALS

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
#: AIC lead the chosen class needs over the next one to count as decided.
#: Two is the textbook threshold for a single comparison, but every fit here
#: is a contest between seven classes on a handful of points: on synthetic
#: sweeps a wrong class led by 2-6 about as often as a right one, while a lead
#: of six ("strong" evidence) was almost never wrong.
MIN_MODEL_MARGIN = 6.0
#: Fitted overhead below this share of the smallest reading is too little to
#: depress the measured exponent, so it cannot excuse one too low for the class.
_OVERHEAD_FREE_SHARE = 0.05

_NOTE_TEXT = {
    "few-points": f"fewer than {MIN_POINTS} input sizes",
    "narrow-n-range": f"input sizes span less than {MIN_N_RATIO:g}x",
    "flat-signal": f"durations span less than {MIN_Y_RATIO:g}x",
    "wide-exponent-ci": "empirical exponent interval is wider than "
    f"{MAX_EXPONENT_CI_WIDTH:g}",
    "overhead-dominated": "more than "
    f"{MAX_OVERHEAD_SHARE:.0%} of the largest reading is constant overhead",
    "ambiguous-class": "another class fits almost as well "
    f"(AIC lead under {MIN_MODEL_MARGIN:g})",
    "exponent-mismatch": "the measured growth exponent lies outside the range "
    "this class produces",
    "single-point-growth": "all of the growth comes from the largest input "
    "size — one reading decides the class",
    "outlier-dropped": "constant only after discarding one outlying reading",
    "non-positive-timings": "some durations are zero or negative — below the "
    "timer's resolution?",
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
    """Return max/min over the positive values, or 1 when there are none.

    Zeros (an n=0 grid point, a reading below timer resolution) have no ratio.
    Treating them as infinite spread would switch off every range check they
    appear in, so they are left out; zero durations get their own caveat.
    """
    positive = [v for v in values if v > 0]
    if not positive:
        return 1.0
    return max(positive) / min(positive)


def _class_exponent_band(model: str, x: Sequence[float]) -> tuple[float, float]:
    """Return the log-log slopes the class can produce over these sizes.

    The fixed bands describe large n.  log n and n·log n have a local exponent
    of 1/ln n (plus one), which is far above its large-n value over n = 2..8,
    so their band is widened to what the class itself traces there.
    """
    lower, upper = _MODEL_SLOPE_INTERVALS.get(model, (-math.inf, math.inf))
    sizes = [v for v in x if v > 1]
    if model in ("O(log n)", "O(n log n)") and sizes:
        base = 1.0 if model == "O(n log n)" else 0.0
        lower = min(lower, base + 1.0 / math.log(max(sizes)))
        upper = max(upper, base + 1.0 / math.log(min(sizes)))
    return lower, upper


def _exponent_contradicts_class(
    model: str,
    x: Sequence[float],
    y: Sequence[float],
    ci_low: float,
    ci_high: float,
    fitted_baseline: float,
) -> bool:
    """Return True when the bootstrap exponent interval rules the class out.

    Growth between two classes — cache and memory effects bending a curve —
    fits the nearer class best without the class describing the data.  A
    constant overhead flattens the raw exponent, so an exponent *below* the
    class band only counts against it when the fit found no overhead that could
    explain it.
    """
    if model == "O(1)" or not (math.isfinite(ci_low) and math.isfinite(ci_high)):
        return False
    lower, upper = _class_exponent_band(model, x)
    if ci_low > upper:
        return True
    positive = [v for v in y if v > 0]
    y_min = min(positive) if positive else 0.0
    return ci_high < lower and fitted_baseline <= _OVERHEAD_FREE_SHARE * y_min


def assess_fit(
    x: Sequence[float],
    y: Sequence[float],
    *,
    effective_baseline: float,
    exponent_ci_low: float,
    exponent_ci_high: float,
    model_margin: float = float("inf"),
    model: str | None = None,
    fitted_baseline: float = 0.0,
    dropped_outlier: bool = False,
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

    if model is not None and _exponent_contradicts_class(
        model, x, y, exponent_ci_low, exponent_ci_high, fitted_baseline
    ):
        notes.append("exponent-mismatch")

    if dropped_outlier:
        notes.append("outlier-dropped")

    # The largest input is never discarded as an outlier, so a flat series
    # with one jump at the end is fitted as growth.  It may be; but it is one
    # reading's word against all the others.
    if model not in (None, "O(1)") and len(x) >= MIN_POINTS:
        pairs = sorted(zip(x, y))
        head_x = [p[0] for p in pairs if p[0] < pairs[-1][0]]
        head_y = [p[1] for p in pairs if p[0] < pairs[-1][0]]
        if len(head_x) >= 2 and _is_effectively_constant(
            head_y, head_x, allow_outlier=False
        ):
            notes.append("single-point-growth")

    if any(v <= 0 for v in y):
        notes.append("non-positive-timings")

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
