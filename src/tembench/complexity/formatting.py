"""Human-readable rendering of fitted upper-bound formulas."""

from __future__ import annotations

import math

from .models import _BASIS_STR

_MODEL_SLOPE_INTERVALS = {
    "O(1)": (-math.inf, 0.08),
    "O(log n)": (0.08, 0.45),
    "O(n)": (0.45, 1.05),
    "O(n log n)": (1.05, 1.55),
    "O(n²)": (1.55, 2.4),
    "O(n³)": (2.4, math.inf),
}


def _format_coeff(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    if abs(v) >= 0.001:
        return f"{v:.4g}"
    return f"{v:.3e}"


def _format_formula(model: str, C: float, effective_baseline: float = 0.0) -> str:
    """Render e.g. 'T(n) ≤ 3.21e-04·n·log(n) + 12.5'."""
    b = _BASIS_STR.get(model, "?")
    c_str = _format_coeff(C)

    if b == "1":
        total = C + effective_baseline
        return f"T(n) ≤ {_format_coeff(total)}"

    term = f"{c_str}·{b}"
    if abs(effective_baseline) < 1e-9:
        return f"T(n) ≤ {term}"
    sign = "+" if effective_baseline >= 0 else "−"
    return f"T(n) ≤ {term} {sign} {_format_coeff(abs(effective_baseline))}"


def _format_model_label(
    model: str,
    empirical_exponent: float,
    exponent_ci_low: float,
    exponent_ci_high: float,
    strategy: str = "heuristic",
) -> str:
    """Render the user-facing complexity label for a fitted series."""
    if strategy != "strict":
        return model
    if not all(
        math.isfinite(v) for v in [empirical_exponent, exponent_ci_low, exponent_ci_high]
    ):
        return model

    lower, upper = _MODEL_SLOPE_INTERVALS.get(model, (-math.inf, math.inf))
    if exponent_ci_low >= lower and exponent_ci_high < upper:
        return model

    half_width = max(0.0, (exponent_ci_high - exponent_ci_low) / 2.0)
    if half_width < 0.015:
        return f"O(n^{empirical_exponent:.2f})"
    return f"O(n^{empirical_exponent:.2f}±{half_width:.2f})"
