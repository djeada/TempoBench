"""Big-O complexity fitting and selection.

Public API:
    FitResult, fit_models, predict_series

Internals exposed for tests / advanced use:
    _select_model, _basis_functions, _ols_fit, _log_log_slope,
    _slope_to_model, _is_effectively_constant, _tail_ratio_favors_linear,
    _upper_bound_offset, _format_formula, _format_coeff,
    _MODEL_ORDER, _ALL_MODELS, _BASIS_STR
"""

from .core import FitResult, fit_models, predict_series
from .fitting import (
    _is_effectively_constant,
    _log_log_slope,
    _ols_fit,
    _slope_to_model,
    _tail_ratio_favors_linear,
    _upper_bound_offset,
)
from .formatting import _format_coeff, _format_formula
from .models import _ALL_MODELS, _BASIS_STR, _MODEL_ORDER, _basis_functions
from .selection import _select_model

__all__ = [
    "FitResult",
    "fit_models",
    "predict_series",
    # internals (kept for back-compat with existing imports / tests)
    "_select_model",
    "_basis_functions",
    "_ols_fit",
    "_log_log_slope",
    "_slope_to_model",
    "_is_effectively_constant",
    "_tail_ratio_favors_linear",
    "_upper_bound_offset",
    "_format_formula",
    "_format_coeff",
    "_MODEL_ORDER",
    "_ALL_MODELS",
    "_BASIS_STR",
]
