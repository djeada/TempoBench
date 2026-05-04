"""HTML report generation for TempoBench.

Public surface (preserved from the legacy single-file implementation):

- ``get_system_info`` — collect platform/CPU/memory metadata for reports
- ``generate_report`` — main HTML report from a summary CSV
- ``compare_summaries`` — diff two summary CSVs into a comparison DataFrame
- ``generate_comparison_report`` — HTML report for a comparison DataFrame
"""

from __future__ import annotations

from .comparison import compare_summaries, generate_comparison_report
from .report import generate_report
from .system import get_system_info

__all__ = [
    "compare_summaries",
    "generate_comparison_report",
    "generate_report",
    "get_system_info",
]
