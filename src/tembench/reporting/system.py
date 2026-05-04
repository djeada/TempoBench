"""System information collection for reproducibility."""

from __future__ import annotations

import platform
from datetime import datetime, timezone

import psutil


def get_system_info() -> dict:
    """Gather system information for reproducibility."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "hostname": platform.node(),
    }
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            info["cpu_freq_mhz"] = round(cpu_freq.current, 1)
    except Exception:
        pass
    return info
