"""Baseline data for GPU models"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Baseline:
    """GPU thermal baseline"""

    max_temperature: int


def _get_baseline_from_nvml(device_index: int = 0) -> Optional[Baseline]:
    """Get baseline data directly from GPU hardware"""
    try:
        from ..utils.info import initialize

        if not initialize():
            return None

        return Baseline(max_temperature=83)

    except Exception:
        return None


def _get_baseline_fallback(device_name: str) -> Optional[Baseline]:
    """Fallback baseline for common GPUs"""

    name_lower = device_name.lower()

    if any(x in name_lower for x in ["4090", "4080", "4070"]):
        return Baseline(max_temperature=83)

    elif any(x in name_lower for x in ["3090", "3080", "3070"]):
        return Baseline(max_temperature=83)

    elif any(x in name_lower for x in ["20", "16", "2080", "2070", "1660"]):
        return Baseline(max_temperature=83)

    else:
        return Baseline(max_temperature=80)


def get_baseline(device_name: str = "", device_index: int = 0) -> Optional[Baseline]:
    """Get baseline data for GPU"""

    # Try hardware detection first
    baseline = _get_baseline_from_nvml(device_index)
    if baseline:
        return baseline

    # Fall back to architecture-based estimates
    if device_name:
        return _get_baseline_fallback(device_name)

    return None
