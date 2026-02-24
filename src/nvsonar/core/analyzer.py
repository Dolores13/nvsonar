"""
GPU workload analyzer
Detects bottleneck patterns from NVML metrics
"""

from dataclasses import dataclass
from enum import Enum

from .baseline import get_baseline
from .monitor import Metrics


class BottleneckType(Enum):
    """Types of GPU bottlenecks we can detect"""

    MEMORY_BOUND = "memory_bound"
    COMPUTE_BOUND = "compute_bound"
    THERMAL_THROTTLING = "thermal_throttling"
    POWER_LIMITED = "power_limited"
    BALANCED = "balanced"
    IDLE = "idle"
    UNKNOWN = "unknown"


@dataclass
class Analysis:
    """Result of bottleneck analysis"""

    bottleneck_type: BottleneckType
    confidence: float  # 0-100
    device_util: float
    mem_util: float
    power_draw: float
    temperature: float

    def __str__(self) -> str:
        return f"{self.bottleneck_type.value} ({self.confidence:.0f}% confidence)"


class Analyzer:
    """Analyzes GPU metrics to detect bottlenecks"""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.baseline = None

        # Try to get baseline for this device
        try:
            self.baseline = get_baseline("", device_index)
        except Exception:
            pass  # No baseline is fine

    def analyze(self, metrics: Metrics) -> Analysis:
        """Analyze GPU metrics and identify bottlenecks"""

        device_util = metrics.device_utilization
        mem_util = metrics.memory_utilization
        power = metrics.power_usage or 0.0
        temp = metrics.temperature

        # Determine bottleneck type
        bottleneck, confidence = self._detect_bottleneck(
            device_util, mem_util, temp, power, metrics.power_limit
        )

        return Analysis(
            bottleneck_type=bottleneck,
            confidence=confidence,
            device_util=device_util,
            mem_util=mem_util,
            power_draw=power,
            temperature=temp,
        )

    def _detect_bottleneck(
        self,
        device_util: float,
        mem_util: float,
        temp: float,
        power: float,
        power_limit: float | None,
    ) -> tuple[BottleneckType, float]:
        """Simple heuristic-based bottleneck detection for GPU workloads"""

        # Idle detection
        if device_util < 5:
            return BottleneckType.IDLE, 95.0

        # Thermal throttling detection
        thermal_threshold = self.baseline.max_temperature - 3 if self.baseline else 83
        if temp > thermal_threshold:
            confidence = 85.0 if self.baseline else 80.0
            return BottleneckType.THERMAL_THROTTLING, confidence

        # Power limiting detection
        if power_limit and power > 0:
            power_ratio = power / power_limit
            if power_ratio > 0.95:
                return BottleneckType.POWER_LIMITED, 85.0

        # Heavy compute bound - GPU maxed out
        if device_util > 90 and mem_util < 80:
            return BottleneckType.COMPUTE_BOUND, 85.0

        # Heavy memory bound - memory maxed out
        if mem_util > 85 and device_util < 90:
            return BottleneckType.MEMORY_BOUND, 80.0

        # Balanced heavy workload
        if device_util > 75 and mem_util > 70:
            return BottleneckType.BALANCED, 70.0

        # Moderate compute workload
        if device_util >= 30 and mem_util < 50:
            return BottleneckType.COMPUTE_BOUND, 60.0

        # Moderate memory workload
        if mem_util >= 40 and device_util < 60:
            return BottleneckType.MEMORY_BOUND, 55.0

        # Light balanced workload
        if device_util >= 20 or mem_util >= 20:
            return BottleneckType.BALANCED, 50.0

        return BottleneckType.UNKNOWN, 30.0
