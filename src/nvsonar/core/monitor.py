"""GPU monitoring using NVML"""

from dataclasses import dataclass

import pynvml as nvml

from nvsonar.utils.gpu_info import initialize


@dataclass
class GPUMetrics:
    """Current GPU metrics snapshot"""

    temperature: float
    power_usage: float | None
    power_limit: float | None
    fan_speed: int | None
    gpu_utilization: int
    memory_utilization: int
    memory_used: int
    memory_total: int
    gpu_clock: int
    memory_clock: int


class GPUMonitor:
    """Monitor GPU metrics using NVML"""

    def __init__(self, device_index: int = 0):
        """Initialize monitor for specific GPU"""

        self.device_index = device_index
        self._handle = None

        if not initialize():
            raise RuntimeError("Failed to initialize NVML")

        try:
            self._handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
        except nvml.NVMLError as e:
            raise RuntimeError(f"Failed to get GPU {device_index}: {e}")

    def get_current_metrics(self) -> GPUMetrics:
        """Get current GPU metrics"""
        if self._handle is None:
            raise RuntimeError("Monitor not initialized")

        try:
            temperature = nvml.nvmlDeviceGetTemperature(
                self._handle, nvml.NVML_TEMPERATURE_GPU
            )

            try:
                power_usage = nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            except nvml.NVMLError:
                power_usage = None

            try:
                power_limit = (
                    nvml.nvmlDeviceGetPowerManagementLimit(self._handle) / 1000.0
                )
            except nvml.NVMLError:
                power_limit = None

            try:
                fan_speed = nvml.nvmlDeviceGetFanSpeed(self._handle)
            except nvml.NVMLError:
                fan_speed = None

            utilization = nvml.nvmlDeviceGetUtilizationRates(self._handle)

            memory_info = nvml.nvmlDeviceGetMemoryInfo(self._handle)

            gpu_clock = nvml.nvmlDeviceGetClockInfo(
                self._handle, nvml.NVML_CLOCK_GRAPHICS
            )
            memory_clock = nvml.nvmlDeviceGetClockInfo(
                self._handle, nvml.NVML_CLOCK_MEM
            )

            return GPUMetrics(
                temperature=temperature,
                power_usage=power_usage,
                power_limit=power_limit,
                fan_speed=fan_speed,
                gpu_utilization=utilization.gpu,
                memory_utilization=utilization.memory,
                memory_used=memory_info.used,
                memory_total=memory_info.total,
                gpu_clock=gpu_clock,
                memory_clock=memory_clock,
            )
        except nvml.NVMLError as e:
            raise RuntimeError(f"Failed to get metrics: {e}")
