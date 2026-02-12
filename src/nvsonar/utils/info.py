"""GPU detection and information utilities"""

from dataclasses import dataclass

import pynvml as nvml


def _decode_if_bytes(value: str | bytes) -> str:
    """Decode bytes to string if needed"""
    return value.decode("utf-8") if isinstance(value, bytes) else value


@dataclass
class Info:
    """GPU device information"""

    index: int
    name: str
    uuid: str
    memory_total: int
    driver_version: str
    cuda_version: str
    pci_bus_id: str


class _NVMLContext:
    """NVML library context"""

    def __init__(self):
        self._initialized = False

    def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            nvml.nvmlInit()
            self._initialized = True
            return True
        except nvml.NVMLError:
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized


_nvml_context = _NVMLContext()


def initialize() -> bool:
    return _nvml_context.initialize()


def get_device_count() -> int:
    """Get number of available GPUS"""
    if not _nvml_context.is_initialized:
        return 0

    try:
        return nvml.nvmlDeviceGetCount()
    except nvml.NVMLError:
        return 0


def get_device_info(device_index: int = 0) -> Info | None:
    """Get GPU information"""
    if not _nvml_context.is_initialized:
        return None

    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)

        name = _decode_if_bytes(nvml.nvmlDeviceGetName(handle))
        uuid = _decode_if_bytes(nvml.nvmlDeviceGetUUID(handle))
        memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)

        driver_version = _decode_if_bytes(nvml.nvmlSystemGetDriverVersion())

        cuda_version = nvml.nvmlSystemGetCudaDriverVersion()
        cuda_version_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"

        pci_info = nvml.nvmlDeviceGetPciInfo(handle)
        pci_bus_id = _decode_if_bytes(pci_info.busId)

        return Info(
            index=device_index,
            name=name,
            uuid=uuid,
            memory_total=memory_info.total,
            driver_version=driver_version,
            cuda_version=cuda_version_str,
            pci_bus_id=pci_bus_id,
        )
    except nvml.NVMLError:
        return None


def list_devices() -> list[Info]:
    """List all available GPUs"""
    if not _nvml_context.initialize():
        return []

    count = get_device_count()
    devices = []

    for i in range(count):
        info = get_device_info(i)
        if info:
            devices.append(info)

    return devices
