"""GPU detection and information utilities"""

from dataclasses import dataclass

try:
    import pynvml as nvml
except ImportError:
    import nvidia_ml_py.nvml as nvml


@dataclass
class GPUInfo:
    """Information about a GPU device"""

    index: int
    name: str
    uuid: str
    memory_total: int
    driver_version: str
    cuda_version: str
    pci_bus_id: str


_initialized = False


def initialize() -> bool:
    """Initialize NVML library"""
    global _initialized
    if _initialized:
        return True

    try:
        nvml.nvmlInit()
        _initialized = True
        return True
    except nvml.NVMLError:
        return False


def get_device_count() -> int:
    """Get number of available GPUs"""
    if not _initialized:
        if not initialize():
            return 0

    try:
        return nvml.nvmlDeviceGetCount()
    except nvml.NVMLError:
        return 0


def get_gpu_info(device_index: int = 0) -> GPUInfo | None:
    """Get information about specific GPU"""
    if not _initialized:
        if not initialize():
            return None

    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)

        name = nvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        uuid = nvml.nvmlDeviceGetUUID(handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode("utf-8")

        memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)

        driver_version = nvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode("utf-8")

        cuda_version = nvml.nvmlSystemGetCudaDriverVersion()
        cuda_version_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"

        pci_info = nvml.nvmlDeviceGetPciInfo(handle)
        pci_bus_id = pci_info.busId
        if isinstance(pci_bus_id, bytes):
            pci_bus_id = pci_bus_id.decode("utf-8")

        return GPUInfo(
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


def list_all_gpus() -> list[GPUInfo]:
    """List all available GPUs"""
    count = get_device_count()
    gpus = []

    for i in range(count):
        info = get_gpu_info(i)
        if info:
            gpus.append(info)

    return gpus


def shutdown():
    """Shutdown NVML library"""
    global _initialized
    if _initialized:
        try:
            nvml.nvmlShutdown()
            _initialized = False
        except pynvml.NVMLError:
            pass
