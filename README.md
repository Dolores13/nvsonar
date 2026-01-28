# NVSonar

[![PyPI version](https://img.shields.io/pypi/v/nvsonar.svg)](https://pypi.org/project/nvsonar/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Active GPU diagnostic tool that identifies performance bottlenecks using micro-probes.

## Why NVSonar?

Traditional GPU monitoring tools show utilization percentages. A GPU at 100% could be:
- Actually computing (good)
- Waiting for memory transfers (memory-bound)
- Waiting for PCIe transfers (PCIe-bound)

NVSonar runs targeted CUDA micro-probes to measure actual hardware performance and identify bottlenecks.




## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
