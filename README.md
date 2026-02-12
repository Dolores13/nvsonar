# NVSonar

[![PyPI version](https://img.shields.io/pypi/v/nvsonar.svg)](https://pypi.org/project/nvsonar/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Downloads](https://pepy.tech/badge/nvsonar)](https://pepy.tech/project/nvsonar)


Active GPU diagnostic tool with real-time bottleneck detection and performance analysis.

## Why NVSonar?

Traditional GPU monitoring tools show utilization percentages, but this can be misleading. A GPU reporting 100% utilization may actually be computing useful work, or wastefully stalled waiting on memory transfers, thermal throttling, or power limits.

NVSonar analyzes real-time patterns from NVML metrics to identify what's actually limiting your GPU performance.
## Features

- Real-time Bottleneck Detection
- Subsystem Utilization Analysis
- Peak Value Tracking
- Visual Progress Bars
- Multi-GPU Support**

## Installation

```bash
pip install nvsonar
```

## Quick Start

```bash
# Launch interactive TUI with all GPUs and live metrics
nvsonar
```

## Interface

```
┌─ NVSonar ──────────────────────────────────────────────────────┐
│  [Overview] [History] [Settings]                               │
├────────────────────────────────────────────────────────────────┤
│                        Available GPUs                          │ 
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━┩
│ Index │ Name                       │ Memory │ Driver    │ CUDA │
│   0   │ NVIDIA GeForce GTX 1650 Ti │ 4.0 GB │ 580.95.05 │ 13.0 │
└───────┴────────────────────────────┴────────┴───────────┴──────┘

╭────────── NVIDIA GeForce GTX 1650 Ti Metrics ─────────────────╮
│  Compute           ████████████░░░░░░░░ 65%                   │
│  Memory            ████████░░░░░░░░░░░░ 35%                   │
│  Thermal           ████████████░░░░░░░░ 68%                   │
│                                                               │
│  Status            GPU cores are the limiting factor          │
│                                                               │
│  Power             ████████████████░░░░ 32.5W / 50.0W         │
│  Temperature       ████████████░░░░░░░░ 68°C / 83°C           │
│  GPU Utilization   ████████████░░░░░░░░ 65%                   │
│  Memory Utilization████████░░░░░░░░░░░░ 35%                   │
│  Memory Used       ██████░░░░░░░░░░░░░░ 2.2 / 4.0 GB          │
│  GPU Clock         1740 MHz                                   │
│  Memory Clock      5000 MHz                                   │
╰───────────────────────────────────────────────────────────────╯
```

## Requirements

- Python 3.10+
- NVIDIA GPU with driver installed
- CUDA toolkit (for active probes)
- Linux (tested on Ubuntu)


## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.


## Author

Maintained by [**Bekmukhamed Tursunbayev**](https://btursunbayev.com)  
GitHub: https://github.com/btursunbayev · PyPI: https://pypi.org/user/btursunbayev/

