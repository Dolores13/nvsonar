# Changelog

## [1.0.1] - 2026-01-30

### Added
- Multi-GPU support

### Changed
- Simplified to use pynvml directly
- Removed unimplemented feature claims from documentation

## [1.0.0] - 2026-01-28

### Added
- GPU detection and information gathering
- Real-time metrics monitoring (temperature, power, utilization, clocks)
- CLI interface with `list`, `monitor`, and `tui` commands
- Interactive terminal UI with live metrics display
- Graceful handling of unsupported GPU features
- Test suite with GPU-specific markers

### Fixed
- CI workflow configuration for PyPI publishing

## [0.0.1] - 2026-01-27

### Added
- Initial project structure
- PyPI publishing setup with GitHub Actions
- Apache 2.0 license
- Code of conduct and security policy
