"""Main TUI application"""

from collections import deque
from dataclasses import dataclass
from time import time

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from textual.app import App as TextualApp
from textual.app import ComposeResult
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

from nvsonar.core.analyzer import Analyzer
from nvsonar.core.monitor import Monitor
from nvsonar.utils.info import (
    get_device_count,
    get_device_info,
    initialize,
    list_devices,
)

UPDATE_INTERVAL = 0.5
PEAK_WINDOW = 60.0


def _make_bar(value: float, max_value: float, width: int = 20) -> str:
    """Create a text progress bar"""
    if max_value <= 0:
        ratio = 0.0
    else:
        ratio = min(value / max_value, 1.0)

    filled = int(ratio * width)
    empty = width - filled
    return "█" * filled + "░" * empty


@dataclass
class MetricSnapshot:
    """Single metric snapshot with timestamp"""

    timestamp: float
    temperature: float
    power_usage: float
    device_utilization: int
    memory_utilization: int
    memory_used: int
    device_clock: int
    memory_clock: int
    compute_util: int | None = None
    mem_util: int | None = None
    thermal_percent: float | None = None
    status: str | None = None


class DeviceList(Static):
    """Display available GPUs"""

    def on_mount(self) -> None:

        if not initialize():
            self.update("[red]Failed to initialize NVML[/red]")
            return

        devices = list_devices()
        if not devices:
            self.update("[yellow]No GPUs found[/yellow]")
            return

        table = Table(title="Available GPUs")
        table.add_column("Index", style="cyan", justify="center")
        table.add_column("Name", style="green")
        table.add_column("Memory", style="yellow", justify="right")
        table.add_column("Driver", style="magenta")
        table.add_column("CUDA", style="blue")

        for device in devices:
            memory_gb = device.memory_total / (1024**3)
            table.add_row(
                str(device.index),
                device.name,
                f"{memory_gb:.1f} GB",
                device.driver_version,
                device.cuda_version,
            )

        self.update(table)


class Metrics(Static):
    """Display live metrics for all GPUs"""

    def __init__(self):
        super().__init__()
        self.monitors = []
        self.device_map = {}
        self.history = {}
        self.device_names = {}

    def on_mount(self) -> None:

        if not initialize():
            self.update("[red]Failed to initialize NVML[/red]")
            return

        device_count = get_device_count()
        if device_count == 0:
            self.update("[yellow]No GPUs found[/yellow]")
            return

        for i in range(device_count):
            try:
                monitor = Monitor(i)
                analyzer = Analyzer(i)
                self.monitors.append((i, monitor))
                self.device_map[i] = (monitor, analyzer)
                self.history[i] = deque()

                device_info = get_device_info(i)
                if device_info:
                    self.device_names[i] = device_info.name
                else:
                    self.device_names[i] = f"GPU {i}"
            except RuntimeError:
                pass

        if self.monitors:
            self.set_interval(UPDATE_INTERVAL, self.update_metrics)

    def update_metrics(self) -> None:
        """Update metrics for all GPUs"""
        if not self.monitors:
            return

        try:
            current_time = time()
            panels = []
            for device_index, monitor in self.monitors:
                m = monitor.get_current_metrics()

                _, analyzer = self.device_map.get(device_index, (None, None))

                # Perform analysis before adding to history
                analysis = None
                if analyzer:
                    analysis = analyzer.analyze(m)

                # Add to history with analysis data
                self._add_snapshot(device_index, m, analysis, analyzer)
                self._clean_old_snapshots(device_index, current_time)

                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="yellow")

                # Subsystem utilization analysis
                if analysis:
                    # Show subsystem utilizations
                    compute_bar = _make_bar(analysis.device_util, 100)
                    table.add_row("Compute", f"{compute_bar} {analysis.device_util}%")

                    memory_bar = _make_bar(analysis.mem_util, 100)
                    table.add_row("Memory", f"{memory_bar} {analysis.mem_util}%")

                    # Show thermal headroom
                    if analyzer.baseline:
                        thermal_percent = (
                            analysis.temperature / analyzer.baseline.max_temperature
                        ) * 100
                        thermal_bar = _make_bar(thermal_percent, 100)
                        table.add_row(
                            "Thermal", f"{thermal_bar} {thermal_percent:.0f}%"
                        )

                    # Status with color
                    bottleneck_color = self._get_bottleneck_color(
                        analysis.bottleneck_type.value
                    )
                    explanation = self._get_bottleneck_explanation(
                        analysis.bottleneck_type.value
                    )
                    table.add_row("", "")
                    table.add_row(
                        "Status",
                        f"[{bottleneck_color}]{explanation}[/{bottleneck_color}]",
                    )
                    table.add_row("", "")

                # Show current values with progress bars
                # Power
                if m.power_usage:
                    if m.power_limit:
                        power_bar = _make_bar(m.power_usage, m.power_limit)
                        power_display = (
                            f"{power_bar} {m.power_usage:.1f}W / {m.power_limit:.1f}W"
                        )
                    else:
                        power_display = f"{m.power_usage:.1f}W"
                    table.add_row("Power", power_display)

                # Temperature
                if analyzer and analyzer.baseline:
                    max_temp = analyzer.baseline.max_temperature
                    temp_bar = _make_bar(m.temperature, max_temp)
                    temp_display = f"{temp_bar} {m.temperature:.1f}°C / {max_temp}°C"
                    table.add_row("Temperature", temp_display)
                else:
                    table.add_row("Temperature", f"{m.temperature:.1f}°C")

                if m.fan_speed is not None:
                    fan_bar = _make_bar(m.fan_speed, 100)
                    table.add_row("Fan Speed", f"{fan_bar} {m.fan_speed}%")

                # GPU Utilization
                gpu_bar = _make_bar(m.device_utilization, 100)
                table.add_row("GPU Utilization", f"{gpu_bar} {m.device_utilization}%")

                # Memory Utilization
                mem_bar = _make_bar(m.memory_utilization, 100)
                table.add_row(
                    "Memory Utilization", f"{mem_bar} {m.memory_utilization}%"
                )

                # Memory Used
                vram_bar = _make_bar(m.memory_used, m.memory_total)
                table.add_row(
                    "Memory Used",
                    f"{vram_bar} {m.memory_used / (1024**3):.1f} / {m.memory_total / (1024**3):.1f} GB",
                )

                # Clocks
                table.add_row("GPU Clock", f"{m.device_clock} MHz")
                table.add_row("Memory Clock", f"{m.memory_clock} MHz")

                device_name = self.device_names.get(device_index, f"GPU {device_index}")
                panel = Panel(
                    table, title=f"{device_name} Metrics", border_style="green"
                )
                panels.append(panel)

            group = Group(*panels)
            self.update(group)
        except Exception as e:
            self.update(f"[red]Error: {e}[/red]")

    def _clean_old_snapshots(self, device_index: int, current_time: float) -> None:
        """Remove snapshots older than PEAK_WINDOW seconds"""
        history = self.history.get(device_index)
        if not history:
            return

        while history and (current_time - history[0].timestamp) > PEAK_WINDOW:
            history.popleft()

    def _add_snapshot(
        self, device_index: int, metrics, analysis=None, analyzer=None
    ) -> None:
        """Add new snapshot to history"""
        # Calculate analyzed metrics if available
        compute_util = analysis.device_util if analysis else None
        mem_util = analysis.mem_util if analysis else None
        thermal_percent = None
        status = None

        if analysis:
            status = analysis.bottleneck_type.value
            if analyzer and analyzer.baseline:
                thermal_percent = (
                    metrics.temperature / analyzer.baseline.max_temperature
                ) * 100

        snapshot = MetricSnapshot(
            timestamp=time(),
            temperature=metrics.temperature,
            power_usage=metrics.power_usage or 0.0,
            device_utilization=metrics.device_utilization,
            memory_utilization=metrics.memory_utilization,
            memory_used=metrics.memory_used,
            device_clock=metrics.device_clock,
            memory_clock=metrics.memory_clock,
            compute_util=compute_util,
            mem_util=mem_util,
            thermal_percent=thermal_percent,
            status=status,
        )
        self.history[device_index].append(snapshot)

    def _get_peaks(self, device_index: int, current_time: float) -> dict:
        """Get peak values from history"""
        self._clean_old_snapshots(device_index, current_time)
        history = self.history.get(device_index, [])

        if not history:
            return {}

        # Get peaks for basic metrics
        peaks = {
            "temperature": max(s.temperature for s in history),
            "power_usage": max(s.power_usage for s in history),
            "device_utilization": max(s.device_utilization for s in history),
            "memory_utilization": max(s.memory_utilization for s in history),
            "memory_used": max(s.memory_used for s in history),
            "device_clock": max(s.device_clock for s in history),
            "memory_clock": max(s.memory_clock for s in history),
        }

        # Add analyzed metrics if available
        compute_values = [s.compute_util for s in history if s.compute_util is not None]
        if compute_values:
            peaks["compute_util"] = max(compute_values)

        mem_values = [s.mem_util for s in history if s.mem_util is not None]
        if mem_values:
            peaks["mem_util"] = max(mem_values)

        thermal_values = [
            s.thermal_percent for s in history if s.thermal_percent is not None
        ]
        if thermal_values:
            peaks["thermal_percent"] = max(thermal_values)

        # Find the status at peak compute utilization or fallback to most recent
        if compute_values:
            peak_compute_snapshot = max(
                (s for s in history if s.compute_util is not None),
                key=lambda s: s.compute_util,
            )
            peaks["status"] = peak_compute_snapshot.status

        return peaks

    def _get_bottleneck_color(self, bottleneck_type: str) -> str:
        """Get color for bottleneck type"""
        colors = {
            "memory_bound": "cyan",
            "compute_bound": "blue",
            "thermal_throttling": "red",
            "power_limited": "yellow",
            "balanced": "green",
            "idle": "dim",
            "unknown": "white",
        }
        return colors.get(bottleneck_type, "white")

    def _get_bottleneck_explanation(self, bottleneck_type: str) -> str:
        """Get human-readable explanation for bottleneck type"""
        explanations = {
            "memory_bound": "Memory subsystem is the limiting factor",
            "compute_bound": "GPU cores are the limiting factor",
            "thermal_throttling": "Temperature too high, reducing performance",
            "power_limited": "Power draw at limit, reducing performance",
            "balanced": "GPU and memory working efficiently together",
            "idle": "No significant workload detected",
            "unknown": "Workload pattern unclear",
        }
        return explanations.get(bottleneck_type, "Unknown bottleneck")


class PeakMetrics(Static):
    """Display peak metrics from history"""

    def __init__(self, metrics_widget):
        super().__init__()
        self.metrics_widget = metrics_widget

    def on_mount(self) -> None:
        self.set_interval(UPDATE_INTERVAL, self.update_peaks)

    def update_peaks(self) -> None:
        """Update peak metrics display"""
        if not self.metrics_widget.monitors:
            self.update("[yellow]No peak data available[/yellow]")
            return

        try:
            current_time = time()
            panels = []

            for device_index, monitor in self.metrics_widget.monitors:
                peaks = self.metrics_widget._get_peaks(device_index, current_time)

                if not peaks:
                    continue

                _, analyzer = self.metrics_widget.device_map.get(
                    device_index, (None, None)
                )

                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_column("Metric", style="cyan")
                table.add_column("Peak Value", style="yellow")

                # Subsystem utilization peaks (if available)
                if "compute_util" in peaks:
                    compute_bar = _make_bar(peaks["compute_util"], 100)
                    table.add_row("Compute", f"{compute_bar} {peaks['compute_util']}%")

                if "mem_util" in peaks:
                    mem_bar = _make_bar(peaks["mem_util"], 100)
                    table.add_row("Memory", f"{mem_bar} {peaks['mem_util']}%")

                if "thermal_percent" in peaks:
                    thermal_bar = _make_bar(peaks["thermal_percent"], 100)
                    table.add_row(
                        "Thermal", f"{thermal_bar} {peaks['thermal_percent']:.0f}%"
                    )

                # Status at peak (if available)
                if "status" in peaks and peaks["status"]:
                    bottleneck_color = self.metrics_widget._get_bottleneck_color(
                        peaks["status"]
                    )
                    explanation = self.metrics_widget._get_bottleneck_explanation(
                        peaks["status"]
                    )
                    table.add_row("", "")
                    table.add_row(
                        "Peak Status",
                        f"[{bottleneck_color}]{explanation}[/{bottleneck_color}]",
                    )
                    table.add_row("", "")

                # Power
                if peaks["power_usage"] > 0:
                    # Try to get power limit from current metrics
                    m = monitor.get_current_metrics()
                    if m.power_limit:
                        power_bar = _make_bar(peaks["power_usage"], m.power_limit)
                        table.add_row(
                            "Power",
                            f"{power_bar} {peaks['power_usage']:.1f}W / {m.power_limit:.1f}W",
                        )
                    else:
                        table.add_row("Power", f"{peaks['power_usage']:.1f}W")

                # Temperature
                if analyzer and analyzer.baseline:
                    max_temp = analyzer.baseline.max_temperature
                    temp_bar = _make_bar(peaks["temperature"], max_temp)
                    table.add_row(
                        "Temperature",
                        f"{temp_bar} {peaks['temperature']:.1f}°C / {max_temp}°C",
                    )
                else:
                    table.add_row("Temperature", f"{peaks['temperature']:.1f}°C")

                # GPU Utilization
                gpu_bar = _make_bar(peaks["device_utilization"], 100)
                table.add_row(
                    "GPU Utilization", f"{gpu_bar} {peaks['device_utilization']}%"
                )

                # Memory Utilization
                mem_bar = _make_bar(peaks["memory_utilization"], 100)
                table.add_row(
                    "Memory Utilization", f"{mem_bar} {peaks['memory_utilization']}%"
                )

                # Memory Used
                m = monitor.get_current_metrics()
                vram_bar = _make_bar(peaks["memory_used"], m.memory_total)
                table.add_row(
                    "Memory Used",
                    f"{vram_bar} {peaks['memory_used'] / (1024**3):.1f} / {m.memory_total / (1024**3):.1f} GB",
                )

                # Clocks
                table.add_row("GPU Clock", f"{peaks['device_clock']} MHz")
                table.add_row("Memory Clock", f"{peaks['memory_clock']} MHz")

                device_name = self.metrics_widget.device_names.get(
                    device_index, f"GPU {device_index}"
                )
                panel = Panel(
                    table,
                    title=f"{device_name} Peak Values (last 60s)",
                    border_style="yellow",
                )
                panels.append(panel)

            if panels:
                group = Group(*panels)
                self.update(group)
            else:
                self.update(
                    "[dim]No peak data yet - run a workload to collect data[/dim]"
                )
        except Exception as e:
            self.update(f"[red]Error: {e}[/red]")


class App(TextualApp):
    """NVSonar terminal interface"""

    TITLE = "NVSonar"
    SUB_TITLE = "GPU Diagnostic Tool"
    CSS = """
    DeviceList {
        height: auto;
        padding: 1;
        margin: 1;
    }

    Metrics {
        height: auto;
        padding: 0;
        margin: 1;
    }
    
    TabbedContent {
        height: auto;
    }
    
    .placeholder {
        padding: 2;
        text-align: center;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()

        # Create metrics widget to share history with peak metrics
        metrics_widget = Metrics()

        with TabbedContent():
            with TabPane("Overview", id="overview"):
                yield DeviceList()
                yield metrics_widget
            with TabPane("History", id="history"):
                yield PeakMetrics(metrics_widget)
            with TabPane("Settings", id="settings"):
                yield Static("[dim]Settings coming soon[/dim]", classes="placeholder")
        yield Footer()

    def action_quit(self) -> None:
        """Quit the application"""
        self.exit()


if __name__ == "__main__":
    app = App()
    app.run()
