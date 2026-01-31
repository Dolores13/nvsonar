"""Main TUI application"""

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static

from nvsonar.core.monitor import GPUMonitor
from nvsonar.utils.gpu_info import get_device_count, initialize, list_all_gpus

UPDATE_INTERVAL = 0.5  # Update interval in seconds


class GPUListDisplay(Static):
    """Display all available GPUs in a table"""

    def on_mount(self) -> None:
        """Initialize GPU list"""

        if not initialize():
            self.update("[red]Failed to initialize NVML[/red]")
            return

        gpus = list_all_gpus()
        if not gpus:
            self.update("[yellow]No GPUs found[/yellow]")
            return

        table = Table(title="Available GPUs")
        table.add_column("Index", style="cyan", justify="center")
        table.add_column("Name", style="green")
        table.add_column("Memory", style="yellow", justify="right")
        table.add_column("Driver", style="magenta")
        table.add_column("CUDA", style="blue")

        for gpu in gpus:
            memory_gb = gpu.memory_total / (1024**3)
            table.add_row(
                str(gpu.index),
                gpu.name,
                f"{memory_gb:.1f} GB",
                gpu.driver_version,
                gpu.cuda_version,
            )

        self.update(table)


class LiveMetrics(Static):
    """Display live metrics for all GPUs"""

    def __init__(self):
        super().__init__()
        self.monitors = []

    def on_mount(self) -> None:
        """Start monitoring all GPUs"""

        if not initialize():
            self.update("[red]Failed to initialize NVML[/red]")
            return

        gpu_count = get_device_count()
        if gpu_count == 0:
            self.update("[yellow]No GPUs found[/yellow]")
            return

        for i in range(gpu_count):
            try:
                monitor = GPUMonitor(i)
                self.monitors.append((i, monitor))
            except RuntimeError:
                pass

        if self.monitors:
            self.set_interval(UPDATE_INTERVAL, self.update_metrics)

    def update_metrics(self) -> None:
        """Update metrics for all GPUs"""
        if not self.monitors:
            return

        try:
            panels = []
            for device_index, monitor in self.monitors:
                m = monitor.get_current_metrics()

                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="yellow")

                table.add_row("Temperature", f"{m.temperature:.1f}°C")

                if m.power_usage and m.power_limit:
                    table.add_row(
                        "Power", f"{m.power_usage:.1f}W / {m.power_limit:.1f}W"
                    )
                elif m.power_usage:
                    table.add_row("Power", f"{m.power_usage:.1f}W")

                if m.fan_speed is not None:
                    table.add_row("Fan Speed", f"{m.fan_speed}%")

                table.add_row("GPU Utilization", f"{m.gpu_utilization}%")
                table.add_row("Memory Utilization", f"{m.memory_utilization}%")
                table.add_row(
                    "Memory Used",
                    f"{m.memory_used / (1024**3):.1f} / {m.memory_total / (1024**3):.1f} GB",
                )
                table.add_row("GPU Clock", f"{m.gpu_clock} MHz")
                table.add_row("Memory Clock", f"{m.memory_clock} MHz")

                panel = Panel(
                    table, title=f"GPU {device_index} Metrics", border_style="green"
                )
                panels.append(panel)

            group = Group(*panels)
            self.update(group)
        except Exception as e:
            self.update(f"[red]Error: {e}[/red]")


class NVSonarApp(App):
    """NVSonar terminal user interface"""

    TITLE = "NVSonar"
    SUB_TITLE = "GPU Diagnostic Tool"
    CSS = """
    GPUListDisplay {
        height: auto;
        padding: 1;
        margin: 1;
    }

    LiveMetrics {
        height: auto;
        padding: 0;
        margin: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()
        yield GPUListDisplay()
        yield LiveMetrics()
        yield Footer()

    def action_quit(self) -> None:
        """Quit the application"""
        self.exit()


if __name__ == "__main__":
    app = NVSonarApp()
    app.run()
