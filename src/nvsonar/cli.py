"""Command-line interface for NVSonar"""

import sys

import typer
from rich.console import Console

app = typer.Typer(
    name="nvsonar",
    help="Active GPU diagnostic tool",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()


@app.callback()
def callback(ctx: typer.Context):
    """Launch interactive TUI"""
    if ctx.invoked_subcommand is None:
        try:
            from nvsonar.tui.app import NVSonarApp

            tui_app = NVSonarApp()
            tui_app.run()
        except ImportError as e:
            console.print(f"[red]Error: Failed to import TUI: {e}[/red]")
            console.print("[yellow]Install dependencies: pip install nvsonar[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        ctx.exit()


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
