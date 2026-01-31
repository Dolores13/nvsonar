"""Command-line interface for NVSonar"""

import sys

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main():
    """Launch GPU monitoring interface"""
    try:
        from nvsonar.tui.app import NVSonarApp

        tui_app = NVSonarApp()
        tui_app.run()
    except ImportError as e:
        typer.echo(f"Error: Failed to import TUI: {e}", err=True)
        typer.echo("Install dependencies: pip install nvsonar", err=True)
        sys.exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
