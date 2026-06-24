"""
Harness Agent – main entry point.

Usage
-----
    python main.py
    python main.py --requirements "Build a REST API with FastAPI"
    python main.py --dry-run --log-format console

Environment
-----------
All configuration is read from ``.env`` (see ``config/settings.py``).
CLI flags override environment variables for one-off runs.
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harness – LLM-driven code-generation agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--requirements",
        "-r",
        type=str,
        default=None,
        help="Free-text project requirements (overrides hard-coded default).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Preview mode: no files written, lint simulated.",
    )
    parser.add_argument(
        "--log-format",
        choices=["json", "console"],
        default=None,
        help="Log output format.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Override the workspace directory.",
    )
    return parser.parse_args()


def _print_summary_table(summary) -> None:
    table = Table(title=f"Run Summary  [dim]({summary.run_id})[/dim]", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    status_color = {
        "completed": "green",
        "failed": "red",
        "aborted": "yellow",
    }.get(summary.status.value, "white")

    table.add_row("Status", f"[{status_color}]{summary.status.value.upper()}[/{status_color}]")
    table.add_row("Total tasks", str(summary.total_tasks))
    table.add_row("Completed", f"[green]{summary.completed_tasks}[/green]")
    table.add_row("Failed", f"[red]{summary.failed_tasks}[/red]")
    table.add_row("Skipped", str(summary.skipped_tasks))
    if summary.duration_seconds is not None:
        table.add_row("Duration", f"{summary.duration_seconds:.1f}s")

    console.print(table)

    if summary.failed_tasks:
        console.print("\n[bold red]Failed tasks:[/bold red]")
        for task in summary.tasks:
            if task.status.value == "failed":
                console.print(f"  • [yellow]{task.file_path}[/yellow]")
                if task.last_error:
                    console.print(f"    [dim]{task.last_error[:200]}[/dim]")


def main() -> int:
    args = _parse_args()

    # ── Apply CLI overrides before importing settings ──────────────────────────
    import os
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
    if args.log_format:
        os.environ["LOG_FORMAT"] = args.log_format

    # ── Configure logging (must happen before any other imports) ───────────────
    from config.logging import configure_logging
    from config.settings import settings

    run_id = configure_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        run_id=settings.run_id,
    )

    # ── Build the orchestrator ─────────────────────────────────────────────────
    from agent.container import build_orchestrator
    orchestrator, run_id = build_orchestrator(
        workspace_dir=args.workspace,
        run_id=run_id,
    )

    # ── Default requirements ───────────────────────────────────────────────────
    requirements = args.requirements or (
        "src/math_tool.py 에 add, subtract, multiply, divide 함수를 구현하고, "
        "각 함수에 대한 단위 테스트를 tests/test_math_tool.py 에 작성하고, "
        "src/run.py 에서 모든 함수를 실행해 결과를 출력하는 CLI를 만들어줘."
    )

    console.print(
        Panel.fit(
            f"[bold cyan]🚀 Harness Agent[/bold cyan]  [dim]run_id={run_id}[/dim]\n"
            f"[dim]dry_run={settings.dry_run}  model={settings.ai_model}[/dim]",
            border_style="cyan",
        )
    )

    # ── Execute ────────────────────────────────────────────────────────────────
    try:
        summary = orchestrator.execute(requirements)
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted by user.[/yellow]")
        return 130
    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[bold red]Fatal error:[/bold red] {exc}")
        return 1

    _print_summary_table(summary)

    return 0 if summary.failed_tasks == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
