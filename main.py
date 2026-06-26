"""
Harness Agent – main entry point.

Usage
-----
    python main.py
    python main.py --language bash --requirements "로그 백업 스크립트 작성"
    python main.py --language c    --requirements "링크드 리스트 라이브러리 구현"
    python main.py --language cpp  --requirements "스택 템플릿 클래스 구현"
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

# 언어별 기본 요구사항 예시 (--requirements 미지정 시 사용)
_DEFAULT_REQUIREMENTS: dict[str, str] = {
    "python": (
        "src/math_tool.py 에 add, subtract, multiply, divide 함수를 구현하고, "
        "각 함수에 대한 단위 테스트를 tests/test_math_tool.py 에 작성하고, "
        "src/run.py 에서 모든 함수를 실행해 결과를 출력하는 CLI를 만들어줘."
    ),
    "bash": (
        "scripts/backup.sh 를 작성해줘. "
        "SOURCE_DIR 과 DEST_DIR 환경변수를 읽어 타임스탬프 디렉터리로 백업하고, "
        "오래된 백업(30일 초과)을 자동 삭제하며 scripts/utils.sh 에 공통 로깅 함수를 분리해줘."
    ),
    "c": (
        "src/linked_list.h 와 src/linked_list.c 에 단방향 링크드 리스트를 구현하고 "
        "(push_front, push_back, pop_front, find, free_list), "
        "src/main.c 에서 모든 함수를 테스트하는 드라이버를 작성해줘."
    ),
    "cpp": (
        "include/stack.hpp 에 제네릭 Stack<T> 클래스 템플릿을 구현하고 "
        "(push, pop, top, empty, size), "
        "src/main.cpp 에서 int 와 std::string 타입으로 동작을 검증해줘."
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harness – LLM-driven code-generation agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --language python\n"
            "  python main.py --language bash   --requirements '로그 정리 스크립트'\n"
            "  python main.py --language c      --dry-run\n"
            "  python main.py --language cpp    --log-format console\n"
        ),
    )
    parser.add_argument(
        "--requirements", "-r",
        type=str, default=None,
        help="Free-text project requirements (overrides built-in default).",
    )
    parser.add_argument(
        "--language", "-l",
        choices=["python", "bash", "c", "cpp"],
        default=None,
        help="Target language (overrides TARGET_LANGUAGE in .env).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true", default=None,
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
        type=str, default=None,
        help="Override the workspace directory.",
    )
    return parser.parse_args()


def _print_summary_table(summary) -> None:
    table = Table(
        title=f"Run Summary  [dim]({summary.run_id})[/dim]",
        show_lines=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    status_color = {
        "completed": "green",
        "failed":    "red",
        "aborted":   "yellow",
    }.get(summary.status.value, "white")

    table.add_row(
        "Status",
        f"[{status_color}]{summary.status.value.upper()}[/{status_color}]",
    )
    table.add_row("Total tasks", str(summary.total_tasks))
    table.add_row("Completed",   f"[green]{summary.completed_tasks}[/green]")
    table.add_row("Failed",      f"[red]{summary.failed_tasks}[/red]")
    table.add_row("Skipped",     str(summary.skipped_tasks))

    def _verdict(val: bool | None) -> str:
        if val is True:
            return "[green]PASS[/green]"
        if val is False:
            return "[red]FAIL[/red]"
        return "[dim]SKIP[/dim]"

    table.add_row("Build",      _verdict(summary.build_passed))
    table.add_row("Execution",  _verdict(summary.execution_passed))
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

    # ── CLI 오버라이드를 환경변수로 주입 (settings 로드 전) ─────────────────────
    import os
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
    if args.log_format:
        os.environ["LOG_FORMAT"] = args.log_format
    if args.language:
        os.environ["TARGET_LANGUAGE"] = args.language

    # ── 로깅 설정 ─────────────────────────────────────────────────────────────
    from config.logging import configure_logging
    from config.settings import settings

    run_id = configure_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        run_id=settings.run_id,
    )

    # ── Orchestrator 조립 ─────────────────────────────────────────────────────
    from agent.container import build_orchestrator
    orchestrator, run_id = build_orchestrator(
        workspace_dir=args.workspace,
        run_id=run_id,
    )

    # ── 요구사항 결정 ─────────────────────────────────────────────────────────
    lang = settings.target_language.value
    requirements = (
        args.requirements
        or _DEFAULT_REQUIREMENTS.get(lang, "주어진 요구사항을 구현해줘.")
    )

    console.print(
        Panel.fit(
            f"[bold cyan]🚀 Harness Agent[/bold cyan]  "
            f"[dim]run_id={run_id}[/dim]\n"
            f"[dim]language={lang}  "
            f"dry_run={settings.dry_run}  "
            f"model={settings.ai_model}[/dim]",
            border_style="cyan",
        )
    )

    # ── 실행 ─────────────────────────────────────────────────────────────────
    try:
        summary = orchestrator.execute(requirements)
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted by user.[/yellow]")
        return 130
    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[bold red]Fatal error:[/bold red] {exc}")
        return 1

    _print_summary_table(summary)
    return 0 if summary.status.value == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
