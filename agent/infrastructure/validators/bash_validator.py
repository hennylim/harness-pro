"""
Bash ExecutionValidator.

Build 단계: bash -n 으로 모든 .sh 파일 문법 검사.
Smoke-test: 진입점을 동적으로 탐색한 뒤 실행.
  1. 고정 후보(main.sh 등) 확인
  2. 없으면 파일명·shebang·main() 호출 패턴으로 자동 탐색
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import structlog

from agent.domain.entities import BuildResult, ExecutionResult
from agent.domain.interfaces import IExecutionValidator
from agent.infrastructure.validators.entry_point_finder import find_bash_entry
from agent.infrastructure.validators.python_validator import (
    _DRY_RUN_BUILD,
    _DRY_RUN_EXEC,
    _run_command,
)

log = structlog.get_logger(__name__)


class BashExecutionValidator(IExecutionValidator):
    """
    Bash 스크립트 프로젝트 빌드 + 실행 검증기.

    Parameters
    ----------
    build_timeout:
        bash -n 전체 타임아웃 (초).
    run_timeout:
        진입점 실행 타임아웃 (초).
    run_flags:
        진입점 실행 시 전달할 플래그 (기본: ["--help"]).
    extra_entry_candidates:
        고정 후보 목록을 추가로 지정 (동적 탐색 전에 먼저 확인).
    """

    def __init__(
        self,
        build_timeout: int = 30,
        run_timeout: int = 15,
        run_flags: list[str] | None = None,
        extra_entry_candidates: list[str] | None = None,
    ) -> None:
        self._build_timeout = build_timeout
        self._run_timeout = run_timeout
        self._run_flags = run_flags if run_flags is not None else ["--help"]
        self._extra_candidates = extra_entry_candidates or []

    # ── IExecutionValidator ───────────────────────────────────────────────────

    def build(self, workspace_dir: str, dry_run: bool) -> BuildResult:
        """모든 .sh 파일을 bash -n 으로 문법 검사."""
        if dry_run:
            log.debug("validator.build.dry_run", lang="bash")
            return _DRY_RUN_BUILD

        sh_files = sorted(Path(workspace_dir).rglob("*.sh"))
        if not sh_files:
            return BuildResult(
                passed=False,
                errors=["워크스페이스에 .sh 파일이 없습니다."],
                tool="bash -n",
            )

        errors: list[str] = []
        for sh_file in sh_files:
            result = _run_command(
                ["bash", "-n", str(sh_file)],
                workspace_dir,
                self._build_timeout,
            )
            if not result.passed:
                rel = os.path.relpath(str(sh_file), workspace_dir)
                errors.append(
                    f"[{rel}] bash -n 실패:\n"
                    + (result.stderr or result.error_summary)[:300]
                )

        passed = len(errors) == 0
        log.info(
            "validator.build.done",
            lang="bash",
            files=len(sh_files),
            errors=len(errors),
            passed=passed,
        )
        return BuildResult(passed=passed, errors=errors, tool="bash -n")

    def run_smoke_test(self, workspace_dir: str, dry_run: bool) -> ExecutionResult:
        """진입점 스크립트를 동적으로 탐색한 후 실행한다."""
        if dry_run:
            log.debug("validator.run.dry_run", lang="bash")
            return _DRY_RUN_EXEC

        entry = find_bash_entry(workspace_dir, self._extra_candidates or None)

        if entry is None:
            # 워크스페이스의 실제 파일 목록을 에러에 포함
            found = sorted(str(p) for p in Path(workspace_dir).rglob("*.sh"))
            return ExecutionResult(
                passed=False,
                exit_code=-1,
                command="",
                error_summary=(
                    "진입점 .sh 파일을 찾을 수 없습니다.\n"
                    f"워크스페이스 내 .sh 파일: {found}"
                ),
            )

        rel_entry = os.path.relpath(entry, workspace_dir)
        log.info("validator.run.entry_selected", entry=rel_entry)

        # 실행 권한 부여
        try:
            current = os.stat(entry).st_mode
            os.chmod(entry, current | stat.S_IXUSR | stat.S_IXGRP)
        except OSError:
            pass

        # 1차: --help 플래그로 시도
        if self._run_flags:
            cmd = ["bash", entry] + self._run_flags
            result = _run_command(cmd, workspace_dir, self._run_timeout)
            if result.passed:
                return result
            log.debug("validator.run.help_failed", entry=rel_entry)

        # 2차: stdin=/dev/null 으로 비대화형 실행
        log.debug("validator.run.devnull_attempt", entry=rel_entry)
        try:
            proc = subprocess.run(
                ["bash", entry],
                cwd=workspace_dir,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self._run_timeout,
            )
            return ExecutionResult(
                passed=proc.returncode == 0,
                exit_code=proc.returncode,
                stdout=proc.stdout[:2000],
                stderr=proc.stderr[:2000],
                command=f"bash {rel_entry} (stdin=/dev/null)",
                error_summary=(
                    "" if proc.returncode == 0
                    else f"exit code {proc.returncode}\n"
                    + (proc.stderr or proc.stdout)[:300]
                ),
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                passed=False,
                exit_code=-1,
                command=f"bash {rel_entry}",
                error_summary=f"실행 타임아웃 ({self._run_timeout}s)",
            )
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                passed=False,
                exit_code=-1,
                command=f"bash {rel_entry}",
                error_summary=str(exc),
            )
