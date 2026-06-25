"""
Bash ExecutionValidator.

Build 단계
----------
  bash -n <script> 로 모든 .sh 파일의 문법을 검사한다.
  shellcheck 와 달리 bash 자체 파서로 검사하므로 실행 환경과 동일한 결과를 보장한다.

Smoke-test 단계
---------------
  진입점 스크립트(main.sh / src/main.sh)를 실제로 실행한다.
  대화형 입력이 필요한 스크립트는 /dev/null 을 stdin 으로 연결하고,
  --dry-run / --help 등의 플래그를 우선 시도한다.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import structlog

from agent.domain.entities import BuildResult, ExecutionResult
from agent.domain.interfaces import IExecutionValidator
from agent.infrastructure.validators.python_validator import (
    _DRY_RUN_BUILD,
    _DRY_RUN_EXEC,
    _find_entry_point,
    _run_command,
)

log = structlog.get_logger(__name__)

_ENTRY_CANDIDATES = [
    "src/main.sh",
    "main.sh",
    "src/run.sh",
    "run.sh",
]


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
        진입점 실행 시 전달할 플래그 (기본: ["--help"] → 없으면 빈 stdin).
    """

    def __init__(
        self,
        build_timeout: int = 30,
        run_timeout: int = 15,
        run_flags: list[str] | None = None,
    ) -> None:
        self._build_timeout = build_timeout
        self._run_timeout = run_timeout
        # --help 가 있으면 대화형 입력 없이 종료 가능
        self._run_flags = run_flags if run_flags is not None else ["--help"]

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
        """진입점 스크립트를 실제로 실행한다."""
        if dry_run:
            log.debug("validator.run.dry_run", lang="bash")
            return _DRY_RUN_EXEC

        entry = _find_entry_point(workspace_dir, _ENTRY_CANDIDATES)
        if entry is None:
            return ExecutionResult(
                passed=False,
                exit_code=-1,
                command="",
                error_summary=(
                    f"진입점을 찾을 수 없습니다. 후보: {_ENTRY_CANDIDATES}"
                ),
            )

        # 실행 권한 부여
        try:
            current = os.stat(entry).st_mode
            os.chmod(entry, current | stat.S_IXUSR | stat.S_IXGRP)
        except OSError:
            pass

        # --help 플래그로 먼저 시도, 실패하면 빈 stdin 으로 재시도
        cmd = ["bash", entry] + self._run_flags
        result = _run_command(cmd, workspace_dir, self._run_timeout)

        if not result.passed and self._run_flags:
            # --help 를 지원하지 않는 스크립트 → stdin 을 /dev/null 로 연결
            import subprocess
            try:
                proc = subprocess.run(
                    ["bash", entry],
                    cwd=workspace_dir,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=self._run_timeout,
                )
                if proc.returncode == 0:
                    return ExecutionResult(
                        passed=True,
                        exit_code=0,
                        stdout=proc.stdout[:2000],
                        stderr=proc.stderr[:2000],
                        command=f"bash {entry} (stdin=/dev/null)",
                    )
            except Exception:  # noqa: BLE001
                pass

        return result
