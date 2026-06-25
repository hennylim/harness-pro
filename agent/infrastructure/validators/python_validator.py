"""
Python ExecutionValidator.

빌드(문법 검사) → 실행(python 진입점) 두 단계로 검증한다.

Build 단계
----------
  py_compile 로 워크스페이스의 모든 .py 파일을 컴파일해 문법 오류를 잡는다.
  개별 lint 를 통과한 파일이라도 서로 import 할 때 문제가 생길 수 있으므로
  여기서 전체 컴파일 검사를 수행한다.

Smoke-test 단계
---------------
  워크스페이스에서 진입점(main.py / src/main.py 등)을 찾아 실행한다.
  --smoke-test 플래그를 지원하면 해당 플래그로 실행하고,
  아니면 단순 python <entry> 로 실행해 exit code 0 을 확인한다.

  실행 시간은 run_timeout_seconds 로 제한한다.
"""
from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path

import structlog

from agent.domain.entities import BuildResult, ExecutionResult
from agent.domain.interfaces import IExecutionValidator

log = structlog.get_logger(__name__)

_DRY_RUN_BUILD = BuildResult(passed=True, tool="dry_run")
_DRY_RUN_EXEC = ExecutionResult(
    passed=True, exit_code=0, command="dry_run",
    stdout="[dry-run] execution skipped"
)

# 진입점 후보 (우선순위 순)
_ENTRY_CANDIDATES = [
    "src/main.py",
    "main.py",
    "src/app.py",
    "app.py",
    "src/run.py",
    "run.py",
]


class PythonExecutionValidator(IExecutionValidator):
    """
    Python 프로젝트 빌드 + 실행 검증기.

    Parameters
    ----------
    build_timeout:
        py_compile 전체 타임아웃 (초).
    run_timeout:
        진입점 실행 타임아웃 (초).
    extra_run_args:
        진입점 실행 시 추가 인자 (예: ["--help"]).
    """

    def __init__(
        self,
        build_timeout: int = 60,
        run_timeout: int = 30,
        extra_run_args: list[str] | None = None,
    ) -> None:
        self._build_timeout = build_timeout
        self._run_timeout = run_timeout
        self._extra_run_args = extra_run_args or []

    # ── IExecutionValidator ───────────────────────────────────────────────────

    def build(self, workspace_dir: str, dry_run: bool) -> BuildResult:
        """워크스페이스의 모든 .py 파일을 py_compile 로 문법 검사."""
        if dry_run:
            log.debug("validator.build.dry_run", lang="python")
            return _DRY_RUN_BUILD

        py_files = sorted(Path(workspace_dir).rglob("*.py"))
        if not py_files:
            return BuildResult(
                passed=False,
                errors=["워크스페이스에 .py 파일이 없습니다."],
                tool="py_compile",
            )

        errors: list[str] = []
        for py_file in py_files:
            # __pycache__ 건너뜀
            if "__pycache__" in str(py_file):
                continue
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", str(py_file)],
                    capture_output=True,
                    text=True,
                    timeout=self._build_timeout,
                )
                if result.returncode != 0:
                    rel = os.path.relpath(str(py_file), workspace_dir)
                    errors.append(f"[{rel}] {result.stderr.strip()}")
            except subprocess.TimeoutExpired:
                errors.append(f"[{py_file.name}] py_compile timed out")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"[{py_file.name}] {exc}")

        passed = len(errors) == 0
        log.info(
            "validator.build.done",
            lang="python",
            files=len(py_files),
            errors=len(errors),
            passed=passed,
        )
        return BuildResult(passed=passed, errors=errors, tool="py_compile")

    def run_smoke_test(self, workspace_dir: str, dry_run: bool) -> ExecutionResult:
        """진입점을 찾아 python 으로 실행하고 exit code 를 확인한다."""
        if dry_run:
            log.debug("validator.run.dry_run", lang="python")
            return _DRY_RUN_EXEC

        entry = _find_entry_point(workspace_dir, _ENTRY_CANDIDATES)
        if entry is None:
            return ExecutionResult(
                passed=False,
                exit_code=-1,
                command="",
                error_summary=(
                    f"진입점을 찾을 수 없습니다. "
                    f"후보: {_ENTRY_CANDIDATES}"
                ),
            )

        cmd = ["python3", entry] + self._extra_run_args
        return _run_command(cmd, workspace_dir, self._run_timeout)


# ── 공용 헬퍼 ─────────────────────────────────────────────────────────────────

def _find_entry_point(workspace_dir: str, candidates: list[str]) -> str | None:
    """후보 목록에서 실제로 존재하는 첫 번째 파일 경로를 반환."""
    for candidate in candidates:
        full = os.path.join(workspace_dir, candidate)
        if os.path.isfile(full):
            return full
    return None


def _run_command(
    cmd: list[str],
    cwd: str,
    timeout: int,
) -> ExecutionResult:
    """명령어를 실행하고 ExecutionResult 로 변환한다."""
    cmd_str = " ".join(cmd)
    log.info("validator.run.exec", command=cmd_str, cwd=cwd)
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = result.returncode == 0
        error_summary = ""
        if not passed:
            error_summary = (
                f"exit code {result.returncode}\n"
                + (result.stderr.strip() or result.stdout.strip())[:500]
            )
        log.info(
            "validator.run.done",
            command=cmd_str,
            exit_code=result.returncode,
            passed=passed,
        )
        return ExecutionResult(
            passed=passed,
            exit_code=result.returncode,
            stdout=result.stdout[:2000],
            stderr=result.stderr[:2000],
            command=cmd_str,
            error_summary=error_summary,
        )
    except subprocess.TimeoutExpired:
        log.warning("validator.run.timeout", command=cmd_str, timeout=timeout)
        return ExecutionResult(
            passed=False,
            exit_code=-1,
            command=cmd_str,
            error_summary=f"실행 타임아웃 ({timeout}s): {cmd_str}",
        )
    except Exception as exc:  # noqa: BLE001
        log.error("validator.run.error", command=cmd_str, error=str(exc))
        return ExecutionResult(
            passed=False,
            exit_code=-1,
            command=cmd_str,
            error_summary=f"실행 오류: {exc}",
        )
