"""
C / C++ ExecutionValidator.

Build 단계
----------
  gcc/g++ 로 소스 전체를 컴파일해 바이너리를 생성한다.
  단일 파일 프로젝트와 다중 파일 프로젝트 모두 지원한다.
  컴파일 결과물은 workspace_dir/.build/ 에 저장된다.

Smoke-test 단계
---------------
  컴파일된 바이너리를 실행하고 exit code 0 을 확인한다.
  실행 타임아웃은 run_timeout_seconds 로 제한한다.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import structlog

from agent.domain.entities import BuildResult, ExecutionResult
from agent.domain.interfaces import IExecutionValidator
from agent.infrastructure.validators.python_validator import (
    _DRY_RUN_BUILD,
    _DRY_RUN_EXEC,
    _run_command,
)

log = structlog.get_logger(__name__)


class CExecutionValidator(IExecutionValidator):
    """
    C 프로젝트 빌드(gcc) + 실행 검증기.

    Parameters
    ----------
    std:
        C 표준 (기본 "c17").
    extra_flags:
        gcc 추가 플래그.
    build_timeout / run_timeout:
        각 단계 타임아웃 (초).
    """

    def __init__(
        self,
        std: str = "c17",
        extra_flags: list[str] | None = None,
        build_timeout: int = 60,
        run_timeout: int = 15,
    ) -> None:
        self._std = std
        self._extra_flags = extra_flags or ["-Wall", "-Wextra"]
        self._build_timeout = build_timeout
        self._run_timeout = run_timeout
        self._binary_path: str | None = None

    def build(self, workspace_dir: str, dry_run: bool) -> BuildResult:
        if dry_run:
            return _DRY_RUN_BUILD

        compiler = shutil.which("gcc") or shutil.which("cc")
        if not compiler:
            return BuildResult(
                passed=False,
                errors=["gcc/cc 를 찾을 수 없습니다."],
                tool="gcc",
            )

        src_files = [
            str(p) for p in sorted(Path(workspace_dir).rglob("*.c"))
            if "__pycache__" not in str(p) and ".build" not in str(p)
        ]
        if not src_files:
            return BuildResult(
                passed=False, errors=["워크스페이스에 .c 파일이 없습니다."], tool="gcc"
            )

        build_dir = os.path.join(workspace_dir, ".build")
        os.makedirs(build_dir, exist_ok=True)
        binary = os.path.join(build_dir, "program")
        self._binary_path = binary

        # include 경로: 모든 소스 디렉터리
        include_dirs = {os.path.dirname(f) for f in src_files}
        inc_flags = [f"-I{d}" for d in include_dirs]

        cmd = (
            [compiler, f"-std={self._std}"]
            + self._extra_flags
            + inc_flags
            + src_files
            + ["-o", binary]
        )
        return _build_with_compiler(cmd, "gcc", self._build_timeout, workspace_dir)

    def run_smoke_test(self, workspace_dir: str, dry_run: bool) -> ExecutionResult:
        if dry_run:
            return _DRY_RUN_EXEC
        if not self._binary_path or not os.path.isfile(self._binary_path):
            return ExecutionResult(
                passed=False, exit_code=-1, command="",
                error_summary="빌드 결과물을 찾을 수 없습니다. build() 를 먼저 실행하세요.",
            )
        return _run_command([self._binary_path], workspace_dir, self._run_timeout)


class CppExecutionValidator(IExecutionValidator):
    """
    C++ 프로젝트 빌드(g++) + 실행 검증기.

    Parameters
    ----------
    std:
        C++ 표준 (기본 "c++17").
    extra_flags / build_timeout / run_timeout:
        CExecutionValidator 와 동일.
    """

    def __init__(
        self,
        std: str = "c++17",
        extra_flags: list[str] | None = None,
        build_timeout: int = 60,
        run_timeout: int = 15,
    ) -> None:
        self._std = std
        self._extra_flags = extra_flags or ["-Wall", "-Wextra"]
        self._build_timeout = build_timeout
        self._run_timeout = run_timeout
        self._binary_path: str | None = None

    def build(self, workspace_dir: str, dry_run: bool) -> BuildResult:
        if dry_run:
            return _DRY_RUN_BUILD

        compiler = shutil.which("g++") or shutil.which("c++")
        if not compiler:
            return BuildResult(
                passed=False, errors=["g++/c++ 를 찾을 수 없습니다."], tool="g++"
            )

        src_files = [
            str(p) for p in sorted(Path(workspace_dir).rglob("*.cpp"))
            if ".build" not in str(p)
        ]
        src_files += [
            str(p) for p in sorted(Path(workspace_dir).rglob("*.cc"))
            if ".build" not in str(p)
        ]
        if not src_files:
            return BuildResult(
                passed=False, errors=["워크스페이스에 .cpp/.cc 파일이 없습니다."], tool="g++"
            )

        build_dir = os.path.join(workspace_dir, ".build")
        os.makedirs(build_dir, exist_ok=True)
        binary = os.path.join(build_dir, "program")
        self._binary_path = binary

        include_dirs = {os.path.dirname(f) for f in src_files}
        # 헤더 디렉터리도 포함
        for p in Path(workspace_dir).rglob("*.hpp"):
            if ".build" not in str(p):
                include_dirs.add(str(p.parent))

        inc_flags = [f"-I{d}" for d in include_dirs]

        cmd = (
            [compiler, f"-std={self._std}"]
            + self._extra_flags
            + inc_flags
            + src_files
            + ["-o", binary]
        )
        return _build_with_compiler(cmd, "g++", self._build_timeout, workspace_dir)

    def run_smoke_test(self, workspace_dir: str, dry_run: bool) -> ExecutionResult:
        if dry_run:
            return _DRY_RUN_EXEC
        if not self._binary_path or not os.path.isfile(self._binary_path):
            return ExecutionResult(
                passed=False, exit_code=-1, command="",
                error_summary="빌드 결과물을 찾을 수 없습니다. build() 를 먼저 실행하세요.",
            )
        return _run_command([self._binary_path], workspace_dir, self._run_timeout)


# ── 공용 헬퍼 ─────────────────────────────────────────────────────────────────

def _build_with_compiler(
    cmd: list[str],
    tool: str,
    timeout: int,
    workspace_dir: str,
) -> BuildResult:
    """컴파일러 명령을 실행하고 BuildResult 로 변환한다."""
    cmd_str = " ".join(cmd)
    log.info("validator.build.compile", command=cmd_str[:120])
    try:
        result = subprocess.run(
            cmd,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = result.returncode == 0
        errors = []
        if not passed:
            raw = result.stderr.strip() or result.stdout.strip()
            errors = [line for line in raw.splitlines() if line.strip()]
        log.info(
            "validator.build.done",
            tool=tool,
            passed=passed,
            error_count=len(errors),
        )
        return BuildResult(
            passed=passed,
            errors=errors,
            stdout=result.stdout[:2000],
            stderr=result.stderr[:2000],
            tool=tool,
        )
    except subprocess.TimeoutExpired:
        return BuildResult(
            passed=False,
            errors=[f"컴파일 타임아웃 ({timeout}s)"],
            tool=tool,
        )
    except Exception as exc:  # noqa: BLE001
        return BuildResult(
            passed=False, errors=[str(exc)], tool=tool
        )
