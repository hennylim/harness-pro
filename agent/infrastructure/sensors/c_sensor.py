"""
C 코드 품질 센서.

단계 1: gcc -fsyntax-only  → 컴파일 에러/경고 즉시 확인
단계 2: clang-tidy          → 정적 분석 (설치된 경우)

헤더 파일(.h)은 문법 검사만 수행하고 tidy 는 건너뛴다.
"""
from __future__ import annotations

import os
import shutil

import structlog

from agent.domain.entities import LintResult
from agent.domain.exceptions import SensorError
from agent.domain.interfaces import ISensorAdapter
from agent.infrastructure.sensors.lint_sensor import (
    _DRY_RUN_RESULT,
    run_subprocess_sensor,
)

log = structlog.get_logger(__name__)

# clang-tidy 에서 활성화할 체크 그룹
_TIDY_CHECKS = ",".join([
    "clang-diagnostic-*",
    "clang-analyzer-*",
    "bugprone-*",
    "-bugprone-easily-swappable-parameters",
])

_TIDY_WERROR = "clang-diagnostic-*,clang-analyzer-*"


class CSensorAdapter(ISensorAdapter):
    """
    C 소스 파일을 gcc + clang-tidy 로 검증하는 센서.

    Parameters
    ----------
    std:
        C 표준 (기본 "c17").
    extra_flags:
        gcc/clang 에 전달할 추가 컴파일 플래그.
    use_tidy:
        clang-tidy 를 사용할지 여부 (설치된 경우에만 실행).
    """

    def __init__(
        self,
        std: str = "c17",
        extra_flags: list[str] | None = None,
        use_tidy: bool = True,
    ) -> None:
        self._std = std
        self._extra_flags = extra_flags or ["-Wall", "-Wextra", "-Wpedantic"]
        self._use_tidy = use_tidy

    def verify_code(self, absolute_path: str, dry_run: bool) -> LintResult:
        if dry_run:
            log.debug("sensor.dry_run", path=absolute_path, tool="c-sensor")
            return _DRY_RUN_RESULT

        # 헤더 파일은 -x c 로 언어를 명시해 문법 검사만 진행
        is_header = absolute_path.endswith(".h")
        lang_flag = ["-x", "c"] if is_header else []

        # ── Step 1: gcc 문법 검사 ──────────────────────────────────────────────
        compiler = shutil.which("gcc") or shutil.which("cc")
        if not compiler:
            raise SensorError("C 컴파일러(gcc/cc)를 찾을 수 없습니다.")

        gcc_cmd = [
            compiler,
            f"-std={self._std}",
            *self._extra_flags,
            *lang_flag,
            "-fsyntax-only",
            absolute_path,
        ]
        result = run_subprocess_sensor(gcc_cmd, "gcc", absolute_path)
        if not result.passed:
            return result

        # 헤더 파일은 tidy 건너뜀
        if is_header:
            return LintResult(passed=True, tool="gcc")

        # ── Step 2: clang-tidy (옵션) ──────────────────────────────────────────
        if self._use_tidy and shutil.which("clang-tidy"):
            # clang-tidy 는 compile_commands.json 없이도 -- 로 플래그를 전달 가능
            workspace_dir = os.path.dirname(absolute_path)
            tidy_cmd = [
                "clang-tidy",
                absolute_path,
                f"--checks={_TIDY_CHECKS}",
                f"--warnings-as-errors={_TIDY_WERROR}",
                "--",
                f"-std={self._std}",
                f"-I{workspace_dir}",   # 같은 디렉터리의 헤더 포함
            ]
            tidy_result = run_subprocess_sensor(tidy_cmd, "clang-tidy", absolute_path)
            if not tidy_result.passed:
                return tidy_result

        return LintResult(passed=True, tool="gcc+clang-tidy")
