"""
C++ 코드 품질 센서.

단계 1: g++ -fsyntax-only  → 컴파일 에러/경고 즉시 확인
단계 2: clang-tidy          → 정적 분석 (설치된 경우)

헤더 파일(.hpp/.h)은 문법 검사만 수행한다.
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

_HEADER_EXTS = frozenset({".hpp", ".h"})

_TIDY_CHECKS = ",".join([
    "clang-diagnostic-*",
    "clang-analyzer-*",
    "bugprone-*",
    "-bugprone-easily-swappable-parameters",
    # modernize/readability 는 노이즈가 많아 비활성화
    # (실제 코드 품질에 중요한 clang-diagnostic 과 analyzer 만 유지)
])

# warnings-as-errors 는 clang-diagnostic 만 적용 (스타일 경고 제외)
_TIDY_WERROR = "clang-diagnostic-*,clang-analyzer-*"


class CppSensorAdapter(ISensorAdapter):
    """
    C++ 소스 파일을 g++ + clang-tidy 로 검증하는 센서.

    Parameters
    ----------
    std:
        C++ 표준 (기본 "c++17").
    extra_flags:
        g++/clang++ 에 전달할 추가 컴파일 플래그.
    use_tidy:
        clang-tidy 를 사용할지 여부.
    """

    def __init__(
        self,
        std: str = "c++17",
        extra_flags: list[str] | None = None,
        use_tidy: bool = True,
    ) -> None:
        self._std = std
        self._extra_flags = extra_flags or ["-Wall", "-Wextra", "-Wpedantic"]
        self._use_tidy = use_tidy

    def verify_code(self, absolute_path: str, dry_run: bool) -> LintResult:
        if dry_run:
            log.debug("sensor.dry_run", path=absolute_path, tool="cpp-sensor")
            return _DRY_RUN_RESULT

        is_header = any(absolute_path.endswith(ext) for ext in _HEADER_EXTS)
        lang_flag = ["-x", "c++-header"] if is_header else []

        # ── Step 1: g++ 문법 검사 ──────────────────────────────────────────────
        compiler = shutil.which("g++") or shutil.which("c++")
        if not compiler:
            raise SensorError("C++ 컴파일러(g++/c++)를 찾을 수 없습니다.")

        gpp_cmd = [
            compiler,
            f"-std={self._std}",
            *self._extra_flags,
            *lang_flag,
            "-fsyntax-only",
            absolute_path,
        ]
        result = run_subprocess_sensor(gpp_cmd, "g++", absolute_path)
        if not result.passed:
            return result

        if is_header:
            return LintResult(passed=True, tool="g++")

        # ── Step 2: clang-tidy (옵션) ──────────────────────────────────────────
        if self._use_tidy and shutil.which("clang-tidy"):
            workspace_dir = os.path.dirname(absolute_path)
            tidy_cmd = [
                "clang-tidy",
                absolute_path,
                f"--checks={_TIDY_CHECKS}",
                f"--warnings-as-errors={_TIDY_WERROR}",
                "--",
                f"-std={self._std}",
                f"-I{workspace_dir}",
            ]
            tidy_result = run_subprocess_sensor(tidy_cmd, "clang-tidy", absolute_path)
            if not tidy_result.passed:
                return tidy_result

        return LintResult(passed=True, tool="g+++clang-tidy")
