"""
LanguageSensorRouter – 파일 확장자를 보고 올바른 ISensorAdapter 를 선택한다.

새 언어를 추가할 때 EXTENSION_MAP 에 항목만 추가하면 된다.
Orchestrator / LLM 어댑터는 전혀 수정하지 않아도 된다.

등록 규칙
---------
* 키: 소문자 파일 확장자 (점 포함), 예: ".py", ".sh", ".c"
* 값: ISensorAdapter 인스턴스 (싱글턴; 상태를 갖지 않으므로 공유 안전)
"""
from __future__ import annotations

import os

import structlog

from agent.domain.entities import LintResult
from agent.domain.exceptions import SensorError
from agent.domain.interfaces import ISensorAdapter
from agent.infrastructure.sensors.bash_sensor import ShellCheckSensorAdapter
from agent.infrastructure.sensors.c_sensor import CSensorAdapter
from agent.infrastructure.sensors.cpp_sensor import CppSensorAdapter
from agent.infrastructure.sensors.lint_sensor import (
    SmartLintSensor,
    _DRY_RUN_RESULT,
)

log = structlog.get_logger(__name__)

# ── 확장자 → 센서 매핑 테이블 ─────────────────────────────────────────────────
# 새 언어 추가: 이 딕셔너리에 항목 1개 추가
EXTENSION_MAP: dict[str, ISensorAdapter] = {
    # Python
    ".py":  SmartLintSensor(),
    # Bash / Shell
    ".sh":  ShellCheckSensorAdapter(),
    # C
    ".c":   CSensorAdapter(),
    ".h":   CSensorAdapter(),
    # C++
    ".cpp": CppSensorAdapter(),
    ".cc":  CppSensorAdapter(),
    ".cxx": CppSensorAdapter(),
    ".hpp": CppSensorAdapter(),
}


class LanguageSensorRouter(ISensorAdapter):
    """
    파일 경로의 확장자를 보고 적절한 센서로 위임한다.

    알 수 없는 확장자는 dry_run=False 일 때 경고를 남기고 통과시킨다
    (알 수 없는 언어를 무조건 실패시키면 다국어 프로젝트에서 방해가 됨).
    """

    def __init__(self, extension_map: dict[str, ISensorAdapter] | None = None) -> None:
        self._map = extension_map or EXTENSION_MAP

    def verify_code(self, absolute_path: str, dry_run: bool) -> LintResult:
        ext = os.path.splitext(absolute_path)[1].lower()
        sensor = self._map.get(ext)

        if sensor is None:
            log.warning(
                "sensor.router.unknown_extension",
                ext=ext,
                path=absolute_path,
                action="skip_lint",
            )
            return LintResult(passed=True, tool="no-op", warnings=[
                f"No sensor registered for extension '{ext}'; lint skipped."
            ])

        log.debug("sensor.router.dispatch", ext=ext, sensor=type(sensor).__name__)
        return sensor.verify_code(absolute_path, dry_run)
