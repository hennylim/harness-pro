"""
Bash 코드 품질 센서.

도구: shellcheck (https://www.shellcheck.net/)
설치: apt install shellcheck  /  brew install shellcheck

새 언어 센서 작성 시 이 파일을 템플릿으로 참고한다.
"""
from __future__ import annotations

import os
import shutil
import stat

import structlog

from agent.domain.entities import LintResult
from agent.domain.exceptions import SensorError
from agent.domain.interfaces import ISensorAdapter
from agent.infrastructure.sensors.lint_sensor import (
    _DRY_RUN_RESULT,
    run_subprocess_sensor,
)

log = structlog.get_logger(__name__)


class ShellCheckSensorAdapter(ISensorAdapter):
    """
    shellcheck 를 사용한 Bash/Shell 스크립트 정적 분석 센서.

    Parameters
    ----------
    severity:
        리포트할 최소 심각도 ("error" | "warning" | "info" | "style").
        기본값 "warning" – style 제안은 무시하고 실질적 버그만 잡는다.
    shell:
        대상 셸 지정 ("bash" | "sh" | "dash" | "ksh").
    """

    def __init__(
        self,
        severity: str = "warning",
        shell: str = "bash",
    ) -> None:
        self._severity = severity
        self._shell = shell

    def verify_code(self, absolute_path: str, dry_run: bool) -> LintResult:
        if dry_run:
            log.debug("sensor.dry_run", path=absolute_path, tool="shellcheck")
            return _DRY_RUN_RESULT

        if not shutil.which("shellcheck"):
            raise SensorError(
                "shellcheck 를 찾을 수 없습니다. "
                "'apt install shellcheck' 또는 'brew install shellcheck' 로 설치하세요."
            )

        # shellcheck 실행 전 파일에 실행 권한 부여 (없으면 일부 경고 발생)
        try:
            current = os.stat(absolute_path).st_mode
            os.chmod(absolute_path, current | stat.S_IXUSR)
        except OSError:
            pass

        cmd = [
            "shellcheck",
            f"--severity={self._severity}",
            f"--shell={self._shell}",
            "--format=gcc",   # gcc 포맷 → 다른 도구와 일관된 출력
            absolute_path,
        ]
        return run_subprocess_sensor(cmd, "shellcheck", absolute_path)
