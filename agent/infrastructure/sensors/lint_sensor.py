"""
Python 코드 품질 센서 (ruff / flake8).

새 언어를 추가할 때 이 파일은 건드리지 않는다.
언어별 센서는 별도 파일에 작성하고 router.py 에 등록한다.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Optional

import structlog

from agent.domain.entities import LintResult
from agent.domain.exceptions import SensorError
from agent.domain.interfaces import ISensorAdapter

log = structlog.get_logger(__name__)

_DRY_RUN_RESULT = LintResult(passed=True, tool="dry_run")


def run_subprocess_sensor(
    cmd: list[str],
    tool: str,
    absolute_path: str,
    timeout: int = 30,
) -> LintResult:
    """
    공용 헬퍼: 외부 프로세스를 실행하고 결과를 LintResult 로 변환한다.
    센서 구현체들이 공유해서 사용한다.
    """
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SensorError(f"{tool} timed out on {absolute_path}") from exc
    except FileNotFoundError as exc:
        raise SensorError(f"{tool} executable not found: {exc}") from exc

    if result.returncode == 0:
        log.debug("sensor.lint.pass", tool=tool, path=absolute_path)
        return LintResult(passed=True, tool=tool)

    raw = result.stdout.strip() or result.stderr.strip()
    errors = [line for line in raw.splitlines() if line.strip()]
    log.debug("sensor.lint.fail", tool=tool, error_count=len(errors), path=absolute_path)
    return LintResult(passed=False, errors=errors, tool=tool)


class RuffSensorAdapter(ISensorAdapter):
    """ruff 기반 Python 린트 센서."""

    def __init__(self, ignore_codes: str = "E501,W292,W391") -> None:
        self._ignore = ignore_codes

    def verify_code(self, absolute_path: str, dry_run: bool) -> LintResult:
        if dry_run:
            return _DRY_RUN_RESULT
        cmd = ["ruff", "check", absolute_path, "--select=E,W,F"]
        if self._ignore:
            cmd += ["--ignore", self._ignore]
        return run_subprocess_sensor(cmd, "ruff", absolute_path)


class Flake8SensorAdapter(ISensorAdapter):
    """flake8 기반 Python 린트 센서."""

    def __init__(self, ignore_codes: str = "E501,W292,W391") -> None:
        self._ignore = ignore_codes

    def verify_code(self, absolute_path: str, dry_run: bool) -> LintResult:
        if dry_run:
            return _DRY_RUN_RESULT
        cmd = ["flake8", absolute_path]
        if self._ignore:
            cmd += [f"--ignore={self._ignore}"]
        return run_subprocess_sensor(cmd, "flake8", absolute_path)


class SmartLintSensor(ISensorAdapter):
    """ruff → flake8 순서로 자동 선택하는 Python 센서."""

    def __init__(self, ignore_codes: str = "E501,W292,W391") -> None:
        self._delegate: Optional[ISensorAdapter] = None
        self._ignore = ignore_codes

    def _get_delegate(self) -> ISensorAdapter:
        if self._delegate is not None:
            return self._delegate
        if shutil.which("ruff"):
            log.info("sensor.backend.selected", backend="ruff", lang="python")
            self._delegate = RuffSensorAdapter(self._ignore)
        elif shutil.which("flake8"):
            log.info("sensor.backend.selected", backend="flake8", lang="python")
            self._delegate = Flake8SensorAdapter(self._ignore)
        else:
            raise SensorError(
                "Python 린트 백엔드를 찾을 수 없습니다. "
                "ruff(pip install ruff) 또는 flake8(pip install flake8)을 설치하세요."
            )
        return self._delegate

    def verify_code(self, absolute_path: str, dry_run: bool) -> LintResult:
        return self._get_delegate().verify_code(absolute_path, dry_run)
