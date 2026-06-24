"""
Code-quality sensor adapters.

Two concrete implementations are provided:

* ``RuffSensorAdapter``   – preferred; much faster than flake8.
* ``Flake8SensorAdapter`` – fallback when ruff is unavailable.

``SmartLintSensor`` auto-selects the available backend at runtime.
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


def _run_linter(
    cmd: list[str],
    tool: str,
    absolute_path: str,
) -> LintResult:
    """Execute *cmd* and parse stdout into a :class:`LintResult`."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
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
    """
    Lint sensor using ``ruff`` (Rust-based, ~100x faster than flake8).

    Parameters
    ----------
    ignore_codes:
        Comma-separated rule codes to suppress (e.g. "E501,W292").
    """

    def __init__(self, ignore_codes: str = "E501,W292,W391") -> None:
        self._ignore = ignore_codes

    def verify_code(self, absolute_path: str, dry_run: bool) -> LintResult:
        if dry_run:
            log.debug("sensor.dry_run", path=absolute_path)
            return _DRY_RUN_RESULT
        cmd = ["ruff", "check", absolute_path, "--select=E,W,F"]
        if self._ignore:
            cmd += ["--ignore", self._ignore]
        return _run_linter(cmd, "ruff", absolute_path)


class Flake8SensorAdapter(ISensorAdapter):
    """
    Lint sensor using ``flake8``.

    Parameters
    ----------
    ignore_codes:
        Comma-separated error codes to ignore.
    """

    def __init__(self, ignore_codes: str = "E501,W292,W391") -> None:
        self._ignore = ignore_codes

    def verify_code(self, absolute_path: str, dry_run: bool) -> LintResult:
        if dry_run:
            log.debug("sensor.dry_run", path=absolute_path)
            return _DRY_RUN_RESULT
        cmd = ["flake8", absolute_path]
        if self._ignore:
            cmd += [f"--ignore={self._ignore}"]
        return _run_linter(cmd, "flake8", absolute_path)


class SmartLintSensor(ISensorAdapter):
    """
    Auto-selects the best available lint backend at runtime.

    Priority: ruff > flake8.
    Raises :class:`SensorError` if neither is installed.
    """

    def __init__(self, ignore_codes: str = "E501,W292,W391") -> None:
        self._delegate: Optional[ISensorAdapter] = None
        self._ignore = ignore_codes

    def _get_delegate(self) -> ISensorAdapter:
        if self._delegate is not None:
            return self._delegate
        if shutil.which("ruff"):
            log.info("sensor.backend.selected", backend="ruff")
            self._delegate = RuffSensorAdapter(self._ignore)
        elif shutil.which("flake8"):
            log.info("sensor.backend.selected", backend="flake8")
            self._delegate = Flake8SensorAdapter(self._ignore)
        else:
            raise SensorError(
                "No lint backend found. Install ruff ('pip install ruff') "
                "or flake8 ('pip install flake8')."
            )
        return self._delegate

    def verify_code(self, absolute_path: str, dry_run: bool) -> LintResult:
        return self._get_delegate().verify_code(absolute_path, dry_run)
