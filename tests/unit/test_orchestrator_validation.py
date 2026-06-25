"""
Orchestrator 빌드/실행 검증 단계 단위 테스트.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.domain.entities import (
    BuildResult,
    ExecutionResult,
    LintResult,
    RunStatus,
    TaskAction,
)
from agent.domain.exceptions import PlanValidationError
from agent.usecase.orchestrator import HarnessOrchestrator
from config.logging import configure_logging

configure_logging(level="ERROR", fmt="console")


def _make_task(tid=1, path="src/a.py"):
    from agent.domain.entities import Task
    return Task(task_id=tid, file_path=path,
                action=TaskAction.CREATE, description="create")


def _make_plan(*paths):
    from agent.domain.entities import ProjectPlan
    return ProjectPlan(tasks=[_make_task(i+1, p) for i, p in enumerate(paths)])


def _make_patch(path="src/a.py"):
    from agent.domain.entities import CodePatch
    return CodePatch(target_file=path, explanation="ok", code_block="x = 1\n")


def _make_orch(
    llm=None, fs=None, sensor=None, run_repo=None,
    execution_validator=None,
    enable_build=True, enable_exec=True,
    dry_run=False,
):
    llm = llm or MagicMock()
    fs = fs or MagicMock()
    sensor = sensor or MagicMock()
    run_repo = run_repo or MagicMock()
    fs.backup_file.return_value = None
    fs.get_workspace_skeleton.return_value = ""
    fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"
    sensor.verify_code.return_value = LintResult(passed=True, tool="ruff")
    return HarnessOrchestrator(
        llm=llm, fs=fs, sensor=sensor, run_repo=run_repo,
        run_id="test", execution_validator=execution_validator,
        enable_build_validation=enable_build,
        enable_execution_validation=enable_exec,
        dry_run=dry_run,
    )


# ── 빌드 + 실행 모두 PASS ─────────────────────────────────────────────────────

class TestBuildExecPass:
    def test_summary_completed_when_all_pass(self):
        llm = MagicMock()
        llm.generate_plan.return_value = _make_plan("src/a.py")
        llm.generate_code.return_value = _make_patch("src/a.py")

        validator = MagicMock()
        validator.build.return_value = BuildResult(passed=True, tool="py_compile")
        validator.run_smoke_test.return_value = ExecutionResult(
            passed=True, exit_code=0, stdout="ok", command="python3 src/main.py"
        )

        orch = _make_orch(llm=llm, execution_validator=validator)
        summary = orch.execute("build something")

        assert summary.status == RunStatus.COMPLETED
        assert summary.build_passed is True
        assert summary.execution_passed is True
        validator.build.assert_called_once()
        validator.run_smoke_test.assert_called_once()

    def test_execution_stdout_logged(self):
        """smoke-test 의 stdout 이 기록되는지 확인."""
        llm = MagicMock()
        llm.generate_plan.return_value = _make_plan("src/a.py")
        llm.generate_code.return_value = _make_patch("src/a.py")

        validator = MagicMock()
        validator.build.return_value = BuildResult(passed=True, tool="gcc")
        validator.run_smoke_test.return_value = ExecutionResult(
            passed=True, exit_code=0, stdout="hello world\n", command="./program"
        )

        orch = _make_orch(llm=llm, execution_validator=validator)
        summary = orch.execute("build something")
        assert summary.execution_passed is True


# ── 빌드 FAIL → 실행 스킵 ────────────────────────────────────────────────────

class TestBuildFail:
    def test_status_failed_when_build_fails(self):
        llm = MagicMock()
        llm.generate_plan.return_value = _make_plan("src/a.py")
        llm.generate_code.return_value = _make_patch("src/a.py")

        validator = MagicMock()
        validator.build.return_value = BuildResult(
            passed=False, errors=["src/a.c:1: error: syntax error"], tool="gcc"
        )

        orch = _make_orch(llm=llm, execution_validator=validator)
        summary = orch.execute("build something")

        assert summary.status == RunStatus.FAILED
        assert summary.build_passed is False
        # 빌드 실패 시 실행 단계 건너뜀
        validator.run_smoke_test.assert_not_called()
        assert summary.execution_passed is None

    def test_build_skipped_when_task_failed(self):
        """태스크 실패 시 빌드/실행 단계 모두 건너뜀."""
        llm = MagicMock()
        llm.generate_plan.return_value = _make_plan("src/a.py")
        llm.generate_code.return_value = _make_patch("src/a.py")

        sensor = MagicMock()
        # 모든 lint 시도 실패
        sensor.verify_code.return_value = LintResult(
            passed=False, errors=["E999 SyntaxError"], tool="ruff"
        )

        validator = MagicMock()
        # _make_orch 의 기본 sensor 대신 위에서 만든 failing sensor 를 사용
        fs = MagicMock()
        fs.backup_file.return_value = None
        fs.get_workspace_skeleton.return_value = ""
        fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"

        from agent.usecase.orchestrator import HarnessOrchestrator
        orch = HarnessOrchestrator(
            llm=llm, fs=fs, sensor=sensor,
            run_repo=MagicMock(), run_id="test",
            execution_validator=validator,
            max_self_heal=1,
        )
        summary = orch.execute("build something")

        assert summary.failed_tasks > 0
        validator.build.assert_not_called()
        validator.run_smoke_test.assert_not_called()


# ── 실행 FAIL ─────────────────────────────────────────────────────────────────

class TestExecFail:
    def test_status_failed_when_exec_fails(self):
        llm = MagicMock()
        llm.generate_plan.return_value = _make_plan("src/a.py")
        llm.generate_code.return_value = _make_patch("src/a.py")

        validator = MagicMock()
        validator.build.return_value = BuildResult(passed=True, tool="py_compile")
        validator.run_smoke_test.return_value = ExecutionResult(
            passed=False, exit_code=1, error_summary="RuntimeError: crash",
            command="python3 src/main.py"
        )

        orch = _make_orch(llm=llm, execution_validator=validator)
        summary = orch.execute("build something")

        assert summary.status == RunStatus.FAILED
        assert summary.build_passed is True
        assert summary.execution_passed is False


# ── validator 없음 (None) ─────────────────────────────────────────────────────

class TestNoValidator:
    def test_skips_build_exec_when_no_validator(self):
        llm = MagicMock()
        llm.generate_plan.return_value = _make_plan("src/a.py")
        llm.generate_code.return_value = _make_patch("src/a.py")

        orch = _make_orch(llm=llm, execution_validator=None)
        summary = orch.execute("build something")

        assert summary.status == RunStatus.COMPLETED
        assert summary.build_passed is None
        assert summary.execution_passed is None


# ── 개별 단계 비활성화 ────────────────────────────────────────────────────────

class TestPartialValidation:
    def test_build_only(self):
        llm = MagicMock()
        llm.generate_plan.return_value = _make_plan("src/a.py")
        llm.generate_code.return_value = _make_patch("src/a.py")

        validator = MagicMock()
        validator.build.return_value = BuildResult(passed=True, tool="gcc")

        orch = _make_orch(llm=llm, execution_validator=validator,
                          enable_build=True, enable_exec=False)
        summary = orch.execute("build something")

        validator.build.assert_called_once()
        validator.run_smoke_test.assert_not_called()
        assert summary.build_passed is True
        assert summary.execution_passed is None

    def test_exec_only(self):
        """빌드 스킵, 실행만 수행."""
        llm = MagicMock()
        llm.generate_plan.return_value = _make_plan("src/a.py")
        llm.generate_code.return_value = _make_patch("src/a.py")

        validator = MagicMock()
        validator.run_smoke_test.return_value = ExecutionResult(
            passed=True, exit_code=0, command="python3 src/main.py"
        )

        orch = _make_orch(llm=llm, execution_validator=validator,
                          enable_build=False, enable_exec=True)
        summary = orch.execute("build something")

        validator.build.assert_not_called()
        validator.run_smoke_test.assert_called_once()
        assert summary.build_passed is None
        assert summary.execution_passed is True
