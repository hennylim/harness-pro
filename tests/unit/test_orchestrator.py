"""Unit tests for HarnessOrchestrator."""
import pytest
from unittest.mock import MagicMock, call

from agent.domain.entities import (
    CodePatch,
    LintResult,
    ProjectPlan,
    RunStatus,
    Task,
    TaskAction,
    TaskStatus,
)
from agent.domain.exceptions import PlanValidationError, TaskLimitExceededError
from agent.usecase.orchestrator import HarnessOrchestrator
from config.logging import configure_logging

configure_logging(level="ERROR", fmt="console")


def _make_task(tid: int = 1, path: str = "src/a.py") -> Task:
    return Task(
        task_id=tid,
        file_path=path,
        action=TaskAction.CREATE,
        description="Create module",
    )


def _make_plan(*paths: str) -> ProjectPlan:
    return ProjectPlan(
        tasks=[_make_task(i + 1, p) for i, p in enumerate(paths)]
    )


def _patch(path: str = "src/a.py") -> CodePatch:
    return CodePatch(
        target_file=path,
        explanation="ok",
        code_block="print('hello')\n",
    )


def _make_orch(
    llm=None, fs=None, sensor=None, run_repo=None,
    max_self_heal=3, dry_run=False, max_tasks=50,
) -> HarnessOrchestrator:
    llm = llm or MagicMock()
    fs = fs or MagicMock()
    sensor = sensor or MagicMock()
    run_repo = run_repo or MagicMock()
    fs.backup_file.return_value = None
    fs.get_workspace_skeleton.return_value = ""
    fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"
    return HarnessOrchestrator(
        llm=llm, fs=fs, sensor=sensor, run_repo=run_repo,
        run_id="test-run", dry_run=dry_run,
        max_self_heal=max_self_heal, max_tasks=max_tasks,
    )


class TestHappyPath:
    def test_single_task_success(self):
        llm = MagicMock()
        fs = MagicMock()
        sensor = MagicMock()
        run_repo = MagicMock()

        llm.generate_plan.return_value = _make_plan("src/a.py")
        llm.generate_code.return_value = _patch("src/a.py")
        sensor.verify_code.return_value = LintResult(passed=True, tool="ruff")
        fs.backup_file.return_value = None
        fs.get_workspace_skeleton.return_value = ""
        fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"

        orch = HarnessOrchestrator(
            llm=llm, fs=fs, sensor=sensor, run_repo=run_repo,
            run_id="t1", dry_run=False,
        )
        summary = orch.execute("build something")

        assert summary.status == RunStatus.COMPLETED
        assert summary.completed_tasks == 1
        assert summary.failed_tasks == 0
        fs.write_file.assert_called_once()

    def test_multiple_tasks_all_pass(self):
        llm = MagicMock()
        fs = MagicMock()
        sensor = MagicMock()
        run_repo = MagicMock()

        llm.generate_plan.return_value = _make_plan("a.py", "b.py", "c.py")
        llm.generate_code.side_effect = [_patch("a.py"), _patch("b.py"), _patch("c.py")]
        sensor.verify_code.return_value = LintResult(passed=True)
        fs.backup_file.return_value = None
        fs.get_workspace_skeleton.return_value = ""
        fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"

        orch = HarnessOrchestrator(
            llm=llm, fs=fs, sensor=sensor, run_repo=run_repo, run_id="t2"
        )
        summary = orch.execute("build something")
        assert summary.completed_tasks == 3


class TestSelfHeal:
    def test_heals_after_one_lint_failure(self):
        llm = MagicMock()
        fs = MagicMock()
        sensor = MagicMock()
        run_repo = MagicMock()

        llm.generate_plan.return_value = _make_plan("src/a.py")
        llm.generate_code.return_value = _patch("src/a.py")
        # First attempt fails, second passes
        sensor.verify_code.side_effect = [
            LintResult(passed=False, errors=["E101 bad indent"], tool="ruff"),
            LintResult(passed=True, tool="ruff"),
        ]
        fs.backup_file.return_value = None
        fs.get_workspace_skeleton.return_value = ""
        fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"

        orch = HarnessOrchestrator(
            llm=llm, fs=fs, sensor=sensor, run_repo=run_repo,
            run_id="t3", max_self_heal=3,
        )
        summary = orch.execute("build something")
        assert summary.completed_tasks == 1
        assert llm.generate_code.call_count == 2

    def test_fails_when_all_retries_exhausted(self):
        llm = MagicMock()
        fs = MagicMock()
        sensor = MagicMock()
        run_repo = MagicMock()

        llm.generate_plan.return_value = _make_plan("src/a.py")
        llm.generate_code.return_value = _patch("src/a.py")
        sensor.verify_code.return_value = LintResult(
            passed=False, errors=["E999"], tool="ruff"
        )
        fs.backup_file.return_value = None
        fs.get_workspace_skeleton.return_value = ""
        fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"

        orch = HarnessOrchestrator(
            llm=llm, fs=fs, sensor=sensor, run_repo=run_repo,
            run_id="t4", max_self_heal=2,
        )
        summary = orch.execute("build something")
        assert summary.failed_tasks == 1
        assert summary.status == RunStatus.FAILED


class TestValidation:
    def test_empty_plan_raises(self):
        llm = MagicMock()
        llm.generate_plan.return_value = ProjectPlan(tasks=[])
        orch = _make_orch(llm=llm)
        with pytest.raises(PlanValidationError):
            orch.execute("requirements")

    def test_task_limit_exceeded_raises(self):
        llm = MagicMock()
        llm.generate_plan.return_value = _make_plan(*[f"f{i}.py" for i in range(10)])
        orch = _make_orch(llm=llm, max_tasks=5)
        with pytest.raises(TaskLimitExceededError):
            orch.execute("requirements")

    def test_run_repo_always_called(self):
        llm = MagicMock()
        run_repo = MagicMock()
        llm.generate_plan.return_value = ProjectPlan(tasks=[])
        orch = _make_orch(llm=llm, run_repo=run_repo)
        with pytest.raises(PlanValidationError):
            orch.execute("r")
        run_repo.save.assert_called_once()
