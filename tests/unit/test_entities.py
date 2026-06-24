"""Unit tests for domain entities."""
import pytest
from pydantic import ValidationError

from agent.domain.entities import (
    CodePatch,
    ProjectPlan,
    RunStatus,
    RunSummary,
    Task,
    TaskAction,
    TaskStatus,
)
from datetime import datetime, timezone


# ── Task ──────────────────────────────────────────────────────────────────────

class TestTask:
    def _make(self, **kw) -> Task:
        defaults = dict(
            task_id=1,
            file_path="src/foo.py",
            action=TaskAction.CREATE,
            description="Create foo module",
        )
        return Task(**(defaults | kw))

    def test_defaults(self):
        t = self._make()
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.error_history == []

    def test_mark_done(self):
        t = self._make()
        t.mark_done()
        assert t.status == TaskStatus.DONE
        assert t.completed_at is not None

    def test_mark_failed(self):
        t = self._make()
        t.mark_failed("lint error")
        assert t.status == TaskStatus.FAILED
        assert t.last_error == "lint error"

    def test_record_retry_increments(self):
        t = self._make()
        t.record_retry_error("err1")
        t.record_retry_error("err2")
        assert t.retry_count == 2
        assert t.last_error == "err2"

    def test_path_traversal_rejected(self):
        with pytest.raises(ValidationError):
            self._make(file_path="../etc/passwd")

    def test_absolute_path_rejected(self):
        with pytest.raises(ValidationError):
            self._make(file_path="/etc/passwd")


# ── CodePatch ─────────────────────────────────────────────────────────────────

class TestCodePatch:
    def test_empty_code_block_rejected(self):
        with pytest.raises(ValidationError):
            CodePatch(target_file="a.py", explanation="x", code_block="   ")

    def test_path_traversal_rejected(self):
        with pytest.raises(ValidationError):
            CodePatch(target_file="../hack.py", explanation="x", code_block="pass")


# ── ProjectPlan ───────────────────────────────────────────────────────────────

class TestProjectPlan:
    def _task(self, tid: int, path: str) -> Task:
        return Task(
            task_id=tid,
            file_path=path,
            action=TaskAction.CREATE,
            description="desc",
        )

    def test_duplicate_creates_rejected(self):
        tasks = [self._task(1, "a.py"), self._task(2, "a.py")]
        with pytest.raises(ValidationError):
            ProjectPlan(tasks=tasks)

    def test_valid_plan(self):
        tasks = [self._task(1, "a.py"), self._task(2, "b.py")]
        plan = ProjectPlan(tasks=tasks)
        assert len(plan.tasks) == 2


# ── RunSummary ────────────────────────────────────────────────────────────────

class TestRunSummary:
    def test_finish_computes_counts(self):
        tasks = [
            Task(task_id=1, file_path="a.py", action=TaskAction.CREATE, description="d",
                 status=TaskStatus.DONE),
            Task(task_id=2, file_path="b.py", action=TaskAction.CREATE, description="d",
                 status=TaskStatus.FAILED),
            Task(task_id=3, file_path="c.py", action=TaskAction.CREATE, description="d",
                 status=TaskStatus.SKIPPED),
        ]
        s = RunSummary(
            run_id="test123",
            started_at=datetime.now(timezone.utc),
            tasks=tasks,
        )
        s.finish(RunStatus.COMPLETED)
        assert s.completed_tasks == 1
        assert s.failed_tasks == 1
        assert s.skipped_tasks == 1
        assert s.duration_seconds is not None
