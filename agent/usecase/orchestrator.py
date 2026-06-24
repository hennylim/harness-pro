"""
HarnessOrchestrator – the core use-case that drives the agent loop.

Responsibilities
----------------
* Accept a free-text requirements string.
* Ask the LLM to produce a structured plan.
* Execute each task with a self-healing retry loop.
* Coordinate file-system writes, lint checks, backups, and notifications.
* Persist a RunSummary on every exit path (success, failure, exception).
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Optional

import structlog

from agent.domain.entities import (
    RunStatus,
    RunSummary,
    Task,
    TaskStatus,
)
from agent.domain.exceptions import (
    LLMError,
    MaxRetriesExceededError,
    PlanValidationError,
    TaskLimitExceededError,
)
from agent.domain.language_profile import ILanguageProfile
from agent.domain.interfaces import (
    IFileSystemAdapter,
    ILLMAdapter,
    INotificationAdapter,
    IRunRepository,
    ISensorAdapter,
)

log = structlog.get_logger(__name__)


class HarnessOrchestrator:
    """
    Drives the full agent pipeline.

    Parameters
    ----------
    llm:
        Adapter for LLM plan/code generation.
    fs:
        Adapter for workspace file-system operations.
    sensor:
        Adapter for static analysis / linting.
    run_repo:
        Repository for persisting run summaries.
    notifier:
        Optional notification adapter (Slack, email, …).
    dry_run:
        When True, no files are written and lint is simulated.
    max_self_heal:
        Maximum retry iterations before a task is marked failed.
    max_tasks:
        Safety limit on plan size.
    """

    def __init__(
        self,
        llm: ILLMAdapter,
        fs: IFileSystemAdapter,
        sensor: ISensorAdapter,
        run_repo: IRunRepository,
        run_id: str,
        notifier: Optional[INotificationAdapter] = None,
        language_profile: Optional[ILanguageProfile] = None,
        dry_run: bool = False,
        max_self_heal: int = 3,
        max_tasks: int = 50,
    ) -> None:
        self._llm = llm
        self._fs = fs
        self._sensor = sensor
        self._repo = run_repo
        self._notifier = notifier
        self._language_profile = language_profile
        self._dry_run = dry_run
        self._max_self_heal = max_self_heal
        self._max_tasks = max_tasks
        self._run_id = run_id

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute(self, user_requirements: str) -> RunSummary:
        """
        Execute the full agent pipeline for the given requirements.

        Returns a :class:`RunSummary` on all exit paths.
        """
        summary = RunSummary(
            run_id=self._run_id,
            started_at=datetime.now(timezone.utc),
        )

        log.info(
            "agent.run.start",
            dry_run=self._dry_run,
            max_self_heal=self._max_self_heal,
        )

        try:
            plan = self._build_plan(user_requirements, summary)
            self._execute_plan(plan.tasks, summary)
            pass
        except (PlanValidationError, TaskLimitExceededError) as exc:
            log.error("agent.run.plan_error", error=str(exc))
            summary.finish(RunStatus.ABORTED)
            self._repo.save(summary)
            raise
        except Exception as exc:  # noqa: BLE001
            log.critical(
                "agent.run.unexpected_error",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            summary.finish(RunStatus.ABORTED)
            self._repo.save(summary)
            raise

        # Recount from tasks list before calling finish()
        failed = sum(1 for t in summary.tasks if t.status == TaskStatus.FAILED)
        final_status = RunStatus.COMPLETED if failed == 0 else RunStatus.FAILED
        summary.finish(final_status)
        self._repo.save(summary)
        self._send_completion_notification(summary)

        log.info(
            "agent.run.finish",
            status=final_status,
            completed=summary.completed_tasks,
            failed=summary.failed_tasks,
            duration_s=summary.duration_seconds,
        )
        return summary

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_plan(self, requirements: str, summary: RunSummary):
        """Call the LLM planner and validate the result."""
        log.info("agent.plan.generating")
        try:
            plan = self._llm.generate_plan(requirements)
        except LLMError as exc:
            raise PlanValidationError(f"LLM failed to generate plan: {exc}") from exc

        if not plan.tasks:
            raise PlanValidationError("LLM returned an empty task plan.")

        if len(plan.tasks) > self._max_tasks:
            raise TaskLimitExceededError(
                f"Plan has {len(plan.tasks)} tasks; limit is {self._max_tasks}."
            )

        summary.total_tasks = len(plan.tasks)
        summary.tasks = list(plan.tasks)
        log.info("agent.plan.ready", task_count=len(plan.tasks))
        return plan

    def _execute_plan(self, tasks: list[Task], summary: RunSummary) -> None:
        """Iterate through the task queue, self-healing on lint failures."""
        memory: list[str] = []

        for task in tasks:
            task.status = TaskStatus.IN_PROGRESS
            task_log = log.bind(task_id=task.task_id, file=task.file_path)
            task_log.info("agent.task.start", action=task.action)

            success = self._run_task_with_healing(task, memory, task_log)

            if success:
                task.mark_done()
                memory.append(task.file_path)
                task_log.info("agent.task.done", retries=task.retry_count)
            else:
                task.mark_failed(task.last_error or "max retries exceeded")
                task_log.error(
                    "agent.task.failed",
                    retries=task.retry_count,
                    last_error=task.last_error,
                )
                if self._notifier:
                    self._notifier.notify_task_failed(task, task.last_error or "")

    def _run_task_with_healing(
        self,
        task: Task,
        completed_files: list[str],
        task_log,
    ) -> bool:
        """
        Attempt to generate, write, and validate code for *task*.
        Retries up to ``max_self_heal`` times on lint failure.
        Returns True on success, False when all retries are exhausted.
        """
        # Back up any existing file before the first write
        backup_path = self._fs.backup_file(task.file_path)
        if backup_path:
            task_log.debug("agent.task.backup_created", backup=backup_path)

        for attempt in range(1, self._max_self_heal + 1):
            skeleton = self._fs.get_workspace_skeleton(
                accepted_extensions=(
                    self._language_profile.skeleton_extensions
                    if self._language_profile else None
                )
            )
            error_feedback = (
                f"이전 오류 ({task.retry_count}회 시도):\n{task.last_error}"
                if task.last_error
                else ""
            )
            memory_summary = (
                "완료된 파일:\n" + "\n".join(f"  - {p}" for p in completed_files)
                if completed_files
                else ""
            )

            # ── Code generation ────────────────────────────────────────────────
            try:
                patch = self._llm.generate_code(
                    task=task,
                    memory_summary=memory_summary,
                    workspace_skeleton=skeleton,
                    error_feedback=error_feedback,
                )
            except LLMError as exc:
                task.record_retry_error(f"LLM generation error: {exc}")
                task_log.warning(
                    "agent.task.llm_error",
                    attempt=attempt,
                    error=str(exc),
                )
                continue

            # ── File write ─────────────────────────────────────────────────────
            self._fs.write_file(patch.target_file, patch.code_block, self._dry_run)

            # ── Lint gate ──────────────────────────────────────────────────────
            abs_path = self._fs.get_absolute_path(patch.target_file)
            lint_result = self._sensor.verify_code(abs_path, self._dry_run)

            if lint_result.passed:
                task_log.info(
                    "agent.task.lint_pass",
                    attempt=attempt,
                    tool=lint_result.tool,
                )
                return True

            # ── Self-heal loop ─────────────────────────────────────────────────
            task.record_retry_error(lint_result.error_summary)
            task_log.warning(
                "agent.task.lint_fail",
                attempt=attempt,
                errors=lint_result.errors[:3],   # log first 3 to avoid noise
            )

        return False

    def _send_completion_notification(self, summary: RunSummary) -> None:
        if self._notifier is None:
            return
        try:
            self._notifier.notify_run_complete(summary)
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.notification.error", error=str(exc))
