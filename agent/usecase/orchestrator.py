"""
HarnessOrchestrator – the core use-case that drives the agent loop.

Pipeline (per task)
-------------------
  LLM generate → post-process → write → lint
  └── self-heal loop (max_self_heal retries)

Pipeline (post all tasks)
--------------------------
  build  → smoke-test execution → PASS / FAIL
  └── 실패 시 RunSummary 에 기록 (태스크 레벨 재시도 아님)
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Optional

import structlog

from agent.domain.entities import (
    BuildResult,
    ExecutionResult,
    RunStatus,
    RunSummary,
    Task,
    TaskStatus,
)
from agent.domain.exceptions import (
    LLMError,
    PlanValidationError,
    TaskLimitExceededError,
)
from agent.domain.interfaces import (
    IExecutionValidator,
    IFileSystemAdapter,
    ILLMAdapter,
    INotificationAdapter,
    IRunRepository,
    ISensorAdapter,
)
from agent.domain.language_profile import ILanguageProfile
from agent.infrastructure.llm.post_processor import (
    CodePostProcessor,
    TruncatedCodeError,
)

log = structlog.get_logger(__name__)


class HarnessOrchestrator:
    """
    Drives the full agent pipeline.

    Parameters
    ----------
    llm / fs / sensor / run_repo / run_id:
        Core adapters.
    notifier:
        Optional Slack/webhook notification adapter.
    language_profile:
        Active language profile (controls skeleton extensions).
    post_processor:
        Code post-processor (trailing whitespace, unused imports, truncation).
    execution_validator:
        Build + smoke-test validator. None = skip validation.
    dry_run:
        When True, no files are written and all external tools are simulated.
    max_self_heal:
        Max retry iterations per task before marking it failed.
    max_tasks:
        Safety cap on plan size.
    enable_build_validation / enable_execution_validation:
        Fine-grained control over post-task pipeline steps.
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
        post_processor: Optional[CodePostProcessor] = None,
        execution_validator: Optional[IExecutionValidator] = None,
        dry_run: bool = False,
        max_self_heal: int = 3,
        max_tasks: int = 50,
        enable_build_validation: bool = True,
        enable_execution_validation: bool = True,
    ) -> None:
        self._llm = llm
        self._fs = fs
        self._sensor = sensor
        self._repo = run_repo
        self._notifier = notifier
        self._language_profile = language_profile
        self._post_processor = post_processor or CodePostProcessor()
        self._execution_validator = execution_validator
        self._dry_run = dry_run
        self._max_self_heal = max_self_heal
        self._max_tasks = max_tasks
        self._run_id = run_id
        self._enable_build = enable_build_validation
        self._enable_exec = enable_execution_validation

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute(self, user_requirements: str) -> RunSummary:
        """Run the full pipeline. Always returns a RunSummary."""
        summary = RunSummary(
            run_id=self._run_id,
            started_at=datetime.now(timezone.utc),
        )
        log.info("agent.run.start", dry_run=self._dry_run,
                 max_self_heal=self._max_self_heal)

        try:
            plan = self._build_plan(user_requirements, summary)
            self._execute_plan(plan.tasks, summary)

            # ── 전체 빌드 및 실행 검증 ─────────────────────────────────────────
            if self._execution_validator is not None:
                self._run_build_and_exec_validation(summary)

        except (PlanValidationError, TaskLimitExceededError) as exc:
            log.error("agent.run.plan_error", error=str(exc))
            summary.finish(RunStatus.ABORTED)
            self._repo.save(summary)
            raise
        except Exception as exc:  # noqa: BLE001
            log.critical("agent.run.unexpected_error",
                         error=str(exc), traceback=traceback.format_exc())
            summary.finish(RunStatus.ABORTED)
            self._repo.save(summary)
            raise

        task_failed = sum(1 for t in summary.tasks if t.status == TaskStatus.FAILED)
        build_ok = summary.build_passed is not False      # None(skipped) or True
        exec_ok = summary.execution_passed is not False   # None(skipped) or True

        if task_failed == 0 and build_ok and exec_ok:
            final_status = RunStatus.COMPLETED
        else:
            final_status = RunStatus.FAILED

        summary.finish(final_status)
        self._repo.save(summary)
        self._send_completion_notification(summary)

        log.info(
            "agent.run.finish",
            status=final_status,
            completed=summary.completed_tasks,
            failed=summary.failed_tasks,
            build_passed=summary.build_passed,
            execution_passed=summary.execution_passed,
            duration_s=summary.duration_seconds,
        )
        return summary

    # ── Private: plan ──────────────────────────────────────────────────────────

    def _build_plan(self, requirements: str, summary: RunSummary):
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

    # ── Private: task loop ─────────────────────────────────────────────────────

    def _execute_plan(self, tasks: list[Task], summary: RunSummary) -> None:
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
                task_log.error("agent.task.failed",
                               retries=task.retry_count, last_error=task.last_error)
                if self._notifier:
                    self._notifier.notify_task_failed(task, task.last_error or "")

    def _run_task_with_healing(
        self, task: Task, completed_files: list[str], task_log
    ) -> bool:
        """
        Generate → post-process → write → lint  (retry loop).

        Returns True on success, False when all retries exhausted.
        """
        backup_path = self._fs.backup_file(task.file_path)
        if backup_path:
            task_log.debug("agent.task.backup_created", backup=backup_path)

        for attempt in range(1, self._max_self_heal + 1):
            skeleton = self._fs.get_workspace_skeleton(
                accepted_extensions=(
                    self._language_profile.skeleton_extensions
                    if self._language_profile else None
                ),
                header_extensions=(
                    self._language_profile.header_extensions
                    if self._language_profile
                    and hasattr(self._language_profile, "header_extensions")
                    else None
                ),
            )

            # ── Generate ───────────────────────────────────────────────────────
            try:
                patch = self._llm.generate_code(
                    task=task,
                    memory_summary=self._build_memory_summary(completed_files),
                    workspace_skeleton=skeleton,
                    error_feedback=self._build_error_feedback(task),
                )
            except LLMError as exc:
                task.record_retry_error(f"LLM generation error: {exc}")
                task_log.warning("agent.task.llm_error", attempt=attempt, error=str(exc))
                continue

            # ── Post-process ───────────────────────────────────────────────────
            try:
                patch = self._post_processor.process(patch)
            except TruncatedCodeError as exc:
                task.record_retry_error(
                    f"[코드 잘림 감지] {exc}\n"
                    "전체 파일을 처음부터 끝까지 완성해 주세요. "
                    "파일의 마지막 줄이 완전한 문장/구문이어야 합니다."
                )
                task_log.warning("agent.task.truncation_detected",
                                 attempt=attempt, reason=str(exc)[:200])
                continue

            # ── Write ──────────────────────────────────────────────────────────
            self._fs.write_file(patch.target_file, patch.code_block, self._dry_run)

            # ── Lint ───────────────────────────────────────────────────────────
            abs_path = self._fs.get_absolute_path(patch.target_file)
            lint_result = self._sensor.verify_code(abs_path, self._dry_run)

            if lint_result.passed:
                task_log.info("agent.task.lint_pass", attempt=attempt,
                              tool=lint_result.tool)
                return True

            task.record_retry_error(lint_result.error_summary)
            task_log.warning("agent.task.lint_fail",
                             attempt=attempt, errors=lint_result.errors[:5])

        return False

    # ── Private: build & execution validation ─────────────────────────────────

    def _run_build_and_exec_validation(self, summary: RunSummary) -> None:
        """
        모든 태스크가 완료된 후 전체 프로젝트를 빌드하고 실행한다.

        실패해도 예외를 던지지 않는다 – RunSummary 에 결과를 기록하고
        최종 상태 판정은 execute() 에서 수행한다.
        """
        # 태스크 실패가 있으면 빌드 시도 의미 없음
        task_failed = sum(1 for t in summary.tasks if t.status == TaskStatus.FAILED)
        if task_failed > 0:
            log.info("agent.validation.skipped",
                     reason=f"{task_failed} task(s) failed – skip build/exec")
            return

        assert self._execution_validator is not None
        workspace = self._fs.get_absolute_path(".")

        # ── Build ──────────────────────────────────────────────────────────────
        if self._enable_build:
            log.info("agent.build.start")
            build_result: BuildResult = self._execution_validator.build(
                workspace, self._dry_run
            )
            summary.build_passed = build_result.passed

            if build_result.passed:
                log.info("agent.build.pass", tool=build_result.tool)
            else:
                log.error(
                    "agent.build.fail",
                    tool=build_result.tool,
                    errors=build_result.errors[:5],
                )
                # 빌드 실패 → 실행 단계 건너뜀
                return
        else:
            summary.build_passed = None  # skipped

        # ── Smoke-test execution ───────────────────────────────────────────────
        if self._enable_exec:
            log.info("agent.execution.start")
            exec_result: ExecutionResult = self._execution_validator.run_smoke_test(
                workspace, self._dry_run
            )
            summary.execution_passed = exec_result.passed

            if exec_result.passed:
                log.info(
                    "agent.execution.pass",
                    command=exec_result.command,
                    exit_code=exec_result.exit_code,
                    stdout_preview=exec_result.stdout[:200],
                )
            else:
                log.error(
                    "agent.execution.fail",
                    command=exec_result.command,
                    exit_code=exec_result.exit_code,
                    error=exec_result.error_summary[:300],
                )
        else:
            summary.execution_passed = None  # skipped

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_error_feedback(self, task: Task) -> str:
        if not task.error_history:
            return ""
        last_error = task.error_history[-1]

        # C/C++ 파일인 경우 헤더 파일 내용을 피드백에 포함
        header_note = ""
        if self._language_profile and hasattr(
            self._language_profile, "header_extensions"
        ):
            header_exts = self._language_profile.header_extensions
            if any(
                task.file_path.endswith((".c", ".cpp", ".cc", ".cxx"))
                for _ in [None]
            ):
                header_contents = self._collect_header_contents(header_exts)
                if header_contents:
                    header_note = (
                        "\n=== 헤더 파일 선언 (반드시 이 시그니처를 사용하세요) ===\n"
                        + header_contents
                        + "\n=== 헤더 파일 끝 ===\n"
                    )

        return (
            f"=== 이전 시도 #{task.retry_count} 실패 – 아래 문제를 반드시 수정하세요 ===\n"
            f"{last_error}\n"
            f"{header_note}"
            "=== 수정 지침 ===\n"
            "1. 위 에러가 발생한 줄을 찾아 정확히 수정하세요.\n"
            "2. 헤더 파일에 선언된 함수 시그니처와 정확히 일치해야 합니다.\n"
            "3. 코드 전체를 처음부터 끝까지 완성된 형태로 반환하세요.\n"
            "4. 줄 끝에 공백(trailing whitespace)이 없어야 합니다.\n"
            "5. 코드가 중간에 잘리지 않도록 전체 파일을 완성하세요.\n"
        )

    def _collect_header_contents(self, header_exts: frozenset[str]) -> str:
        """워크스페이스의 헤더 파일 전체 내용을 수집한다."""
        result = []
        try:
            ws = self._fs.get_absolute_path(".")
            import os
            from pathlib import Path
            for p in sorted(Path(ws).rglob("*")):
                if (p.suffix.lower() in header_exts
                        and ".harness_backups" not in str(p)
                        and ".build" not in str(p)):
                    rel = os.path.relpath(str(p), ws)
                    try:
                        text = p.read_text(encoding="utf-8")
                        result.append(f"--- [{rel}] ---\n{text}")
                    except OSError:
                        pass
        except Exception:  # noqa: BLE001
            pass
        return "\n".join(result)

    @staticmethod
    def _build_memory_summary(completed_files: list[str]) -> str:
        if not completed_files:
            return ""
        return (
            "이미 완료된 파일 (import 시 참고):\n"
            + "\n".join(f"  - {p}" for p in completed_files)
        )

    def _send_completion_notification(self, summary: RunSummary) -> None:
        if self._notifier is None:
            return
        try:
            self._notifier.notify_run_complete(summary)
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.notification.error", error=str(exc))
