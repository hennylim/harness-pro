"""
HarnessOrchestrator – 적응형 코드 생성 파이프라인.

파이프라인 (전체 흐름)
----------------------
1. 강화 계획 생성 (AdaptivePlanner)
   - 파일 복잡도 분석 → 일반 Task / ChunkedTask 분류
   - ChunkedTask: 큰 파일을 섹션별로 분할

2. 태스크 실행 루프
   a. 일반 Task → 기존 생성 → post-process → lint → self-heal
   b. ChunkedTask → 섹션별 순차 생성 → 조립 → post-process → lint → self-heal

3. 전체 빌드 + 실행 검증
   - 모든 태스크 완료 후 빌드 및 smoke-test 실행
   - 실패 시 에러를 분석해 관련 태스크를 자동 재시도

섹션 생성 맥락 유지
-------------------
- 이전 섹션 코드를 다음 섹션 프롬프트에 포함
- 헤더 파일 전체 내용을 에러 피드백에 포함
- 워크스페이스 스켈레톤으로 파일 간 의존성 파악
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import os

import structlog

from agent.domain.entities import (
    BuildResult,
    ChunkedTask,
    ChunkStatus,
    EnhancedProjectPlan,
    ExecutionResult,
    FileSection,
    ProjectPlan,
    RunStatus,
    RunSummary,
    Task,
    TaskAction,
    TaskStatus,
)
from agent.domain.exceptions import (
    LLMError,
    LLMParseError,
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
from agent.usecase.adaptive_planner import AdaptivePlanner

log = structlog.get_logger(__name__)

# 청크 분할 임계값 기본값 (이 줄 수 이상 예상 파일은 섹션 분할)
DEFAULT_CHUNK_THRESHOLD = 80
# 섹션별 최대 self-heal 재시도 횟수
DEFAULT_SECTION_RETRIES = 3
# 빌드 실패 후 자동 재시도 최대 횟수
DEFAULT_BUILD_FIX_RETRIES = 2


class HarnessOrchestrator:
    """
    적응형 코드 생성 파이프라인 오케스트레이터.

    Parameters
    ----------
    llm / fs / sensor / run_repo / run_id:
        핵심 어댑터.
    notifier:
        선택적 알림 어댑터.
    language_profile:
        활성 언어 프로필.
    post_processor:
        코드 후처리기.
    execution_validator:
        빌드 + smoke-test 검증기.
    dry_run:
        True 이면 파일 저장 및 외부 도구 실행 없음.
    max_self_heal:
        태스크당 최대 재시도 횟수.
    max_tasks:
        플랜 크기 안전 한도.
    chunk_threshold_lines:
        이 줄 수 이상 예상 파일은 섹션 분할.
    enable_build_validation / enable_execution_validation:
        검증 단계 활성화 여부.
    build_fix_retries:
        빌드 실패 시 자동 수정 재시도 횟수.
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
        chunk_threshold_lines: int = DEFAULT_CHUNK_THRESHOLD,
        enable_build_validation: bool = True,
        enable_execution_validation: bool = True,
        build_fix_retries: int = DEFAULT_BUILD_FIX_RETRIES,
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
        self._chunk_threshold = chunk_threshold_lines
        self._run_id = run_id
        self._enable_build = enable_build_validation
        self._enable_exec = enable_execution_validation
        self._build_fix_retries = build_fix_retries

        # 적응형 플래너 (청크 분할 전담)
        self._planner = AdaptivePlanner(
            llm_adapter=llm,
            chunk_threshold_lines=chunk_threshold_lines,
            section_max_lines=60,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute(self, user_requirements: str) -> RunSummary:
        """전체 파이프라인 실행. 항상 RunSummary 를 반환한다."""
        summary = RunSummary(
            run_id=self._run_id,
            started_at=datetime.now(timezone.utc),
        )
        log.info(
            "agent.run.start",
            dry_run=self._dry_run,
            max_self_heal=self._max_self_heal,
            chunk_threshold=self._chunk_threshold,
        )

        try:
            # ── 1. 강화 계획 생성 ──────────────────────────────────────────────
            plan = self._build_enhanced_plan(user_requirements, summary)

            # ── 2. 태스크 실행 ─────────────────────────────────────────────────
            completed_files: list[str] = []
            self._execute_normal_tasks(plan.tasks, summary, completed_files)
            self._execute_chunked_tasks(plan.chunked_tasks, summary, completed_files)

            # ── 3. 빌드 + 실행 검증 ────────────────────────────────────────────
            if self._execution_validator is not None:
                self._run_build_and_exec_validation(summary, plan)

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

        # 최종 상태 결정
        task_failed = sum(
            1 for t in summary.tasks if t.status == TaskStatus.FAILED
        )
        build_ok = summary.build_passed is not False
        exec_ok = summary.execution_passed is not False
        final_status = (
            RunStatus.COMPLETED if task_failed == 0 and build_ok and exec_ok
            else RunStatus.FAILED
        )

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

    # ── Plan ───────────────────────────────────────────────────────────────────

    def _build_enhanced_plan(
        self, requirements: str, summary: RunSummary
    ) -> EnhancedProjectPlan:
        """강화 계획 생성 (청크 분할 포함)."""
        log.info("agent.plan.generating", mode="enhanced")
        try:
            plan = self._planner.generate_enhanced_plan(
                requirements=requirements,
                chunk_threshold_lines=self._chunk_threshold,
            )
        except Exception as exc:
            # 강화 계획 실패 시 일반 계획으로 폴백
            log.warning(
                "agent.plan.enhanced_failed_fallback",
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            log.debug(
                "agent.plan.enhanced_failed_traceback",
                traceback=traceback.format_exc(),
            )
            plan = self._fallback_to_normal_plan(requirements)

        total = plan.all_tasks_count
        if total == 0:
            raise PlanValidationError("계획에 태스크가 없습니다.")
        if total > self._max_tasks:
            raise TaskLimitExceededError(
                f"계획에 {total}개 태스크 (한도: {self._max_tasks})"
            )

        # RunSummary 용 태스크 목록 구성
        summary.total_tasks = total
        summary.tasks = list(plan.tasks)
        # ChunkedTask 도 Task 형식으로 요약에 추가
        for ct in plan.chunked_tasks:
            proxy = Task(
                task_id=ct.task_id,
                file_path=ct.file_path,
                action=ct.action,
                description=ct.description,
            )
            summary.tasks.append(proxy)

        log.info(
            "agent.plan.ready",
            normal_tasks=len(plan.tasks),
            chunked_tasks=len(plan.chunked_tasks),
            total=total,
        )
        return plan

    def _fallback_to_normal_plan(self, requirements: str) -> EnhancedProjectPlan:
        """강화 계획 실패 시 일반 LLM 플래너로 폴백."""
        log.info("agent.plan.fallback_normal")
        try:
            normal_plan = self._llm.generate_plan(requirements)
        except LLMError as exc:
            raise PlanValidationError(
                f"LLM 계획 생성 실패: {exc}"
            ) from exc
        return EnhancedProjectPlan(tasks=normal_plan.tasks, chunked_tasks=[])

    # ── Normal task execution ──────────────────────────────────────────────────

    def _execute_normal_tasks(
        self,
        tasks: list[Task],
        summary: RunSummary,
        completed_files: list[str],
    ) -> None:
        for task in tasks:
            task.status = TaskStatus.IN_PROGRESS
            task_log = log.bind(task_id=task.task_id, file=task.file_path)
            task_log.info("agent.task.start", action=task.action, mode="normal")

            success = self._run_task_with_healing(task, completed_files, task_log)

            if success:
                task.mark_done()
                completed_files.append(task.file_path)
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
        """Generate → post-process → write → lint (retry loop)."""
        self._fs.backup_file(task.file_path)

        for attempt in range(1, self._max_self_heal + 1):
            skeleton = self._get_skeleton()
            try:
                patch = self._llm.generate_code(
                    task=task,
                    memory_summary=self._build_memory_summary(completed_files),
                    workspace_skeleton=skeleton,
                    error_feedback=self._build_error_feedback(task),
                )
            except LLMError as exc:
                task.record_retry_error(f"LLM 생성 오류: {exc}")
                task_log.warning("agent.task.llm_error", attempt=attempt, error=str(exc))
                continue

            try:
                patch = self._post_processor.process(patch)
            except TruncatedCodeError as exc:
                task.record_retry_error(
                    f"[코드 잘림] {exc}\n전체 파일을 완성하세요."
                )
                task_log.warning("agent.task.truncated", attempt=attempt)
                continue

            self._fs.write_file(patch.target_file, patch.code_block, self._dry_run)
            abs_path = self._fs.get_absolute_path(patch.target_file)
            lint = self._sensor.verify_code(abs_path, self._dry_run)

            if lint.passed:
                task_log.info("agent.task.lint_pass", attempt=attempt, tool=lint.tool)
                return True

            task.record_retry_error(lint.error_summary)
            task_log.warning(
                "agent.task.lint_fail", attempt=attempt, errors=lint.errors[:5]
            )

        return False

    # ── Chunked task execution ─────────────────────────────────────────────────

    def _execute_chunked_tasks(
        self,
        chunked_tasks: list[ChunkedTask],
        summary: RunSummary,
        completed_files: list[str],
    ) -> None:
        for ct in chunked_tasks:
            ct.status = TaskStatus.IN_PROGRESS
            ct_log = log.bind(task_id=ct.task_id, file=ct.file_path)
            ct_log.info(
                "agent.task.start",
                action=ct.action,
                mode="chunked",
                sections=len(ct.sections),
            )

            success = self._run_chunked_task(ct, completed_files, ct_log)

            if success:
                ct.mark_done()
                completed_files.append(ct.file_path)
                ct_log.info("agent.task.done", sections=len(ct.sections))
                # RunSummary 의 proxy task 도 업데이트
                self._update_summary_task(summary, ct.file_path, TaskStatus.DONE)
            else:
                ct.mark_failed(ct.last_error or "섹션 생성 실패")
                ct_log.error(
                    "agent.task.failed",
                    last_error=ct.last_error,
                )
                self._update_summary_task(summary, ct.file_path, TaskStatus.FAILED)
                if self._notifier:
                    proxy = Task(
                        task_id=ct.task_id,
                        file_path=ct.file_path,
                        action=ct.action,
                        description=ct.description,
                    )
                    self._notifier.notify_task_failed(proxy, ct.last_error or "")

    def _run_chunked_task(
        self,
        ct: ChunkedTask,
        completed_files: list[str],
        ct_log,
    ) -> bool:
        """
        청크 태스크의 모든 섹션을 순서대로 생성한다.
        이전 섹션 코드를 컨텍스트로 누적해 맥락을 유지한다.
        """
        self._fs.backup_file(ct.file_path)

        for section in ct.sections:
            s_log = ct_log.bind(
                section=section.section_name,
                idx=section.section_index,
            )
            s_log.info("agent.section.start")

            # 이 섹션보다 앞에 이미 생성된 섹션들
            prev_sections = [
                s for s in ct.sections
                if s.section_index < section.section_index
                and s.status == ChunkStatus.GENERATED
            ]

            success = self._run_section_with_healing(
                ct=ct,
                section=section,
                prev_sections=prev_sections,
                completed_files=completed_files,
                s_log=s_log,
            )

            if not success:
                ct_log.error(
                    "agent.section.failed",
                    section=section.section_name,
                )
                return False

            s_log.info("agent.section.done")

        # 모든 섹션 생성 완료 → 조립 → post-process → lint
        return self._assemble_and_lint_chunked(ct, ct_log)

    def _run_section_with_healing(
        self,
        ct: ChunkedTask,
        section: FileSection,
        prev_sections: list[FileSection],
        completed_files: list[str],
        s_log,
    ) -> bool:
        """섹션 생성 with self-heal 재시도."""
        error_feedback = ""

        for attempt in range(1, DEFAULT_SECTION_RETRIES + 1):
            skeleton = self._get_skeleton()
            lang_name = (
                self._language_profile.name
                if self._language_profile else "code"
            )

            try:
                code = self._planner.generate_section(
                    file_path=ct.file_path,
                    section=section,
                    previous_sections=prev_sections,
                    workspace_skeleton=skeleton,
                    error_feedback=error_feedback,
                    language_name=lang_name,
                )
            except (LLMError, LLMParseError) as exc:
                error_feedback = f"생성 오류: {exc}"
                s_log.warning(
                    "agent.section.llm_error",
                    attempt=attempt,
                    error=str(exc)[:200],
                )
                continue

            # 섹션 코드 기본 검증
            code = code.strip()
            if not code:
                error_feedback = "빈 코드가 반환됨. 내용을 완성하세요."
                continue

            section.generated_code = code + "\n"
            section.status = ChunkStatus.GENERATED
            s_log.info(
                "agent.section.generated",
                attempt=attempt,
                chars=len(code),
            )
            return True

        section.status = ChunkStatus.FAILED
        ct.error_history.append(
            f"섹션 '{section.section_name}' 생성 실패 "
            f"({DEFAULT_SECTION_RETRIES}회 시도)"
        )
        return False

    def _assemble_and_lint_chunked(
        self, ct: ChunkedTask, ct_log
    ) -> bool:
        """
        생성된 모든 섹션을 합쳐 하나의 파일로 조립하고 lint 를 실행한다.
        lint 실패 시 전체 파일 재생성(일반 태스크 모드)으로 폴백한다.
        """
        assembled = ct.assembled_code

        # post-process
        from agent.domain.entities import CodePatch
        try:
            patch = CodePatch(
                target_file=ct.file_path,
                explanation="assembled from sections",
                code_block=assembled,
            )
            patch = self._post_processor.process(patch)
            assembled = patch.code_block
        except TruncatedCodeError:
            pass  # 섹션 조립 파일은 truncation 체크 완화

        self._fs.write_file(ct.file_path, assembled, self._dry_run)
        abs_path = self._fs.get_absolute_path(ct.file_path)
        lint = self._sensor.verify_code(abs_path, self._dry_run)

        if lint.passed:
            ct_log.info(
                "agent.task.lint_pass",
                tool=lint.tool,
                total_chars=len(assembled),
            )
            return True

        ct_log.warning(
            "agent.task.assembled_lint_fail",
            errors=lint.errors[:5],
        )

        # lint 실패 → 일반 태스크 모드로 폴백 재시도
        ct_log.info("agent.task.fallback_normal_mode")
        proxy_task = Task(
            task_id=ct.task_id,
            file_path=ct.file_path,
            action=ct.action,
            description=ct.description,
        )
        proxy_task.record_retry_error(lint.error_summary)

        return self._run_task_with_healing(
            proxy_task, [], ct_log
        )

    # ── Build & execution validation ───────────────────────────────────────────

    def _run_build_and_exec_validation(
        self,
        summary: RunSummary,
        plan: Optional[EnhancedProjectPlan] = None,
    ) -> None:
        """빌드 → 실행 검증. 실패 시 자동 수정 재시도."""
        task_failed = sum(
            1 for t in summary.tasks if t.status == TaskStatus.FAILED
        )
        if task_failed > 0:
            log.info(
                "agent.validation.skipped",
                reason=f"{task_failed} task(s) failed – skip build/exec",
            )
            return

        assert self._execution_validator is not None
        workspace = self._fs.get_absolute_path(".")

        for attempt in range(1, self._build_fix_retries + 2):
            # ── Build ──────────────────────────────────────────────────────────
            if self._enable_build:
                log.info("agent.build.start", attempt=attempt)
                build_result = self._execution_validator.build(
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
                        attempt=attempt,
                    )
                    # 마지막 재시도가 아니면 빌드 에러를 LLM 에 전달해 수정 시도
                    if attempt <= self._build_fix_retries:
                        fixed = self._try_fix_build_error(
                            build_result, workspace, attempt
                        )
                        if not fixed:
                            break
                        continue
                    break
            else:
                summary.build_passed = None

            # ── Execution ──────────────────────────────────────────────────────
            if self._enable_exec:
                log.info("agent.execution.start")
                exec_result = self._execution_validator.run_smoke_test(
                    workspace, self._dry_run
                )
                summary.execution_passed = exec_result.passed

                if exec_result.passed:
                    log.info(
                        "agent.execution.pass",
                        command=exec_result.command,
                        stdout_preview=exec_result.stdout[:200],
                    )
                    return  # 성공
                else:
                    log.error(
                        "agent.execution.fail",
                        command=exec_result.command,
                        exit_code=exec_result.exit_code,
                        error=exec_result.error_summary[:300],
                        attempt=attempt,
                    )
                    # 실행 실패도 재시도
                    if attempt <= self._build_fix_retries:
                        self._try_fix_exec_error(exec_result, workspace, attempt)
                        continue
                    break
            else:
                summary.execution_passed = None
                return

    def _try_fix_build_error(
        self,
        build_result: BuildResult,
        workspace: str,
        attempt: int,
    ) -> bool:
        """
        빌드 에러를 분석해 관련 파일을 자동으로 수정 시도.

        에러 메시지에서 파일명을 추출하고 해당 파일에 에러 피드백을 줘서 재생성.
        외부 라이브러리 헤더 누락(fatal error: X.h: No such file) 인 경우
        해당 라이브러리 의존을 제거하라는 명시적 지침을 추가한다.
        """
        log.info("agent.build.auto_fix", attempt=attempt)
        error_text = "\n".join(build_result.errors)

        missing_header_hint = self._detect_missing_header_hint(error_text)

        # 에러에서 파일 경로 추출
        affected_files = self._extract_files_from_errors(error_text, workspace)
        if not affected_files:
            log.warning("agent.build.auto_fix.no_files")
            return False

        fixed_any = False
        for rel_path in affected_files[:3]:  # 최대 3개 파일만 수정 시도
            log.info("agent.build.fix_file", file=rel_path)
            fix_task = Task(
                task_id=0,
                file_path=rel_path,
                action=TaskAction.MODIFY,
                description=f"Fix build error in {rel_path}",
            )
            feedback = (
                f"빌드 에러 (시도 {attempt}):\n{error_text[:800]}\n\n"
                + missing_header_hint
                + self._get_header_context()
            )
            fix_task.record_retry_error(feedback)

            fix_log = log.bind(file=rel_path, mode="build_fix")
            if self._run_task_with_healing(fix_task, [], fix_log):
                fixed_any = True
                log.info("agent.build.fix_file.done", file=rel_path)
            else:
                log.warning("agent.build.fix_file.failed", file=rel_path)

        return fixed_any

    @staticmethod
    def _detect_missing_header_hint(error_text: str) -> str:
        """
        'fatal error: X.h: No such file or directory' (또는 한글 메시지)
        패턴을 감지해 외부 라이브러리 의존 제거 지침을 생성한다.
        """
        import re

        pattern = re.compile(
            r"fatal error:\s*([\w./]+\.(?:h|hpp)):\s*"
            r"(?:No such file or directory|그런 파일이나 디렉터리가 없습니다)"
        )
        missing = pattern.findall(error_text)
        if not missing:
            return ""

        headers_list = ", ".join(sorted(set(missing)))
        return (
            f"\n⚠️  치명적 빌드 오류: 다음 헤더 파일을 찾을 수 없습니다: "
            f"{headers_list}\n"
            "이 빌드 환경에는 서드파티 라이브러리(curl, openssl, json-c 등)가 "
            "설치되어 있지 않습니다.\n"
            "수정 방법:\n"
            f"1. {headers_list} 를 #include 하지 마세요.\n"
            "2. 표준 C/C++ 라이브러리와 POSIX 헤더(sys/socket.h, "
            "netinet/in.h, arpa/inet.h, unistd.h)만 사용하세요.\n"
            "3. HTTP 통신이 필요하면 POSIX 소켓으로 직접 구현하세요.\n"
            "4. JSON 파싱이 필요하면 간단한 수동 파서를 직접 작성하세요.\n\n"
        )

    def _try_fix_exec_error(
        self,
        exec_result: ExecutionResult,
        workspace: str,
        attempt: int,
    ) -> bool:
        """실행 에러를 분석해 자동 수정 시도."""
        log.info("agent.exec.auto_fix", attempt=attempt)
        error_text = exec_result.error_summary

        affected_files = self._extract_files_from_errors(error_text, workspace)
        if not affected_files:
            return False

        for rel_path in affected_files[:2]:
            fix_task = Task(
                task_id=0,
                file_path=rel_path,
                action=TaskAction.MODIFY,
                description=f"Fix runtime error in {rel_path}",
            )
            fix_task.record_retry_error(
                f"실행 오류:\n{error_text[:600]}"
            )
            fix_log = log.bind(file=rel_path, mode="exec_fix")
            self._run_task_with_healing(fix_task, [], fix_log)

        return True

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_skeleton(self) -> str:
        return self._fs.get_workspace_skeleton(
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

    def _build_error_feedback(self, task: Task) -> str:
        if not task.error_history:
            return ""
        last_error = task.error_history[-1]
        header_note = self._get_header_context()

        return (
            f"=== 이전 시도 #{task.retry_count} 실패 – 아래 문제를 수정하세요 ===\n"
            f"{last_error}\n"
            f"{header_note}"
            "=== 수정 지침 ===\n"
            "1. 에러 줄을 찾아 정확히 수정하세요.\n"
            "2. 헤더 파일 선언과 정확히 일치하는 시그니처를 사용하세요.\n"
            "3. 전체 파일을 완성된 형태로 반환하세요.\n"
            "4. 줄 끝 공백 없이, 코드가 잘리지 않도록 완성하세요.\n"
        )

    def _get_header_context(self) -> str:
        """워크스페이스의 헤더 파일 전체 내용을 수집한다."""
        if not (
            self._language_profile
            and hasattr(self._language_profile, "header_extensions")
        ):
            return ""
        header_exts = self._language_profile.header_extensions
        result = []
        try:
            ws = self._fs.get_absolute_path(".")
            for p in sorted(Path(ws).rglob("*")):
                if (
                    p.suffix.lower() in header_exts
                    and ".harness_backups" not in str(p)
                    and ".build" not in str(p)
                ):
                    rel = os.path.relpath(str(p), ws)
                    try:
                        text = p.read_text(encoding="utf-8")
                        result.append(
                            f"\n=== 헤더 파일 선언 [{rel}] ===\n{text}"
                            "=== 헤더 끝 ===\n"
                        )
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
            "이미 완료된 파일:\n"
            + "\n".join(f"  - {p}" for p in completed_files)
        )

    @staticmethod
    def _extract_files_from_errors(
        error_text: str, workspace: str
    ) -> list[str]:
        """에러 메시지에서 상대 파일 경로를 추출한다."""
        import re
        # workspace 절대경로 prefix 를 제거해 상대경로 추출
        ws_normalized = workspace.rstrip("/") + "/"
        pattern = re.compile(
            r'(?:' + re.escape(ws_normalized) + r')?'
            r'((?:src|include|lib|tests?)/[^\s:,]+\.[a-zA-Z]+)',
        )
        found = []
        seen = set()
        for m in pattern.finditer(error_text):
            rel = m.group(1)
            if rel not in seen:
                seen.add(rel)
                found.append(rel)
        return found

    def _update_summary_task(
        self, summary: RunSummary, file_path: str, status: TaskStatus
    ) -> None:
        """RunSummary 의 proxy 태스크 상태를 업데이트한다."""
        for t in summary.tasks:
            if t.file_path == file_path:
                t.status = status
                if status == TaskStatus.DONE:
                    t.mark_done()
                break

    def _send_completion_notification(self, summary: RunSummary) -> None:
        if self._notifier is None:
            return
        try:
            self._notifier.notify_run_complete(summary)
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.notification.error", error=str(exc))
