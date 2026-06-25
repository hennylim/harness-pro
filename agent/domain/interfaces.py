"""
Abstract interfaces (ports) that the infrastructure layer must implement.
The usecase layer depends only on these; never on concrete adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from agent.domain.entities import (
    BuildResult,
    CodePatch,
    ExecutionResult,
    LintResult,
    ProjectPlan,
    RunSummary,
    Task,
)


class ILLMAdapter(ABC):
    """Port for all large-language-model interactions."""

    @abstractmethod
    def generate_plan(self, requirements: str) -> ProjectPlan:
        """Convert a free-text requirements string into an ordered task plan."""

    @abstractmethod
    def generate_code(
        self,
        task: Task,
        memory_summary: str,
        workspace_skeleton: str,
        error_feedback: str,
    ) -> CodePatch:
        """Generate (or regenerate) source code for a single task."""

    @abstractmethod
    def summarise_progress(
        self,
        completed_files: list[str],
        failed_files: list[str],
    ) -> str:
        """Return a short natural-language summary of what has been built."""


class IFileSystemAdapter(ABC):
    """Port for all workspace file-system operations."""

    @abstractmethod
    def get_absolute_path(self, relative_path: str) -> str:
        """Resolve a workspace-relative path to an absolute OS path."""

    @abstractmethod
    def write_file(self, relative_path: str, content: str, dry_run: bool) -> None:
        """Write *content* to *relative_path* inside the workspace."""

    @abstractmethod
    def read_file(self, relative_path: str) -> str:
        """Read the content of a file inside the workspace."""

    @abstractmethod
    def delete_file(self, relative_path: str, dry_run: bool) -> None:
        """Delete a file from the workspace."""

    @abstractmethod
    def get_workspace_skeleton(
        self,
        accepted_extensions: frozenset | None = None,
    ) -> str:
        """Return a compact text representation of existing workspace files.

        Parameters
        ----------
        accepted_extensions:
            파일 확장자 필터 (점 포함, 소문자).
            None 이면 구현체 기본값(보통 .py)을 사용한다.
        """

    @abstractmethod
    def backup_file(self, relative_path: str) -> Optional[str]:
        """Create a timestamped backup; return backup path or None if file absent."""


class ISensorAdapter(ABC):
    """Port for static-analysis / code-quality gates."""

    @abstractmethod
    def verify_code(self, absolute_path: str, dry_run: bool) -> LintResult:
        """Run linting on the file at *absolute_path*."""


class INotificationAdapter(ABC):
    """Port for external notifications (Slack, email, webhook, …)."""

    @abstractmethod
    def notify_run_complete(self, summary: RunSummary) -> None:
        """Send a completion notification."""

    @abstractmethod
    def notify_task_failed(self, task: Task, reason: str) -> None:
        """Send an alert when a task cannot be self-healed."""


class IRunRepository(ABC):
    """Port for persisting run summaries."""

    @abstractmethod
    def save(self, summary: RunSummary) -> None:
        """Persist or overwrite a run summary."""

    @abstractmethod
    def load(self, run_id: str) -> Optional[RunSummary]:
        """Load a run summary by its ID. Returns None if not found."""


class IExecutionValidator(ABC):
    """
    Port for build-and-run validation.

    생성된 전체 프로젝트(워크스페이스)를 대상으로
    빌드 → 실행 → 종료코드 검증을 수행한다.

    lint(ISensorAdapter)가 개별 파일을 검사하는 것과 달리,
    IExecutionValidator 는 모든 파일이 완성된 후 프로젝트 전체를 검증한다.
    """

    @abstractmethod
    def build(self, workspace_dir: str, dry_run: bool) -> BuildResult:
        """
        워크스페이스를 빌드(컴파일)한다.
        인터프리터 언어는 문법 검사(syntax check)를 수행한다.

        Returns
        -------
        BuildResult
        """

    @abstractmethod
    def run_smoke_test(self, workspace_dir: str, dry_run: bool) -> ExecutionResult:
        """
        빌드된 프로그램을 실제로 실행하고 결과를 반환한다.
        성공 기준: exit code 0, stderr 에 치명적 오류 없음.

        Returns
        -------
        ExecutionResult
        """
