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
    def generate_plan(self, requirements: str, repo_context: str = "") -> ProjectPlan:
        """Convert a free-text requirements string into an ordered task plan."""

    @abstractmethod
    def generate_code(
        self,
        task: Task,
        memory_summary: str,
        workspace_skeleton: str,
        error_feedback: str,
        repo_context: str = "",
    ) -> CodePatch:
        """Generate (or regenerate) source code for a single task."""

    @abstractmethod
    def summarise_progress(
        self,
        completed_files: list[str],
        failed_files: list[str],
    ) -> str:
        """Return a short natural-language summary of what has been built."""

    @abstractmethod
    def generate_raw(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        임의의 system/user 프롬프트로 LLM 을 직접 호출해 raw 텍스트를 반환한다.

        AdaptivePlanner 같은 상위 유스케이스가 표준 generate_plan/generate_code
        스키마에 맞지 않는 커스텀 프롬프트(강화 계획, 섹션 생성)를 보낼 때 사용한다.

        Parameters
        ----------
        system:
            System 프롬프트.
        user:
            User 프롬프트.
        max_tokens:
            None 이면 어댑터의 코드 생성 기본값(보통 가장 큰 토큰 한도)을 사용한다.
        """


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
        header_extensions: frozenset | None = None,
        max_lines_per_file: int = 30,
    ) -> str:
        """Return a compact text representation of existing workspace files.

        Parameters
        ----------
        accepted_extensions:
            파일 확장자 필터 (점 포함, 소문자).
        header_extensions:
            전체 내용을 포함할 헤더 확장자 (기본: .h, .hpp).
        max_lines_per_file:
            일반 소스 파일 최대 포함 줄 수.
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


class IAdaptivePlanner(ABC):
    """
    Port for adaptive planning.

    요구사항을 분석해 파일 복잡도를 예측하고,
    큰 파일은 자동으로 섹션으로 분할하는 강화 플래너.
    """

    @abstractmethod
    def generate_enhanced_plan(
        self,
        requirements: str,
        chunk_threshold_lines: int,
        repo_context: str = "",
    ) -> "EnhancedProjectPlan":
        """
        요구사항을 분석해 EnhancedProjectPlan 을 생성한다.

        복잡한 파일은 자동으로 ChunkedTask 로 분류한다.

        Parameters
        ----------
        requirements:
            사용자 요구사항 자유 형식 문자열.
        chunk_threshold_lines:
            이 줄 수 이상으로 예상되는 파일은 청크 분할 대상.
        """

    @abstractmethod
    def generate_section(
        self,
        file_path: str,
        section: "FileSection",
        previous_sections: list["FileSection"],
        workspace_skeleton: str,
        error_feedback: str,
        language_name: str,
        repo_context: str = "",
    ) -> str:
        """
        청크 분할된 파일의 한 섹션을 생성한다.

        Parameters
        ----------
        file_path:
            대상 파일 경로.
        section:
            생성할 섹션 정보.
        previous_sections:
            이미 생성된 이전 섹션들 (컨텍스트 유지용).
        workspace_skeleton:
            현재 워크스페이스 파일 구조.
        error_feedback:
            이전 시도에서의 에러 피드백.
        language_name:
            대상 언어 이름.

        Returns
        -------
        str
            생성된 섹션 코드.
        """
