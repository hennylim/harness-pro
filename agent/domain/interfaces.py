"""
Abstract interfaces (ports) that the infrastructure layer must implement.
The usecase layer depends only on these; never on concrete adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from agent.domain.entities import (
    CodePatch,
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
    def get_workspace_skeleton(self) -> str:
        """Return a compact text representation of existing workspace files."""

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
