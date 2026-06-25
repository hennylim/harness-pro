"""
Domain entities – the core data model of the harness agent.
These classes are pure Python / Pydantic; they carry no framework dependencies.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class TaskAction(str, Enum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    REFACTOR = "refactor"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


# ── Value Objects ─────────────────────────────────────────────────────────────

_SAFE_PATH_RE = re.compile(r"^[\w\-./]+$")


class SafeFilePath(str):
    """A file path that cannot escape the workspace root."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: str) -> "SafeFilePath":
        if ".." in v or v.startswith("/"):
            raise ValueError(f"Unsafe file path: {v!r}")
        if not _SAFE_PATH_RE.match(v):
            raise ValueError(f"Invalid characters in path: {v!r}")
        return cls(v)


# ── Entities ──────────────────────────────────────────────────────────────────

class Task(BaseModel):
    """A single unit of code-generation work."""

    id: UUID = Field(default_factory=uuid4)
    task_id: int
    file_path: str
    action: TaskAction
    description: str
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_history: List[str] = Field(default_factory=list)

    @field_validator("file_path")
    @classmethod
    def _safe_path(cls, v: str) -> str:
        if ".." in v or v.startswith("/"):
            raise ValueError(f"Unsafe file path rejected: {v!r}")
        return v

    def mark_done(self) -> None:
        self.status = TaskStatus.DONE
        self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, reason: str) -> None:
        self.status = TaskStatus.FAILED
        self.error_history.append(reason)
        self.completed_at = datetime.now(timezone.utc)

    def record_retry_error(self, reason: str) -> None:
        self.retry_count += 1
        self.error_history.append(reason)

    @property
    def last_error(self) -> Optional[str]:
        return self.error_history[-1] if self.error_history else None


class ProjectPlan(BaseModel):
    """Ordered list of tasks returned by the LLM planner."""

    tasks: List[Task]

    @field_validator("tasks")
    @classmethod
    def _no_duplicate_paths(cls, tasks: List[Task]) -> List[Task]:
        create_paths = [t.file_path for t in tasks if t.action == TaskAction.CREATE]
        if len(create_paths) != len(set(create_paths)):
            raise ValueError("Duplicate CREATE targets in plan.")
        return tasks


class CodePatch(BaseModel):
    """A generated source file ready to be written to disk."""

    target_file: str
    explanation: str
    code_block: str

    @field_validator("target_file")
    @classmethod
    def _safe_path(cls, v: str) -> str:
        if ".." in v or v.startswith("/"):
            raise ValueError(f"Unsafe target_file rejected: {v!r}")
        return v

    @field_validator("code_block")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("code_block must not be empty.")
        return v


class LintResult(BaseModel):
    """Outcome of a static-analysis pass."""

    passed: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    tool: str = "unknown"

    @property
    def error_summary(self) -> str:
        return "\n".join(self.errors)


class BuildResult(BaseModel):
    """Outcome of a build/compile step."""

    passed: bool
    errors: List[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    tool: str = "unknown"

    @property
    def error_summary(self) -> str:
        return "\n".join(self.errors) if self.errors else self.stderr[:500]


class ExecutionResult(BaseModel):
    """Outcome of running the generated program."""

    passed: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    command: str = ""
    error_summary: str = ""


class RunSummary(BaseModel):
    """Persisted record of a complete agent run."""

    run_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    status: RunStatus = RunStatus.RUNNING
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    build_passed: Optional[bool] = None
    execution_passed: Optional[bool] = None
    tasks: List[Task] = Field(default_factory=list)

    def finish(self, status: RunStatus) -> None:
        self.finished_at = datetime.now(timezone.utc)
        self.status = status
        self.completed_tasks = sum(1 for t in self.tasks if t.status == TaskStatus.DONE)
        self.failed_tasks = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        self.skipped_tasks = sum(1 for t in self.tasks if t.status == TaskStatus.SKIPPED)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None
