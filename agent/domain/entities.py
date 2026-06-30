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


# ── Chunked file generation ────────────────────────────────────────────────────

class ChunkStatus(str, Enum):
    """청크 단위 파일 생성 상태."""
    PENDING   = "pending"
    GENERATED = "generated"
    FAILED    = "failed"


class FileSection(BaseModel):
    """
    큰 파일을 분할 생성할 때 각 섹션을 표현하는 엔티티.

    Attributes
    ----------
    section_index:
        0-based 섹션 순서 번호.
    section_name:
        섹션 이름 (예: "imports", "data structures", "core logic", "entry point").
    description:
        이 섹션에서 구현할 내용 상세.
    depends_on:
        이전에 생성된 섹션 인덱스 목록 (컨텍스트 참조용).
    generated_code:
        LLM 이 생성한 섹션 코드 (아직 생성 전이면 빈 문자열).
    status:
        섹션 생성 상태.
    """
    section_index: int
    section_name: str
    description: str
    depends_on: List[int] = Field(default_factory=list)
    generated_code: str = ""
    status: ChunkStatus = ChunkStatus.PENDING


class ChunkedTask(BaseModel):
    """
    단일 파일을 여러 섹션으로 나눠 생성하는 태스크.

    LLM 이 한 번에 생성하기 어려운 큰 파일(복잡한 구조, 많은 함수)에 사용.
    섹션 순서대로 생성하며 이전 섹션 코드를 컨텍스트로 유지한다.
    """
    task_id: int
    file_path: str
    action: "TaskAction"
    description: str
    sections: List[FileSection]
    status: TaskStatus = TaskStatus.PENDING
    error_history: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def assembled_code(self) -> str:
        """생성된 모든 섹션을 순서대로 합쳐 하나의 파일로 반환."""
        return "\n".join(
            s.generated_code for s in self.sections
            if s.status == ChunkStatus.GENERATED
        )

    @property
    def is_complete(self) -> bool:
        return all(s.status == ChunkStatus.GENERATED for s in self.sections)

    @property
    def last_error(self) -> Optional[str]:
        return self.error_history[-1] if self.error_history else None

    def mark_done(self) -> None:
        self.status = TaskStatus.DONE

    def mark_failed(self, reason: str) -> None:
        self.status = TaskStatus.FAILED
        self.error_history.append(reason)


class EnhancedProjectPlan(BaseModel):
    """
    일반 태스크(Task)와 청크 태스크(ChunkedTask)를 혼합한 프로젝트 계획.
    LLM 플래너가 파일 복잡도를 분석해 자동으로 분류한다.
    """
    tasks: List[Task] = Field(default_factory=list)
    chunked_tasks: List[ChunkedTask] = Field(default_factory=list)

    @property
    def all_tasks_count(self) -> int:
        return len(self.tasks) + len(self.chunked_tasks)
