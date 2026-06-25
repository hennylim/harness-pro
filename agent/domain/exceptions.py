"""
Domain-level exceptions.
Using typed exceptions lets callers distinguish recoverable from fatal errors.
"""


class HarnessError(Exception):
    """Base exception for all harness errors."""


# ── LLM layer ─────────────────────────────────────────────────────────────────

class LLMError(HarnessError):
    """Unrecoverable error from the LLM provider."""


class LLMParseError(LLMError):
    """The LLM returned a response that could not be parsed into the expected schema."""


class LLMTimeoutError(LLMError):
    """The LLM call exceeded the configured timeout."""


class LLMRateLimitError(LLMError):
    """The LLM provider returned a rate-limit response."""


# ── File system layer ─────────────────────────────────────────────────────────

class FileSystemError(HarnessError):
    """Generic file system error."""


class PathEscapeError(FileSystemError):
    """A generated path attempted to escape the workspace sandbox."""


# ── Lint / sensor layer ───────────────────────────────────────────────────────

class SensorError(HarnessError):
    """A code-quality sensor could not run (tool missing, permission error, …)."""


# ── Orchestrator layer ────────────────────────────────────────────────────────

class PlanValidationError(HarnessError):
    """The generated project plan is structurally invalid."""


class MaxRetriesExceededError(HarnessError):
    """A task exceeded the maximum allowed self-heal attempts."""


class TaskLimitExceededError(HarnessError):
    """The plan contains more tasks than the configured safety limit."""


# ── Build / execution layer ───────────────────────────────────────────────────

class BuildError(HarnessError):
    """Generated code failed to compile or build."""


class ExecutionError(HarnessError):
    """Generated program failed during smoke-test execution."""
