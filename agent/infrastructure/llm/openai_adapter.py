"""
OpenAI-compatible LLM adapter.

Works with OpenAI, Azure OpenAI, Ollama, and any other provider that
exposes the OpenAI chat-completion API.

Resilience
----------
* Exponential-backoff retries via ``tenacity`` for transient errors.
* Hard timeout enforced per call.
* JSON schema is validated against the expected Pydantic model before
  returning, so the orchestrator never receives a malformed entity.
"""
from __future__ import annotations

import json
from typing import Any

import structlog
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import ValidationError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent.domain.entities import CodePatch, ProjectPlan, Task
from agent.domain.exceptions import (
    LLMParseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMError,
)
from agent.domain.interfaces import ILLMAdapter

log = structlog.get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_PLAN_SYSTEM = """\
You are an expert software architect.
Analyse the requirements and produce a JSON task plan.
Return ONLY a single JSON object matching this schema (no markdown, no commentary):
{
  "tasks": [
    {
      "task_id": <int>,
      "file_path": "<relative path, e.g. src/math_tool.py>",
      "action": "<create|modify|delete|refactor>",
      "description": "<one sentence describing exactly what must be implemented>"
    }
  ]
}
Rules:
- task_id starts at 1 and is sequential.
- file_path must be a relative path, no leading slash, no "..".
- List tasks in dependency order (dependencies first).
- Be specific: the description must be implementable without further context.
"""

_CODE_SYSTEM = """\
You are a senior software engineer writing production-quality Python code.
Return ONLY a single JSON object (no markdown, no commentary):
{
  "target_file": "<same relative path as the task>",
  "explanation": "<one paragraph: what this file does and key design decisions>",
  "code_block": "<complete, runnable Python source code as a single string>"
}
Rules:
- code_block must contain the ENTIRE file, not just a snippet.
- Follow PEP-8; lines ≤ 79 characters.
- Add module-level and function-level docstrings.
- Include all necessary imports.
- Do NOT wrap in markdown fences.
"""

_SUMMARISE_SYSTEM = """\
You are a technical writer summarising a code-generation run.
Return a concise (≤ 150 words) plain-text summary suitable for a Slack message.
"""


class OpenAICompatibleAdapter(ILLMAdapter):
    """
    Adapter for any OpenAI-compatible API endpoint.

    Parameters
    ----------
    model:
        Model identifier (e.g. "gpt-4o", "qwen2.5-coder:7b").
    base_url:
        API base URL (e.g. "https://api.openai.com/v1").
    api_key:
        API key (use "ollama" for local Ollama instances).
    temperature:
        Sampling temperature; 0.0 for deterministic output.
    max_tokens:
        Maximum tokens in the LLM response.
    timeout_seconds:
        Hard per-call timeout.
    max_retries:
        Maximum retry attempts for transient errors.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(timeout_seconds),
            max_retries=0,   # tenacity handles retries
        )

    # ── ILLMAdapter ───────────────────────────────────────────────────────────

    def generate_plan(self, requirements: str) -> ProjectPlan:
        log.info("llm.plan.request", model=self._model)
        raw = self._call_with_retry(
            system=_PLAN_SYSTEM,
            user=f"Requirements:\n{requirements}",
        )
        try:
            data = json.loads(raw)
            plan = ProjectPlan(**data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMParseError(
                f"Could not parse plan response: {exc}\nRaw:\n{raw[:500]}"
            ) from exc
        log.info("llm.plan.ok", task_count=len(plan.tasks))
        return plan

    def generate_code(
        self,
        task: Task,
        memory_summary: str,
        workspace_skeleton: str,
        error_feedback: str,
    ) -> CodePatch:
        log.info("llm.code.request", file=task.file_path, retry=task.retry_count)
        context_parts = [
            f"Task: [{task.action.value}] {task.file_path}",
            f"Description: {task.description}",
        ]
        if memory_summary:
            context_parts.append(f"\nAlready completed files:\n{memory_summary}")
        if workspace_skeleton:
            context_parts.append(f"\nWorkspace skeleton (first 10 lines each):\n{workspace_skeleton}")
        if error_feedback:
            context_parts.append(f"\n⚠️  Previous lint errors to fix:\n{error_feedback}")

        raw = self._call_with_retry(
            system=_CODE_SYSTEM,
            user="\n".join(context_parts),
        )
        try:
            data = json.loads(raw)
            patch = CodePatch(**data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMParseError(
                f"Could not parse code response: {exc}\nRaw:\n{raw[:500]}"
            ) from exc
        log.info("llm.code.ok", file=patch.target_file)
        return patch

    def summarise_progress(
        self,
        completed_files: list[str],
        failed_files: list[str],
    ) -> str:
        completed_str = "\n".join(f"  ✅ {f}" for f in completed_files) or "  (none)"
        failed_str = "\n".join(f"  ❌ {f}" for f in failed_files) or "  (none)"
        user = (
            f"Completed:\n{completed_str}\n\n"
            f"Failed:\n{failed_str}\n\n"
            "Write a brief summary."
        )
        return self._call_with_retry(system=_SUMMARISE_SYSTEM, user=user)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call_with_retry(self, system: str, user: str) -> str:
        """
        Call the API with exponential-backoff retry.
        Maps provider exceptions to domain exceptions before surfacing them.
        """
        @retry(
            retry=retry_if_exception_type((APIConnectionError, APIStatusError)),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=False,
        )
        def _inner() -> str:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                log.debug(
                    "llm.raw_response",
                    tokens_in=response.usage.prompt_tokens if response.usage else None,
                    tokens_out=response.usage.completion_tokens if response.usage else None,
                )
                return content
            except APITimeoutError as exc:
                raise LLMTimeoutError(
                    f"LLM call timed out after {self._timeout}s"
                ) from exc
            except RateLimitError as exc:
                raise LLMRateLimitError("Rate limit hit") from exc
            except APIStatusError as exc:
                log.warning("llm.api_error", status=exc.status_code, detail=str(exc))
                raise

        try:
            return _inner()
        except RetryError as exc:
            raise LLMError(
                f"LLM call failed after {self._max_retries} retries."
            ) from exc
        except (LLMTimeoutError, LLMRateLimitError):
            raise
        except Exception as exc:
            raise LLMError(f"Unexpected LLM error: {exc}") from exc
