"""
OpenAI-compatible LLM adapter.

Works with OpenAI, Azure OpenAI, Ollama, and any other provider that
exposes the OpenAI chat-completion API.

언어 확장 포인트
---------------
PromptBuilder 를 통해 언어별 프롬프트를 받으므로,
새 언어를 추가해도 이 파일은 수정하지 않는다.
"""
from __future__ import annotations

import json

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
    LLMError,
    LLMParseError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from agent.domain.interfaces import ILLMAdapter
from agent.infrastructure.llm.prompt_builder import PromptBuilder

log = structlog.get_logger(__name__)

# 하위 호환: Gemini 어댑터가 import 하는 상수들
_SUMMARISE_SYSTEM = PromptBuilder.build_summarise_system()


class OpenAICompatibleAdapter(ILLMAdapter):
    """
    Adapter for any OpenAI-compatible API endpoint.

    Parameters
    ----------
    model:
        Model identifier (e.g. "gpt-4o", "qwen2.5-coder:7b").
    base_url:
        API base URL.
    api_key:
        API key ("ollama" for local Ollama).
    prompt_builder:
        언어별 프롬프트 빌더. None 이면 Python 기본값 사용.
    temperature / max_tokens / timeout_seconds / max_retries:
        API 호출 파라미터.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        prompt_builder: PromptBuilder | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._prompts = prompt_builder or _default_builder()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(timeout_seconds),
            max_retries=0,
        )

    # ── ILLMAdapter ───────────────────────────────────────────────────────────

    def generate_plan(self, requirements: str) -> ProjectPlan:
        log.info("llm.plan.request", model=self._model)
        raw = self._call_with_retry(
            system=self._prompts.build_plan_system(),
            user=f"Requirements:\n{requirements}",
        )
        try:
            plan = ProjectPlan(**json.loads(raw))
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
        parts = [
            f"Task: [{task.action.value}] {task.file_path}",
            f"Description: {task.description}",
        ]
        if memory_summary:
            parts.append(f"\nAlready completed files:\n{memory_summary}")
        if workspace_skeleton:
            parts.append(
                f"\nWorkspace skeleton (first 10 lines each):\n{workspace_skeleton}"
            )
        if error_feedback:
            parts.append(f"\n⚠️  Previous lint errors to fix:\n{error_feedback}")

        raw = self._call_with_retry(
            system=self._prompts.build_code_system(),
            user="\n".join(parts),
        )
        try:
            patch = CodePatch(**json.loads(raw))
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
        return self._call_with_retry(
            system=PromptBuilder.build_summarise_system(),
            user=(
                f"Completed:\n{completed_str}\n\n"
                f"Failed:\n{failed_str}\n\nWrite a brief summary."
            ),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call_with_retry(self, system: str, user: str) -> str:
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


def _default_builder() -> PromptBuilder:
    """Python 프로필을 기본값으로 사용하는 빌더 (하위 호환)."""
    from agent.domain.profiles import PROFILE_REGISTRY
    return PromptBuilder(PROFILE_REGISTRY["python"])
