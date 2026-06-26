"""
Google Gemini LLM adapter (google-genai SDK).

언어 확장 포인트
---------------
PromptBuilder 를 통해 언어별 프롬프트를 받으므로,
새 언어를 추가해도 이 파일은 수정하지 않는다.
"""
from __future__ import annotations

import json

import structlog
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
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
)
from agent.domain.interfaces import ILLMAdapter
from agent.infrastructure.llm.prompt_builder import PromptBuilder

log = structlog.get_logger(__name__)


class GeminiAdapter(ILLMAdapter):
    """
    Native google-genai adapter for Gemini models.

    Parameters
    ----------
    api_key:
        Google AI Studio API 키.
    model:
        Gemini 모델명.
    prompt_builder:
        언어별 프롬프트 빌더. None 이면 Python 기본값 사용.
    temperature / max_tokens / max_retries:
        API 호출 파라미터.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-preview-05-20",
        prompt_builder: PromptBuilder | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._prompts = prompt_builder or _default_builder()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = genai.Client(api_key=api_key)
        log.info("gemini.adapter.init", model=model)

    # ── ILLMAdapter ───────────────────────────────────────────────────────────

    def generate_plan(self, requirements: str) -> ProjectPlan:
        log.info("llm.plan.request", provider="gemini", model=self._model)
        raw = self._call(
            system=self._prompts.build_plan_system(),
            user=f"Requirements:\n{requirements}",
        )
        try:
            plan = ProjectPlan(**json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMParseError(f"Plan 파싱 실패: {exc}\nRaw:\n{raw[:500]}") from exc
        log.info("llm.plan.ok", task_count=len(plan.tasks))
        return plan

    def generate_code(
        self,
        task: Task,
        memory_summary: str,
        workspace_skeleton: str,
        error_feedback: str,
    ) -> CodePatch:
        log.info("llm.code.request", provider="gemini", file=task.file_path)
        parts = [
            f"Task: [{task.action.value}] {task.file_path}",
            f"Description: {task.description}",
        ]
        if memory_summary:
            parts.append(f"\nAlready completed files:\n{memory_summary}")
        if workspace_skeleton:
            parts.append(
                f"\nWorkspace source snapshot (bounded preview of existing files):\n{workspace_skeleton}"
            )
        if error_feedback:
            parts.append(f"\n⚠️  Previous lint errors to fix:\n{error_feedback}")

        raw = self._call(
            system=self._prompts.build_code_system(),
            user="\n".join(parts),
        )
        try:
            patch = CodePatch(**json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMParseError(f"CodePatch 파싱 실패: {exc}\nRaw:\n{raw[:500]}") from exc
        log.info("llm.code.ok", file=patch.target_file)
        return patch

    def summarise_progress(
        self,
        completed_files: list[str],
        failed_files: list[str],
    ) -> str:
        completed_str = "\n".join(f"  ✅ {f}" for f in completed_files) or "  (none)"
        failed_str = "\n".join(f"  ❌ {f}" for f in failed_files) or "  (none)"
        return self._call(
            system=PromptBuilder.build_summarise_system(),
            user=(
                f"Completed:\n{completed_str}\n\n"
                f"Failed:\n{failed_str}\n\nWrite a brief summary."
            ),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call(self, system: str, user: str) -> str:
        @retry(
            retry=retry_if_exception_type(genai_errors.ServerError),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=False,
        )
        def _inner() -> str:
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=user,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=self._temperature,
                        max_output_tokens=self._max_tokens,
                        response_mime_type="application/json",
                    ),
                )
                text = response.text or ""
                usage = response.usage_metadata
                log.debug(
                    "gemini.raw_response",
                    tokens_in=getattr(usage, "prompt_token_count", None),
                    tokens_out=getattr(usage, "candidates_token_count", None),
                )
                return text
            except genai_errors.ClientError as exc:
                if hasattr(exc, "status_code") and exc.status_code == 429:
                    raise LLMRateLimitError("Gemini rate limit") from exc
                raise LLMError(f"Gemini client error: {exc}") from exc

        try:
            return _inner()
        except RetryError as exc:
            raise LLMError(
                f"Gemini 호출 {self._max_retries}회 재시도 후 실패"
            ) from exc
        except (LLMRateLimitError, LLMParseError):
            raise
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Gemini 예상치 못한 오류: {exc}") from exc


def _default_builder() -> PromptBuilder:
    from agent.domain.profiles import PROFILE_REGISTRY
    return PromptBuilder(PROFILE_REGISTRY["python"])
