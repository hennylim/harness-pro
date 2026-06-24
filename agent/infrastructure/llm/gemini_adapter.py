"""
Google Gemini LLM adapter (google-genai SDK).

왜 별도 어댑터인가?
--------------------
Gemini는 OpenAI 호환 엔드포인트(generativelanguage.googleapis.com/v1beta/openai/)를
지원하지만, system 역할 메시지를 ``system_instruction`` 파라미터로 분리해야 하고
``response_mime_type="application/json"`` 으로 JSON 모드를 설정해야 합니다.
네이티브 SDK(google-genai)를 쓰면 이 차이를 정확히 처리할 수 있습니다.

지원 모델 (2025 기준)
---------------------
- gemini-2.5-pro-preview-06-05   ← 최신 추론 모델 (권장)
- gemini-2.5-flash-preview-05-20 ← 속도/비용 균형
- gemini-2.0-flash                ← 경량 빠른 모델
- gemini-1.5-pro / gemini-1.5-flash

재시도 전략
-----------
``tenacity`` 지수 백오프:
  - ``ServerError`` (5xx)   → 재시도
  - ``ClientError`` (429)   → 레이트리밋 → LLMRateLimitError
  - ``ClientError`` (기타)  → 즉시 실패 → LLMError
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
    LLMTimeoutError,
)
from agent.domain.interfaces import ILLMAdapter
from agent.infrastructure.llm.openai_adapter import (
    _CODE_SYSTEM,
    _PLAN_SYSTEM,
    _SUMMARISE_SYSTEM,
)

log = structlog.get_logger(__name__)


class GeminiAdapter(ILLMAdapter):
    """
    Native google-genai adapter for Gemini models.

    Parameters
    ----------
    api_key:
        Google AI Studio API 키 (``GEMINI_API_KEY`` 환경변수로도 설정 가능).
    model:
        Gemini 모델명 (e.g. ``"gemini-2.5-flash-preview-05-20"``).
    temperature:
        생성 온도 (0.0 = 결정적).
    max_tokens:
        최대 출력 토큰 수.
    max_retries:
        재시도 최대 횟수.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-preview-05-20",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = genai.Client(api_key=api_key)
        log.info("gemini.adapter.init", model=model)

    # ── ILLMAdapter ───────────────────────────────────────────────────────────

    def generate_plan(self, requirements: str) -> ProjectPlan:
        log.info("llm.plan.request", provider="gemini", model=self._model)
        raw = self._call(
            system=_PLAN_SYSTEM,
            user=f"Requirements:\n{requirements}",
        )
        try:
            plan = ProjectPlan(**json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMParseError(
                f"Plan 파싱 실패: {exc}\nRaw:\n{raw[:500]}"
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
        log.info("llm.code.request", provider="gemini", file=task.file_path)
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

        raw = self._call(system=_CODE_SYSTEM, user="\n".join(parts))
        try:
            patch = CodePatch(**json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMParseError(
                f"CodePatch 파싱 실패: {exc}\nRaw:\n{raw[:500]}"
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
        return self._call(
            system=_SUMMARISE_SYSTEM,
            user=(
                f"Completed:\n{completed_str}\n\n"
                f"Failed:\n{failed_str}\n\nWrite a brief summary."
            ),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call(self, system: str, user: str) -> str:
        """
        google-genai SDK 호출 + 재시도.

        Gemini는 system 메시지를 ``system_instruction``으로 분리하고,
        JSON 출력을 ``response_mime_type`` 으로 지정합니다.
        """
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
                # 429 Rate Limit
                if hasattr(exc, "status_code") and exc.status_code == 429:
                    raise LLMRateLimitError("Gemini rate limit") from exc
                raise LLMError(f"Gemini client error: {exc}") from exc
            except Exception as exc:
                # ServerError는 tenacity가 재시도하므로 여기서 재발생
                raise

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
