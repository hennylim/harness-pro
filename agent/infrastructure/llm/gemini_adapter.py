"""
Google Gemini LLM adapter (google-genai SDK).

잘림(Truncation) 대응 전략
---------------------------
1. max_output_tokens 를 설정값의 2배까지 자동 증가 (최대 32768).
   코드 생성 요청은 plan 생성보다 토큰을 훨씬 많이 사용하므로
   generate_code() 는 max_tokens * 2 로 호출한다.

2. JSON 파싱 실패 시 json_recovery.recover_code_patch_json() 으로
   부분 복구를 시도한다.
   - code_block 이 잘린 경우: 잘린 위치까지 추출해 반환.
   - code_block 이 비어 있는 경우: LLMParseError 로 재시도 유도.

3. generate_code() 재시도 피드백에 "응답이 잘렸습니다" 메시지를 포함.
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
from agent.infrastructure.llm.json_recovery import recover_code_patch_json
from agent.infrastructure.llm.prompt_builder import PromptBuilder

log = structlog.get_logger(__name__)

# 코드 생성은 plan 보다 훨씬 많은 토큰을 사용하므로 배수를 적용한다.
_CODE_TOKEN_MULTIPLIER = 2
_MAX_TOKENS_HARD_LIMIT = 32768


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
        언어별 프롬프트 빌더.
    temperature:
        생성 온도.
    max_tokens:
        plan 생성 최대 토큰. code 생성은 이 값의 2배를 사용.
    max_retries:
        재시도 최대 횟수.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-preview-05-20",
        prompt_builder: PromptBuilder | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._prompts = prompt_builder or _default_builder()
        self._temperature = temperature
        self._max_tokens = max_tokens
        # 코드 생성용 토큰: 설정값의 2배 (최대 32768)
        self._code_max_tokens = min(
            max_tokens * _CODE_TOKEN_MULTIPLIER, _MAX_TOKENS_HARD_LIMIT
        )
        self._max_retries = max_retries
        self._client = genai.Client(api_key=api_key)
        log.info(
            "gemini.adapter.init",
            model=model,
            plan_tokens=max_tokens,
            code_tokens=self._code_max_tokens,
        )

    # ── ILLMAdapter ───────────────────────────────────────────────────────────

    def generate_plan(self, requirements: str, repo_context: str = "") -> ProjectPlan:
        log.info("llm.plan.request", provider="gemini", model=self._model)
        user = (
            f"Repository context:\n{repo_context}\n\nRequirements:\n{requirements}"
            if repo_context else f"Requirements:\n{requirements}"
        )
        raw = self._call(
            system=self._prompts.build_plan_system(),
            user=user,
            max_tokens=self._max_tokens,
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
        repo_context: str = "",
    ) -> CodePatch:
        log.info(
            "llm.code.request",
            provider="gemini",
            file=task.file_path,
            retry=task.retry_count,
            max_tokens=self._code_max_tokens,
        )
        parts = [
            f"Task: [{task.action.value}] {task.file_path}",
            f"Description: {task.description}",
        ]
        if repo_context:
            parts.append(f"\nRepository context:\n{repo_context}")
        if memory_summary:
            parts.append(f"\nAlready completed files:\n{memory_summary}")
        if workspace_skeleton:
            parts.append(
                f"\nWorkspace skeleton (first 10 lines each):\n{workspace_skeleton}"
            )
        if error_feedback:
            parts.append(f"\n⚠️  Previous errors to fix:\n{error_feedback}")

        raw = self._call(
            system=self._prompts.build_code_system(),
            user="\n".join(parts),
            max_tokens=self._code_max_tokens,
        )

        # ── JSON 파싱 (잘림 복구 포함) ─────────────────────────────────────────
        try:
            data = recover_code_patch_json(raw)
        except (json.JSONDecodeError, Exception) as exc:
            raise LLMParseError(
                f"CodePatch 파싱 실패: {exc}\nRaw:\n{raw[:500]}"
            ) from exc

        # code_block 이 비어 있으면 잘림으로 판단 → 재시도 유도
        if not data.get("code_block", "").strip():
            raise LLMParseError(
                f"code_block 이 비어 있습니다 (응답 잘림 의심).\n"
                f"max_output_tokens={self._code_max_tokens} 로 재시도합니다.\n"
                f"Raw 마지막 200자: {raw[-200:]}"
            )

        try:
            patch = CodePatch(**data)
        except ValidationError as exc:
            raise LLMParseError(
                f"CodePatch 검증 실패: {exc}\ndata={data}"
            ) from exc

        log.info("llm.code.ok", file=patch.target_file,
                 code_chars=len(patch.code_block))
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
            max_tokens=512,
        )

    def generate_raw(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
    ) -> str:
        """
        임의의 system/user 프롬프트로 직접 호출한다.
        AdaptivePlanner 의 강화 계획/섹션 생성에서 사용.
        """
        effective_tokens = max_tokens or self._code_max_tokens
        return self._call(system=system, user=user, max_tokens=effective_tokens)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call(self, system: str, user: str, max_tokens: int) -> str:
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
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                    ),
                )
                text = response.text or ""
                usage = response.usage_metadata
                log.debug(
                    "gemini.raw_response",
                    tokens_in=getattr(usage, "prompt_token_count", None),
                    tokens_out=getattr(usage, "candidates_token_count", None),
                    max_tokens=max_tokens,
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
