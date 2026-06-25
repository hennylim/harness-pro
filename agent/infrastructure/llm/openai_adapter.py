"""
OpenAI-compatible LLM adapter.

Works with OpenAI, Azure OpenAI, Ollama, LM Studio, and any other provider
that exposes the OpenAI chat-completion API.

JSON 모드 자동 감지 (LM Studio / 일부 로컬 서버 대응)
-------------------------------------------------------
일부 OpenAI 호환 서버(LM Studio 등)는 ``response_format={"type":"json_object"}``
를 지원하지 않고 400 에러를 반환합니다.

대응 전략:
  1. 첫 호출은 json_object 모드로 시도.
     → 성공: 모드를 캐시하고 그 응답을 바로 반환 (이중 호출 없음).
     → 400 + response_format 관련 메시지: 다음 모드로 폴백.
  2. json_schema 모드로 재시도.
  3. response_format 없이 system prompt 에 JSON 지시를 주입하는 plain_prompt.
  4. 모두 실패하면 LLMError.

감지된 모드는 인스턴스에 캐시되어 이후 호출에서 협상 없이 바로 사용됩니다.

설정으로 강제 지정:
  AI_JSON_MODE=json_object | json_schema | plain_prompt | auto (기본: auto)
"""
from __future__ import annotations

import json
import re
from enum import Enum, auto
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
    LLMError,
    LLMParseError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from agent.domain.interfaces import ILLMAdapter
from agent.infrastructure.llm.prompt_builder import PromptBuilder

log = structlog.get_logger(__name__)

# response_format 관련 400 에러인지 판별하는 패턴
_RF_ERROR_RE = re.compile(
    r"response_format|json_object|json_schema|unsupported.*format",
    re.IGNORECASE,
)

_JSON_ONLY_SUFFIX = (
    "\n\nIMPORTANT: Your entire response MUST be a single valid JSON object. "
    "No markdown, no explanation, no code fences. Raw JSON only."
)


class JsonMode(str, Enum):
    """서버가 지원하는 JSON 출력 모드."""
    AUTO         = "auto"          # 첫 호출 시 자동 감지
    JSON_OBJECT  = "json_object"   # {"type": "json_object"}  ← OpenAI 표준
    JSON_SCHEMA  = "json_schema"   # {"type": "json_schema"}  ← LM Studio
    PLAIN_PROMPT = "plain_prompt"  # response_format 없이 프롬프트로만 강제


def _build_response_format(mode: JsonMode) -> dict | None:
    if mode == JsonMode.JSON_OBJECT:
        return {"type": "json_object"}
    if mode == JsonMode.JSON_SCHEMA:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "harness_response",
                "strict": False,
                "schema": {"type": "object", "additionalProperties": True},
            },
        }
    return None  # PLAIN_PROMPT


def _strip_markdown_fence(text: str) -> str:
    """
    ```json ... ``` 또는 ``` ... ``` 로 감싸진 경우 내부 텍스트만 반환.
    JSON 이 직접 반환된 경우 그대로 반환.
    """
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
    return match.group(1).strip() if match else text


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
        API key ("ollama" for local Ollama, "lm-studio" for LM Studio).
    prompt_builder:
        언어별 프롬프트 빌더. None 이면 Python 기본값 사용.
    json_mode:
        JSON 출력 모드. AUTO 이면 첫 호출 시 서버를 탐색해 자동 결정.
    temperature / max_tokens / timeout_seconds / max_retries:
        API 호출 파라미터.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        prompt_builder: PromptBuilder | None = None,
        json_mode: JsonMode = JsonMode.AUTO,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._prompts = prompt_builder or _default_builder()
        self._json_mode = json_mode
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(timeout_seconds),
            max_retries=0,   # tenacity 가 직접 관리
        )

    # ── ILLMAdapter ───────────────────────────────────────────────────────────

    def generate_plan(self, requirements: str) -> ProjectPlan:
        log.info("llm.plan.request", model=self._model)
        raw = self._call(
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

        raw = self._call(
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
        return self._call(
            system=PromptBuilder.build_summarise_system(),
            user=(
                f"Completed:\n{completed_str}\n\n"
                f"Failed:\n{failed_str}\n\nWrite a brief summary."
            ),
        )

    # ── Internal: 협상 + 재시도 통합 진입점 ──────────────────────────────────

    def _call(self, system: str, user: str) -> str:
        """
        외부에서 사용하는 단일 진입점.

        AUTO 모드이면 _negotiate_and_call() 로 모드를 결정하면서
        첫 응답을 바로 반환 (이중 호출 없음).
        이미 모드가 결정된 경우 _call_with_retry() 로 바로 호출.
        """
        if self._json_mode == JsonMode.AUTO:
            return self._negotiate_and_call(system, user)
        return self._call_with_retry(system, user, self._json_mode)

    def _negotiate_and_call(self, system: str, user: str) -> str:
        """
        모드를 탐색하면서 첫 성공 응답을 바로 반환한다.
        성공한 모드를 캐시하여 이후 호출에서 재탐색하지 않는다.
        """
        probe_order = [JsonMode.JSON_OBJECT, JsonMode.JSON_SCHEMA, JsonMode.PLAIN_PROMPT]

        for mode in probe_order:
            log.info("llm.json_mode.probe", mode=mode.value)
            try:
                result = self._single_raw_call(system, user, mode)
                # 성공 → 모드 캐시 후 결과 바로 반환 (이중 호출 없음)
                self._json_mode = mode
                log.info("llm.json_mode.selected", mode=mode.value)
                return result
            except APIStatusError as exc:
                if exc.status_code == 400 and _RF_ERROR_RE.search(str(exc)):
                    log.warning(
                        "llm.json_mode.unsupported",
                        mode=mode.value,
                        next="trying next mode",
                    )
                    continue
                # response_format 무관한 400 → 즉시 실패
                raise LLMError(f"API error {exc.status_code}: {exc}") from exc
            except (LLMTimeoutError, LLMRateLimitError):
                raise
            except LLMError:
                raise
            except Exception as exc:
                raise LLMError(f"Unexpected error during mode negotiation: {exc}") from exc

        raise LLMError(
            "No supported JSON mode found. "
            "Server rejected json_object, json_schema, and plain_prompt."
        )

    def _call_with_retry(self, system: str, user: str, mode: JsonMode) -> str:
        """확정된 모드로 tenacity 재시도 래퍼를 통해 호출한다."""

        @retry(
            retry=retry_if_exception_type((APIConnectionError, APIStatusError)),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=False,
        )
        def _inner() -> str:
            return self._single_raw_call(system, user, mode)

        try:
            return _inner()
        except RetryError as exc:
            raise LLMError(
                f"LLM call failed after {self._max_retries} retries."
            ) from exc
        except (LLMTimeoutError, LLMRateLimitError, LLMError):
            raise
        except Exception as exc:
            raise LLMError(f"Unexpected LLM error: {exc}") from exc

    def _single_raw_call(self, system: str, user: str, mode: JsonMode) -> str:
        """
        재시도·협상 없는 단일 API 호출.
        성공 시 content 문자열 반환; 실패 시 openai 예외를 그대로 전파.
        """
        effective_system = system
        kwargs: dict[str, Any] = dict(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        rf = _build_response_format(mode)
        if rf is not None:
            kwargs["response_format"] = rf
        else:
            # PLAIN_PROMPT: system 에 JSON 강제 지시 삽입
            effective_system = system + _JSON_ONLY_SUFFIX

        kwargs["messages"] = [
            {"role": "system", "content": effective_system},
            {"role": "user",   "content": user},
        ]

        try:
            response = self._client.chat.completions.create(**kwargs)
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                f"LLM call timed out after {self._timeout}s"
            ) from exc
        except RateLimitError as exc:
            raise LLMRateLimitError("Rate limit hit") from exc

        content = response.choices[0].message.content or ""

        # PLAIN_PROMPT 모드는 마크다운 펜스가 포함될 수 있으므로 제거
        if mode == JsonMode.PLAIN_PROMPT:
            content = _strip_markdown_fence(content)

        log.debug(
            "llm.raw_response",
            mode=mode.value,
            tokens_in=response.usage.prompt_tokens if response.usage else None,
            tokens_out=response.usage.completion_tokens if response.usage else None,
        )
        return content


def _default_builder() -> PromptBuilder:
    """Python 프로필을 기본값으로 사용하는 빌더 (하위 호환)."""
    from agent.domain.profiles import PROFILE_REGISTRY
    return PromptBuilder(PROFILE_REGISTRY["python"])
