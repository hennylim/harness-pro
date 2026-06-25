"""
Dependency injection container.

``build_orchestrator()`` 가 settings 를 읽어 모든 어댑터를 조립합니다.

언어 추가 시 변경 범위
----------------------
이 파일은 건드리지 않는다.
- 언어 프로필: agent/domain/profiles/<lang>.py  +  profiles/__init__.py 등록
- 센서:        agent/infrastructure/sensors/<lang>_sensor.py  +  router.py 등록
- settings:    TargetLanguage Enum 에 값 추가
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

import structlog

from agent.domain.interfaces import ILLMAdapter, INotificationAdapter
from agent.domain.profiles import PROFILE_REGISTRY
from agent.infrastructure.fs.local_adapter import LocalFileSystemAdapter
from agent.infrastructure.fs.run_repository import JsonRunRepository
from agent.infrastructure.llm.gemini_adapter import GeminiAdapter
from agent.infrastructure.llm.openai_adapter import JsonMode
from agent.infrastructure.llm.openai_adapter import OpenAICompatibleAdapter
from agent.infrastructure.llm.prompt_builder import PromptBuilder
from agent.infrastructure.notification.slack_adapter import (
    NullNotificationAdapter,
    SlackWebhookAdapter,
)
from agent.infrastructure.llm.post_processor import CodePostProcessor
from agent.infrastructure.validators import VALIDATOR_REGISTRY
from agent.infrastructure.sensors.router import LanguageSensorRouter
from agent.usecase.orchestrator import HarnessOrchestrator
from config.settings import AIProvider, GEMINI_DEFAULT_MODEL, settings

log = structlog.get_logger(__name__)


def _build_prompt_builder() -> PromptBuilder:
    """settings.target_language → ILanguageProfile → PromptBuilder."""
    lang_key = settings.target_language.value   # e.g. "python", "bash", "c", "cpp"
    profile = PROFILE_REGISTRY.get(lang_key)
    if profile is None:
        raise ValueError(
            f"언어 프로필 '{lang_key}' 을 찾을 수 없습니다. "
            f"PROFILE_REGISTRY 에 등록됐는지 확인하세요. "
            f"등록된 언어: {list(PROFILE_REGISTRY.keys())}"
        )
    log.info("language.profile.selected", language=profile.name)
    return PromptBuilder(profile)


def _build_llm_adapter(prompt_builder: PromptBuilder) -> ILLMAdapter:
    """AI_PROVIDER 값에 따라 적절한 LLM 어댑터를 반환합니다."""
    provider = settings.ai_provider

    if provider == AIProvider.GEMINI:
        api_key = settings.effective_gemini_api_key()
        model = (
            settings.ai_model
            if settings.ai_model != "qwen2.5-coder:7b"
            else GEMINI_DEFAULT_MODEL
        )
        log.info("llm.adapter.selected", provider="gemini", model=model)
        return GeminiAdapter(
            api_key=api_key,
            model=model,
            prompt_builder=prompt_builder,
            temperature=settings.ai_temperature,
            max_tokens=settings.ai_max_tokens,
            max_retries=settings.ai_max_retries,
        )

    log.info(
        "llm.adapter.selected",
        provider=provider.value,
        model=settings.ai_model,
    )
    return OpenAICompatibleAdapter(
        model=settings.ai_model,
        base_url=settings.ai_base_url,
        api_key=settings.ai_api_key,
        prompt_builder=prompt_builder,
        json_mode=JsonMode(settings.ai_json_mode.value),
        temperature=settings.ai_temperature,
        max_tokens=settings.ai_max_tokens,
        timeout_seconds=settings.ai_timeout_seconds,
        max_retries=settings.ai_max_retries,
    )


def build_orchestrator(
    workspace_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> tuple[HarnessOrchestrator, str]:
    """
    모든 컴포넌트를 조립해 HarnessOrchestrator 를 반환합니다.

    Returns
    -------
    tuple[HarnessOrchestrator, str]
        준비된 오케스트레이터와 활성 run_id.
    """
    run_id = run_id or settings.run_id or str(uuid.uuid4())[:8]

    ws = workspace_dir or settings.workspace_dir
    if not os.path.isabs(ws):
        ws = os.path.join(os.getcwd(), ws)
    runs_dir = os.path.join(os.path.dirname(ws), "runs")

    # ── 언어 프로필 → 프롬프트 빌더 ──────────────────────────────────────────
    prompt_builder = _build_prompt_builder()

    # ── 어댑터 조립 ───────────────────────────────────────────────────────────
    llm    = _build_llm_adapter(prompt_builder)
    fs     = LocalFileSystemAdapter(workspace_dir=ws)
    sensor = LanguageSensorRouter()       # 확장자 기반 자동 라우팅
    post_processor = CodePostProcessor(
        fix_trailing_whitespace=settings.post_process_trailing_whitespace,
        fix_unused_imports=settings.post_process_unused_imports,
        detect_truncation=settings.post_process_detect_truncation,
    )
    run_repo = JsonRunRepository(runs_dir=runs_dir)

    # ── Execution validator ────────────────────────────────────────────────
    lang_key = settings.target_language.value
    execution_validator = VALIDATOR_REGISTRY.get(lang_key)

    notifier: INotificationAdapter
    if settings.slack_webhook_url and (
        settings.notify_on_failure or settings.notify_on_success
    ):
        notifier = SlackWebhookAdapter(settings.slack_webhook_url)
        log.info("notifications.enabled", channel="slack")
    else:
        notifier = NullNotificationAdapter()

    orchestrator = HarnessOrchestrator(
        llm=llm,
        fs=fs,
        sensor=sensor,
        run_repo=run_repo,
        run_id=run_id,
        notifier=notifier,
        language_profile=prompt_builder._profile,
        post_processor=post_processor,
        execution_validator=execution_validator,
        enable_build_validation=settings.enable_build_validation,
        enable_execution_validation=settings.enable_execution_validation,
        dry_run=settings.dry_run,
        max_self_heal=settings.max_self_heal_attempts,
        max_tasks=settings.max_tasks_per_run,
    )

    log.info(
        "container.built",
        run_id=run_id,
        provider=settings.ai_provider.value,
        model=settings.ai_model,
        language=settings.target_language.value,
        workspace=ws,
        dry_run=settings.dry_run,
    )
    return orchestrator, run_id
