"""
Dependency injection container.

``build_orchestrator()`` 가 settings 를 읽어 모든 어댑터를 조립합니다.
구체 클래스 인스턴스화는 이 파일에서만 일어납니다.

AI_PROVIDER 별 LLM 어댑터 선택
-------------------------------
OLLAMA / OPENAI / ANTHROPIC / AZURE  →  OpenAICompatibleAdapter
GEMINI                                →  GeminiAdapter (google-genai SDK)
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

import structlog

from agent.domain.interfaces import ILLMAdapter, INotificationAdapter
from agent.infrastructure.fs.local_adapter import LocalFileSystemAdapter
from agent.infrastructure.fs.run_repository import JsonRunRepository
from agent.infrastructure.llm.gemini_adapter import GeminiAdapter
from agent.infrastructure.llm.openai_adapter import OpenAICompatibleAdapter
from agent.infrastructure.notification.slack_adapter import (
    NullNotificationAdapter,
    SlackWebhookAdapter,
)
from agent.infrastructure.sensors.lint_sensor import SmartLintSensor
from agent.usecase.orchestrator import HarnessOrchestrator
from config.settings import AIProvider, GEMINI_DEFAULT_MODEL, settings

log = structlog.get_logger(__name__)


def _build_llm_adapter() -> ILLMAdapter:
    """AI_PROVIDER 값에 따라 적절한 LLM 어댑터를 반환합니다."""
    provider = settings.ai_provider

    if provider == AIProvider.GEMINI:
        api_key = settings.effective_gemini_api_key()
        model = settings.ai_model if settings.ai_model != "qwen2.5-coder:7b" \
            else GEMINI_DEFAULT_MODEL
        log.info("llm.adapter.selected", provider="gemini", model=model)
        return GeminiAdapter(
            api_key=api_key,
            model=model,
            temperature=settings.ai_temperature,
            max_tokens=settings.ai_max_tokens,
            max_retries=settings.ai_max_retries,
        )

    # OpenAI / Ollama / Azure / Anthropic (OpenAI 호환)
    log.info(
        "llm.adapter.selected",
        provider=provider.value,
        model=settings.ai_model,
        base_url=settings.ai_base_url,
    )
    return OpenAICompatibleAdapter(
        model=settings.ai_model,
        base_url=settings.ai_base_url,
        api_key=settings.ai_api_key,
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

    Parameters
    ----------
    workspace_dir:
        워크스페이스 디렉터리 오버라이드 (테스트용).
    run_id:
        실행 ID 오버라이드 (테스트 / CLI용).

    Returns
    -------
    tuple[HarnessOrchestrator, str]
        준비된 오케스트레이터와 활성 run_id.
    """
    run_id = run_id or settings.run_id or str(uuid.uuid4())[:8]

    # ── 워크스페이스 경로 ──────────────────────────────────────────────────────
    ws = workspace_dir or settings.workspace_dir
    if not os.path.isabs(ws):
        ws = os.path.join(os.getcwd(), ws)
    runs_dir = os.path.join(os.path.dirname(ws), "runs")

    # ── 어댑터 조립 ───────────────────────────────────────────────────────────
    llm = _build_llm_adapter()
    fs = LocalFileSystemAdapter(workspace_dir=ws)
    sensor = SmartLintSensor(ignore_codes=settings.lint_ignore_codes)
    run_repo = JsonRunRepository(runs_dir=runs_dir)

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
        dry_run=settings.dry_run,
        max_self_heal=settings.max_self_heal_attempts,
        max_tasks=settings.max_tasks_per_run,
    )

    log.info(
        "container.built",
        run_id=run_id,
        provider=settings.ai_provider.value,
        model=settings.ai_model,
        workspace=ws,
        dry_run=settings.dry_run,
    )
    return orchestrator, run_id
