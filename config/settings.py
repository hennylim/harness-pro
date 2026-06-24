"""
Application configuration via environment variables.
All settings are validated at startup; missing required values fail fast.
"""
from enum import Enum
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AIProvider(str, Enum):
    OLLAMA = "OLLAMA"
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    AZURE = "AZURE"
    GEMINI = "GEMINI"


# Gemini 기본 모델 (모델명이 자주 갱신되므로 상수로 분리)
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"


class LintBackend(str, Enum):
    FLAKE8 = "flake8"
    RUFF = "ruff"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (공통) ────────────────────────────────────────────────────────────
    ai_provider: AIProvider = AIProvider.OLLAMA
    ai_model: str = "qwen2.5-coder:7b"
    ai_base_url: str = "http://localhost:11434/v1"
    ai_api_key: str = "ollama"
    ai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    ai_max_tokens: int = Field(default=4096, ge=256, le=32768)
    ai_timeout_seconds: int = Field(default=120, ge=10)
    ai_max_retries: int = Field(default=3, ge=1, le=10)

    # ── Gemini 전용 ───────────────────────────────────────────────────────────
    # AI_PROVIDER=GEMINI 일 때 AI_API_KEY 대신 이 값을 사용해도 됩니다.
    # 둘 다 설정된 경우 GEMINI_API_KEY 가 우선합니다.
    gemini_api_key: Optional[str] = None

    # ── Agent behaviour ───────────────────────────────────────────────────────
    dry_run: bool = True
    workspace_dir: str = "./sandbox_workspace"
    max_self_heal_attempts: int = Field(default=3, ge=1, le=10)
    max_tasks_per_run: int = Field(default=50, ge=1)

    # ── Code quality ──────────────────────────────────────────────────────────
    lint_backend: LintBackend = LintBackend.RUFF
    lint_ignore_codes: str = "E501,W292,W391"

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"          # "json" | "console"
    run_id: Optional[str] = None      # injected at runtime if not set

    # ── Notifications (optional) ──────────────────────────────────────────────
    slack_webhook_url: Optional[str] = None
    notify_on_failure: bool = True
    notify_on_success: bool = False

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, v: str) -> str:
        allowed = {"json", "console"}
        if v not in allowed:
            raise ValueError(f"log_format must be one of {allowed}")
        return v

    def effective_gemini_api_key(self) -> str:
        """
        Gemini API 키를 반환합니다.
        우선순위: GEMINI_API_KEY > AI_API_KEY
        키가 없으면 ValueError를 발생시켜 시작 전에 실패합니다.
        """
        key = self.gemini_api_key or (
            self.ai_api_key if self.ai_api_key not in ("ollama", "") else None
        )
        if not key:
            raise ValueError(
                "Gemini 사용 시 GEMINI_API_KEY 또는 AI_API_KEY에 "
                "Google AI Studio API 키를 설정해야 합니다."
            )
        return key


# Singleton – imported everywhere else
settings = Settings()
