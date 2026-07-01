"""
Application configuration via environment variables.
All settings are validated at startup; missing required values fail fast.
"""
import re
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


class TargetLanguage(str, Enum):
    """
    코드 생성 대상 언어.

    새 언어 추가 시 이 Enum 에 값을 추가하고,
    agent/domain/profiles/ 와 agent/infrastructure/sensors/ 에 각각
    프로필 / 센서 파일을 작성한 뒤 registry / router 에 등록한다.
    """
    PYTHON = "python"
    BASH   = "bash"
    C      = "c"
    CPP    = "cpp"


# 제공자별 기본 모델 (자주 변경되므로 상수로 분리)
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"


class JsonMode(str, Enum):
    """
    OpenAI 호환 서버의 JSON 출력 모드.
    AUTO: 첫 호출 시 서버를 탐색해 자동 결정 (권장).
    json_object : OpenAI 표준. Ollama 최신 버전 지원.
    json_schema : LM Studio, 일부 로컬 서버.
    plain_prompt: response_format 없이 프롬프트만으로 강제.
    """
    AUTO         = "auto"
    JSON_OBJECT  = "json_object"
    JSON_SCHEMA  = "json_schema"
    PLAIN_PROMPT = "plain_prompt"


class LintBackend(str, Enum):
    FLAKE8 = "flake8"
    RUFF   = "ruff"


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
    ai_max_tokens: int = Field(default=8192, ge=256, le=32768)
    ai_timeout_seconds: int = Field(default=120, ge=10)
    ai_max_retries: int = Field(default=3, ge=1, le=10)

    # ── OpenAI 호환 서버 JSON 모드 ──────────────────────────────────────────────
    # AUTO: 첫 호출 시 서버 능력을 탐색해 자동 결정 (권장)
    # 수동 지정: json_object | json_schema | plain_prompt
    ai_json_mode: JsonMode = JsonMode.AUTO

    # ── Gemini 전용 ───────────────────────────────────────────────────────────
    gemini_api_key: Optional[str] = None

    # ── 대상 언어 ─────────────────────────────────────────────────────────────
    target_language: TargetLanguage = TargetLanguage.PYTHON

    # ── Agent behaviour ───────────────────────────────────────────────────────
    dry_run: bool = True
    workspace_dir: str = "./sandbox_workspace"
    git_repository_urls: Optional[str] = None
    git_repositories_dir: str = "repositories"
    repo_analysis_dir: str = ".repo_analysis"
    max_self_heal_attempts: int = Field(default=3, ge=1, le=10)
    max_tasks_per_run: int = Field(default=50, ge=1)

    # ── Adaptive chunking ────────────────────────────────────────────────────
    # 이 줄 수 이상으로 예상되는 파일은 섹션별 분할 생성
    chunk_threshold_lines: int = Field(default=80, ge=20, le=500)
    # 빌드 실패 시 자동 수정 재시도 횟수
    build_fix_retries: int = Field(default=2, ge=0, le=5)

    # ── Build & execution validation ─────────────────────────────────────────
    # 모든 태스크 완료 후 빌드 및 실행 검증을 수행하는 옵션
    enable_build_validation: bool = True    # 빌드(컴파일/문법검사) 수행
    enable_execution_validation: bool = True  # 실행(smoke-test) 수행
    build_timeout_seconds: int = Field(default=60, ge=10)
    run_timeout_seconds: int = Field(default=30, ge=5)

    # ── Code post-processing ──────────────────────────────────────────────────
    # 생성된 코드를 디스크 저장 전에 자동 수정하는 옵션
    post_process_trailing_whitespace: bool = True   # W291/W293 자동 제거
    post_process_unused_imports: bool = True        # F401 자동 제거
    post_process_detect_truncation: bool = True     # 코드 잘림 감지 후 재시도

    # ── Code quality (Python 전용 – 다른 언어는 센서가 자체 설정 보유) ────────
    lint_backend: LintBackend = LintBackend.RUFF
    lint_ignore_codes: str = "E501,W292,W391"

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"
    run_id: Optional[str] = None

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
        if v not in {"json", "console"}:
            raise ValueError("log_format must be 'json' or 'console'")
        return v

    def effective_gemini_api_key(self) -> str:
        key = self.gemini_api_key or (
            self.ai_api_key if self.ai_api_key not in ("ollama", "") else None
        )
        if not key:
            raise ValueError(
                "Gemini 사용 시 GEMINI_API_KEY 또는 AI_API_KEY에 "
                "Google AI Studio API 키를 설정해야 합니다."
            )
        return key

    @property
    def git_repository_list(self) -> list[str]:
        if not self.git_repository_urls:
            return []
        return [
            url.strip()
            for url in re.split(r"[\n,;]+", self.git_repository_urls)
            if url.strip()
        ]


settings = Settings()
