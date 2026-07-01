# Harness Agent

LLM 기반 코드 자동 생성 에이전트.  
자연어 요구사항 → 구조화 계획 → 코드 생성 → 린트 검증 → 자가 수정의 루프를 자동화합니다.

**지원 언어**: Python · Bash · C · C++  
**지원 LLM**: Ollama · OpenAI · Azure OpenAI · Gemini

---

## 아키텍처

```
harness-pro/
├── config/
│   ├── settings.py          # pydantic-settings 기반 설정 (TargetLanguage Enum 포함)
│   └── logging.py           # structlog 중앙 설정
│
├── agent/
│   ├── domain/
│   │   ├── entities.py          # Task, ProjectPlan, CodePatch, RunSummary …
│   │   ├── exceptions.py        # 도메인 예외 계층
│   │   ├── interfaces.py        # 포트 (ILLMAdapter, ISensorAdapter …)
│   │   ├── language_profile.py  # ★ ILanguageProfile 포트
│   │   └── profiles/
│   │       ├── __init__.py          # PROFILE_REGISTRY (언어 등록 테이블)
│   │       ├── python_profile.py
│   │       ├── bash_profile.py
│   │       ├── c_profile.py
│   │       └── cpp_profile.py
│   │
│   ├── usecase/
│   │   └── orchestrator.py      # 핵심 유스케이스 (자가치유 루프)
│   │
│   ├── infrastructure/
│   │   ├── llm/
│   │   │   ├── prompt_builder.py    # ★ ILanguageProfile → LLM 프롬프트 변환
│   │   │   ├── openai_adapter.py    # OpenAI 호환 어댑터
│   │   │   └── gemini_adapter.py    # Google Gemini 어댑터
│   │   ├── fs/
│   │   │   ├── local_adapter.py     # 로컬 파일시스템 (샌드박스·백업)
│   │   │   └── run_repository.py    # JSON 실행 이력 저장소
│   │   ├── sensors/
│   │   │   ├── lint_sensor.py       # Python: ruff / flake8
│   │   │   ├── bash_sensor.py       # Bash: shellcheck
│   │   │   ├── c_sensor.py          # C: gcc + clang-tidy
│   │   │   ├── cpp_sensor.py        # C++: g++ + clang-tidy
│   │   │   └── router.py            # ★ 확장자 → 센서 라우팅 테이블
│   │   └── notification/
│   │       └── slack_adapter.py
│   │
│   └── container.py             # 의존성 주입 팩토리
│
├── tests/
│   ├── unit/
│   │   ├── test_entities.py
│   │   ├── test_orchestrator.py
│   │   └── test_language_support.py  # 프로필·라우터·PromptBuilder 단위 테스트
│   └── integration/
│       ├── test_fs_adapter.py
│       └── test_language_sensors.py  # 실제 도구(shellcheck/gcc/g++)로 검증
└── main.py                      # CLI 진입점
```

★ 표시된 3개 파일이 언어 확장의 핵심 접점입니다.

---

## 빠른 시작

```bash
pip install -r requirements.txt
cp .env.example .env          # 설정 편집
python main.py                # Python (기본)
python main.py --language bash
python main.py --language c
python main.py --language cpp
```

---

## 언어별 시스템 도구 설치

| 언어   | 도구               | 설치 (Ubuntu)                         | 설치 (macOS)                          |
|--------|--------------------|---------------------------------------|---------------------------------------|
| Python | ruff               | `pip install ruff`                    | `pip install ruff`                    |
| Python | flake8 (fallback)  | `pip install flake8`                  | `pip install flake8`                  |
| Bash   | shellcheck         | `apt install shellcheck`              | `brew install shellcheck`             |
| C      | gcc                | `apt install build-essential`         | `xcode-select --install`              |
| C      | clang-tidy         | `apt install clang-tidy`              | `brew install llvm`                   |
| C++    | g++                | `apt install build-essential`         | `xcode-select --install`              |
| C++    | clang-tidy         | `apt install clang-tidy`              | `brew install llvm`                   |

---

## 사용법

### 기본 실행

```bash
# Python (기본)
python main.py

# Bash 스크립트 생성
python main.py --language bash --requirements "로그 디렉터리 백업 스크립트"

# C 라이브러리 생성
python main.py --language c --requirements "동적 배열 라이브러리 구현"

# C++ 클래스 생성
python main.py --language cpp --requirements "제네릭 Queue<T> 템플릿 클래스"

# 드라이런 (파일 미생성, 린트 시뮬레이션)
python main.py --language c --dry-run

# 콘솔 로그 포맷으로 출력
python main.py --language bash --log-format console

# 워크스페이스 디렉터리 지정
python main.py --language cpp --workspace ./my_project
```

### .env 설정

```bash
# 대상 언어
TARGET_LANGUAGE=python    # python | bash | c | cpp

# Git repository 입력 (선택 사항)
# Git repo URL을 줄바꿈, 쉼표 또는 세미콜론으로 구분해 입력합니다.
# 지정된 repo는 WORKSPACE_DIR/repositories 아래에 복제됩니다.
# 분석 결과는 WORKSPACE_DIR/.repo_analysis 에 저장됩니다.
# GIT_REPOSITORY_URLS=https://github.com/example/repo1.git
# GIT_REPOSITORY_URLS=https://github.com/example/repo1.git;https://github.com/example/repo2.git

# LLM 제공자
AI_PROVIDER=OLLAMA        # OLLAMA | OPENAI | ANTHROPIC | AZURE | GEMINI
AI_MODEL=qwen2.5-coder:7b
AI_API_KEY=ollama

# Gemini 사용 시
# AI_PROVIDER=GEMINI
# GEMINI_API_KEY=AIza...
# AI_MODEL=gemini-2.5-flash-preview-05-20
```

---

## ★ 언어 확장 가이드 (Minimal & Rippable)

새 언어를 추가할 때 **기존 파일은 일절 수정하지 않습니다**.  
4개 파일만 작성/등록하면 끝납니다.

### Step 1 — 언어 프로필 작성

`agent/domain/profiles/rust_profile.py` 생성:

```python
from agent.domain.language_profile import ILanguageProfile

class RustProfile(ILanguageProfile):
    @property
    def name(self) -> str:
        return "Rust"

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".rs"})

    @property
    def code_style_rules(self) -> str:
        return (
            "Write idiomatic Rust (edition 2021).\n"
            "Use snake_case for variables/functions, PascalCase for types.\n"
            "Avoid unwrap() in library code; use Result<T, E>.\n"
            "Add /// doc comments on all public items."
        )

    @property
    def file_header_hint(self) -> str:
        return ""

    @property
    def plan_file_extension_hint(self) -> str:
        return ".rs"
```

### Step 2 — 센서 작성

`agent/infrastructure/sensors/rust_sensor.py` 생성:

```python
from agent.domain.entities import LintResult
from agent.domain.interfaces import ISensorAdapter
from agent.infrastructure.sensors.lint_sensor import _DRY_RUN_RESULT, run_subprocess_sensor

class CargoClippySensor(ISensorAdapter):
    def verify_code(self, absolute_path: str, dry_run: bool) -> LintResult:
        if dry_run:
            return _DRY_RUN_RESULT
        # 파일이 속한 크레이트 루트에서 clippy 실행
        import os
        crate_root = _find_cargo_toml(os.path.dirname(absolute_path))
        cmd = ["cargo", "clippy", "--manifest-path", f"{crate_root}/Cargo.toml",
               "--", "-D", "warnings"]
        return run_subprocess_sensor(cmd, "cargo-clippy", absolute_path)
```

### Step 3 — 레지스트리 등록

`agent/domain/profiles/__init__.py`의 `PROFILE_REGISTRY`에 추가:

```python
from agent.domain.profiles.rust_profile import RustProfile

PROFILE_REGISTRY: dict[str, ILanguageProfile] = {
    "python": PythonProfile(),
    "bash":   BashProfile(),
    "c":      CProfile(),
    "cpp":    CppProfile(),
    "rust":   RustProfile(),   # ← 추가
}
```

### Step 4 — 센서 라우터 등록

`agent/infrastructure/sensors/router.py`의 `EXTENSION_MAP`에 추가:

```python
from agent.infrastructure.sensors.rust_sensor import CargoClippySensor

EXTENSION_MAP: dict[str, ISensorAdapter] = {
    ...
    ".rs": CargoClippySensor(),   # ← 추가
}
```

### Step 5 — settings Enum 등록

`config/settings.py`의 `TargetLanguage`에 추가:

```python
class TargetLanguage(str, Enum):
    ...
    RUST = "rust"   # ← 추가
```

**이것으로 완료입니다.** Orchestrator · LLM 어댑터 · FS 어댑터는 전혀 수정하지 않아도 됩니다.

---

## LLM 제공자별 설정

| 제공자         | AI_PROVIDER | AI_MODEL                        | AI_BASE_URL                         |
|----------------|-------------|---------------------------------|-------------------------------------|
| Ollama (로컬)  | OLLAMA      | qwen2.5-coder:7b                | http://localhost:11434/v1           |
| OpenAI         | OPENAI      | gpt-4o                          | https://api.openai.com/v1           |
| Gemini         | GEMINI      | gemini-2.5-flash-preview-05-20  | (자동 설정)                          |
| Azure OpenAI   | AZURE       | gpt-4                           | https://<resource>.openai.azure.com/… |

---

## 테스트 실행

```bash
pytest                          # 전체 (단위 + 통합, coverage 포함)
pytest tests/unit               # 단위 테스트만
pytest tests/integration        # 통합 테스트만 (도구 필요)
pytest -v --no-cov              # 상세 출력
```

---

## 실행 이력

각 실행 후 `runs/<run_id>.json`이 생성됩니다:

```json
{
  "run_id": "a1b2c3d4",
  "status": "completed",
  "total_tasks": 3,
  "completed_tasks": 3,
  "failed_tasks": 0,
  "duration_seconds": 42.1
}
```
