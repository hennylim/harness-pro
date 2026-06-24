# Harness Agent

LLM 기반 코드 자동 생성 에이전트. 자연어 요구사항을 받아 완성된 Python 소스코드를 생성하고, 린트 검증 + 자가 수정 루프로 품질을 보장합니다.

## 주요 개선 사항 (원본 대비)

| 항목 | 원본 | 고도화 버전 |
|---|---|---|
| 설정 관리 | 하드코딩 + dotenv | `pydantic-settings` 기반 타입 검증 |
| 에러 처리 | 없음 | 도메인 예외 계층 + 전파 체계 |
| LLM 재시도 | 없음 | `tenacity` 지수 백오프 (타임아웃/레이트리밋 구분) |
| 보안 | 없음 | 워크스페이스 경로 탈출 방지 (sandbox) |
| 파일 백업 | 없음 | 타임스탬프 백업 자동 생성 |
| 로깅 | print() | structlog JSON/Console 구조화 로그 |
| 관찰가능성 | 없음 | run_id 추적, 소요시간, 태스크별 상태 |
| 린터 | flake8 전용 | `SmartLintSensor` (ruff ▶ flake8 자동 선택) |
| 알림 | 없음 | Slack Webhook 알림 어댑터 |
| 실행 이력 | 없음 | JSON 파일로 RunSummary 영속화 |
| 테스트 | 없음 | pytest 단위/통합 테스트 + coverage |
| CLI | 없음 | `--requirements`, `--dry-run`, `--workspace` 옵션 |
| Docker | 단순 | 멀티스테이지 빌드 + non-root 사용자 |
| DI | 없음 | `container.py` 의존성 주입 팩토리 |

## 아키텍처

```
harness-pro/
├── config/
│   ├── settings.py          # pydantic-settings 기반 설정 (환경변수/dotenv)
│   └── logging.py           # structlog 중앙 설정
├── agent/
│   ├── domain/
│   │   ├── entities.py      # Task, ProjectPlan, CodePatch, RunSummary …
│   │   ├── exceptions.py    # 도메인 예외 계층
│   │   └── interfaces.py    # 포트 (추상 인터페이스)
│   ├── usecase/
│   │   └── orchestrator.py  # 핵심 유스케이스 (자가치유 루프)
│   ├── infrastructure/
│   │   ├── llm/
│   │   │   └── openai_adapter.py   # OpenAI 호환 LLM (재시도, 타임아웃)
│   │   ├── fs/
│   │   │   ├── local_adapter.py    # 로컬 파일시스템 (샌드박스, 백업)
│   │   │   └── run_repository.py   # JSON 실행 이력 저장소
│   │   ├── sensors/
│   │   │   └── lint_sensor.py      # SmartLintSensor (ruff/flake8)
│   │   └── notification/
│   │       └── slack_adapter.py    # Slack Webhook 알림
│   └── container.py         # 의존성 주입 팩토리
├── tests/
│   ├── unit/                # 단위 테스트 (mock 기반)
│   └── integration/         # 통합 테스트 (tmp_path 기반)
└── main.py                  # CLI 진입점 (rich 출력)
```

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 설정

```bash
cp .env.example .env
# .env 파일을 열어 AI_MODEL, AI_BASE_URL 등을 설정
```

### 3. 실행 (기본 – 드라이런)

```bash
python main.py
```

### 4. 실제 코드 생성

```bash
# .env 에서 DRY_RUN=False 로 변경하거나:
DRY_RUN=False python main.py --requirements "FastAPI로 간단한 TODO API를 만들어줘"
```

### 5. CLI 옵션

```bash
python main.py --help

# 로그를 콘솔 형식으로
python main.py --log-format console

# 드라이런 강제 적용
python main.py --dry-run

# 워크스페이스 오버라이드
python main.py --workspace ./my_project
```

## 테스트 실행

```bash
pytest                      # 전체 (coverage 포함)
pytest tests/unit           # 단위 테스트만
pytest tests/integration    # 통합 테스트만
pytest -v --no-cov          # 상세 출력, coverage 없음
```

## Docker

```bash
# 이미지 빌드
docker build -t harness-agent .

# 실행 (드라이런)
docker run --env-file .env harness-agent

# Docker Compose (Ollama 포함)
docker compose --profile ollama up
```

## 알림 설정 (Slack)

`.env` 에 추가:

```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
NOTIFY_ON_FAILURE=True
NOTIFY_ON_SUCCESS=True
```

## 다른 LLM 제공자 사용

| 제공자 | `AI_BASE_URL` | `AI_MODEL` | `AI_API_KEY` |
|---|---|---|---|
| Ollama (로컬) | `http://localhost:11434/v1` | `qwen2.5-coder:7b` | `ollama` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` | `sk-...` |
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/deployments/<deploy>/` | `gpt-4` | `<azure-key>` |

## 실행 이력

각 실행 후 `runs/<run_id>.json` 파일이 생성됩니다:

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
