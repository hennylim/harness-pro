"""
ILanguageProfile – 언어별 코드 생성 규칙을 캡슐화하는 포트.

설계 원칙
---------
* 도메인 레이어에 위치한다 (순수 추상; 외부 의존 없음).
* 새 언어를 추가할 때 이 인터페이스를 구현하는 클래스 1개만 작성한다.
* Orchestrator / LLM 어댑터 / Sensor 코드는 전혀 수정하지 않는다.

확장 시 해야 할 일 (Minimal & Rippable)
----------------------------------------
1. ``agent/domain/profiles/<lang>.py``  – ILanguageProfile 구현체
2. ``agent/infrastructure/sensors/<lang>_sensor.py``  – ISensorAdapter 구현체
3. ``agent/infrastructure/sensors/router.py``  – EXTENSION_MAP 에 항목 추가
4. ``config/settings.py``  – TargetLanguage Enum 에 값 추가
그 외 파일은 건드리지 않는다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ILanguageProfile(ABC):
    """
    언어 하나를 설명하는 불변 프로필.

    LLM 어댑터가 이 인터페이스를 통해 언어별 프롬프트 지침과
    파일 확장자 목록을 얻어 사용합니다.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable 언어 이름 (예: "Python", "C++")."""

    @property
    @abstractmethod
    def extensions(self) -> frozenset[str]:
        """
        이 언어에 속하는 파일 확장자 집합 (점 포함, 소문자).
        예: frozenset({".py"}), frozenset({".c", ".h"})
        """

    # ── LLM prompt fragments ──────────────────────────────────────────────────

    @property
    @abstractmethod
    def code_style_rules(self) -> str:
        """
        코드 생성 system prompt 에 주입될 언어별 스타일 규칙 블록.

        예 (Python):
            "Follow PEP-8. Lines ≤ 79 chars. Add docstrings."
        예 (C):
            "C17 standard. Use snake_case. No dynamic allocation without free()."
        """

    @property
    @abstractmethod
    def file_header_hint(self) -> str:
        """
        생성 파일 최상단에 넣을 힌트 (주석 형식, 라이선스 헤더 등).
        빈 문자열이면 주입하지 않는다.
        """

    @property
    @abstractmethod
    def plan_file_extension_hint(self) -> str:
        """
        LLM 플래너에게 전달할 "이 언어의 기본 파일 확장자" 힌트.
        예: ".py", ".c / .h", ".sh"
        """

    # ── Build / verify hints ──────────────────────────────────────────────────

    @property
    def requires_compilation(self) -> bool:
        """
        True 이면 센서가 컴파일 단계를 수행한다.
        기본값 False – 스크립트/인터프리터 언어.
        """
        return False

    @property
    def skeleton_extensions(self) -> frozenset[str]:
        """
        워크스페이스 스켈레톤에 포함할 확장자.
        기본값은 ``extensions`` 와 동일.
        헤더 파일 등 추가 확장자가 있으면 오버라이드.
        """
        return self.extensions

    # ── Build / execution hooks ───────────────────────────────────────────────

    @property
    def entry_point_pattern(self) -> str:
        """
        워크스페이스에서 진입점 파일을 찾는 glob 패턴.
        ExecutionValidator 가 실행할 파일을 결정할 때 사용한다.
        예: "src/main.py", "src/main.sh", "src/main"
        빈 문자열이면 자동 탐색하지 않는다.
        """
        return ""

    @property
    def build_timeout_seconds(self) -> int:
        """빌드 단계 타임아웃 (초). 기본 60초."""
        return 60

    @property
    def run_timeout_seconds(self) -> int:
        """실행 단계 타임아웃 (초). 기본 30초."""
        return 30

    @property
    def run_success_criteria(self) -> str:
        """
        실행 성공 판정 기준 설명.
        기본값: "exit code 0"
        ExecutionValidator 구현체가 이 기준을 따른다.
        """
        return "exit code 0"
