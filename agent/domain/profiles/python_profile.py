"""Python 언어 프로필."""
from __future__ import annotations

from agent.domain.language_profile import ILanguageProfile


class PythonProfile(ILanguageProfile):
    """Python 3.x 코드 생성 규칙."""

    @property
    def name(self) -> str:
        return "Python"

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".py"})

    @property
    def code_style_rules(self) -> str:
        return (
            "Write Python 3.11+ code.\n"
            "Follow PEP-8; lines ≤ 79 characters.\n"
            "Add module-level and function-level docstrings.\n"
            "Include all necessary imports at the top.\n"
            "Do NOT use wildcard imports."
        )

    @property
    def file_header_hint(self) -> str:
        return ""

    @property
    def plan_file_extension_hint(self) -> str:
        return ".py"
