"""C 언어 프로필."""
from __future__ import annotations

from agent.domain.language_profile import ILanguageProfile


class CProfile(ILanguageProfile):
    """C17 코드 생성 규칙."""

    @property
    def name(self) -> str:
        return "C"

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".c", ".h"})

    @property
    def skeleton_extensions(self) -> frozenset[str]:
        # 헤더 파일도 스켈레톤에 포함한다
        return frozenset({".c", ".h"})

    @property
    def code_style_rules(self) -> str:
        return (
            "Write C17-standard code.\n"
            "Use snake_case for all identifiers.\n"
            "Every .h file must have an include guard: "
            "#ifndef FOO_H / #define FOO_H / ... / #endif.\n"
            "Every malloc/calloc must have a matching free(); "
            "never leak memory.\n"
            "Return -1 / NULL on error; document return values in comments.\n"
            "Add a file-level Doxygen comment block (\\file, \\brief).\n"
            "Add \\brief Doxygen comments above each function declaration.\n"
            "Lines ≤ 79 characters.\n"
            "Do NOT use deprecated POSIX functions."
        )

    @property
    def file_header_hint(self) -> str:
        return (
            "/**\n"
            " * \\file <filename>\n"
            " * \\brief <one-line description>\n"
            " */"
        )

    @property
    def plan_file_extension_hint(self) -> str:
        return ".c / .h"

    @property
    def requires_compilation(self) -> bool:
        return True
