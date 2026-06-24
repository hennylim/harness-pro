"""C++ 언어 프로필."""
from __future__ import annotations

from agent.domain.language_profile import ILanguageProfile


class CppProfile(ILanguageProfile):
    """C++17 코드 생성 규칙."""

    @property
    def name(self) -> str:
        return "C++"

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".cpp", ".cc", ".cxx"})

    @property
    def skeleton_extensions(self) -> frozenset[str]:
        # 헤더 포함
        return frozenset({".cpp", ".cc", ".cxx", ".hpp", ".h"})

    @property
    def code_style_rules(self) -> str:
        return (
            "Write C++17-standard code.\n"
            "Prefer RAII: use smart pointers (std::unique_ptr, std::shared_ptr) "
            "over raw new/delete.\n"
            "Use snake_case for variables/functions, PascalCase for classes.\n"
            "Every .hpp file must have a #pragma once guard.\n"
            "Mark single-argument constructors explicit unless conversion is intended.\n"
            "Prefer const-correctness: mark methods const where applicable.\n"
            "Use nullptr instead of NULL or 0 for pointers.\n"
            "Add Doxygen \\brief comments above each class and public method.\n"
            "Lines ≤ 99 characters.\n"
            "Do NOT use raw arrays where std::array or std::vector suffice."
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
        return ".cpp / .hpp"

    @property
    def requires_compilation(self) -> bool:
        return True
