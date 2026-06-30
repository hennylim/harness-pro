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
            "Do NOT use deprecated POSIX functions.\n"
            "ONLY use C standard library headers (stdio.h, stdlib.h, string.h, "
            "stddef.h, stdint.h, stdbool.h, math.h, time.h, errno.h, ctype.h, "
            "assert.h, limits.h) and POSIX headers available without extra "
            "install on a minimal Linux system (unistd.h, sys/socket.h, "
            "netinet/in.h, arpa/inet.h, pthread.h, sys/types.h, fcntl.h).\n"
            "NEVER include third-party library headers that require separate "
            "package installation (e.g. curl/curl.h, openssl/*.h, jansson.h, "
            "cjson/cJSON.h, sqlite3.h). If HTTP/JSON/SSL functionality is "
            "needed, implement it manually using POSIX sockets "
            "(sys/socket.h) and write a minimal hand-rolled parser, "
            "since the build environment has no external libraries installed."
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
    def header_extensions(self) -> frozenset[str]:
        """헤더 파일은 전체 내용을 skeleton에 포함 (타입 선언 일관성 유지)."""
        return frozenset({".h"})

    @property
    def requires_compilation(self) -> bool:
        return True
