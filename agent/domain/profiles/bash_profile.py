"""Bash 언어 프로필."""
from __future__ import annotations

from agent.domain.language_profile import ILanguageProfile


class BashProfile(ILanguageProfile):
    """Bash 셸 스크립트 코드 생성 규칙."""

    @property
    def name(self) -> str:
        return "Bash"

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".sh"})

    @property
    def code_style_rules(self) -> str:
        return (
            "Write POSIX-compatible Bash (bash 4+).\n"
            "Always start with '#!/usr/bin/env bash'.\n"
            "Set 'set -euo pipefail' immediately after the shebang.\n"
            "Use snake_case for variable and function names.\n"
            "Quote all variable expansions: \"$var\" not $var.\n"
            "Use [[ ]] for conditionals, not [ ].\n"
            "Add a brief comment block at the top describing the script.\n"
            "Functions must have a short comment explaining their purpose.\n"
            "Lines ≤ 79 characters where practical."
        )

    @property
    def file_header_hint(self) -> str:
        return "#!/usr/bin/env bash\nset -euo pipefail"

    @property
    def plan_file_extension_hint(self) -> str:
        return ".sh"
