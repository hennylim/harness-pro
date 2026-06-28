"""
entry_point_finder 단위 테스트.

실제 보고된 시나리오: LLM이 main.sh 대신 iptables_analyzer.sh 를 생성한 경우.
"""
from __future__ import annotations

import os
import stat
import textwrap

import pytest

from agent.infrastructure.validators.entry_point_finder import (
    find_bash_entry,
    find_python_entry,
    _score_bash_entry,
)
from pathlib import Path


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(text))


# ── Bash 진입점 탐색 ──────────────────────────────────────────────────────────

class TestFindBashEntry:

    def test_fixed_candidate_main_sh(self, tmp_path):
        """고정 후보 main.sh 가 있으면 바로 반환."""
        _write(str(tmp_path / "src/main.sh"), "#!/usr/bin/env bash\necho hi\n")
        result = find_bash_entry(str(tmp_path))
        assert result == str(tmp_path / "src/main.sh")

    def test_fixed_candidate_root_main_sh(self, tmp_path):
        """루트 main.sh 도 고정 후보에 포함."""
        _write(str(tmp_path / "main.sh"), "#!/usr/bin/env bash\necho hi\n")
        result = find_bash_entry(str(tmp_path))
        assert result == str(tmp_path / "main.sh")

    def test_dynamic_name_contains_main(self, tmp_path):
        """파일명에 'main' 이 포함된 경우 동적 탐색으로 발견."""
        # LLM 이 main.sh 대신 iptables_main.sh 생성한 케이스
        _write(str(tmp_path / "src/iptables_main.sh"),
               "#!/usr/bin/env bash\nset -euo pipefail\nmain() { echo ok; }\nmain \"$@\"\n")
        result = find_bash_entry(str(tmp_path))
        assert result is not None
        assert "iptables_main" in result

    def test_dynamic_score_match(self, tmp_path):
        """파일명에 main 없어도 shebang+main()호출 패턴으로 진입점 탐지."""
        # 실제 로그 케이스: iptables_analyzer.sh 가 진입점인 경우
        _write(str(tmp_path / "src/utils.sh"),
               "#!/usr/bin/env bash\n# utility functions\nis_valid_ip() { return 0; }\n")
        _write(str(tmp_path / "src/iptables_analyzer.sh"),
               "#!/usr/bin/env bash\nset -euo pipefail\n"
               "source \"$(dirname \"$0\")/utils.sh\"\n"
               "main() {\n    echo \"Analyzing...\"\n}\nmain \"$@\"\n")
        result = find_bash_entry(str(tmp_path))
        assert result is not None
        assert "iptables_analyzer" in result

    def test_single_file_fallback(self, tmp_path):
        """sh 파일이 1개면 그것을 반환."""
        _write(str(tmp_path / "src/tool.sh"),
               "#!/usr/bin/env bash\necho tool\n")
        result = find_bash_entry(str(tmp_path))
        assert result is not None
        assert "tool.sh" in result

    def test_no_sh_files_returns_none(self, tmp_path):
        """sh 파일이 없으면 None 반환."""
        _write(str(tmp_path / "src/README.md"), "# docs\n")
        result = find_bash_entry(str(tmp_path))
        assert result is None

    def test_extra_candidates_checked_first(self, tmp_path):
        """extra_candidates 가 고정 후보보다 먼저 확인된다."""
        _write(str(tmp_path / "src/analyzer.sh"),
               "#!/usr/bin/env bash\necho analyzer\n")
        _write(str(tmp_path / "src/main.sh"),
               "#!/usr/bin/env bash\necho main\n")
        result = find_bash_entry(str(tmp_path),
                                 candidates=["src/analyzer.sh"])
        assert "analyzer.sh" in result

    def test_backup_dir_excluded(self, tmp_path):
        """백업 디렉터리(.harness_backups) 의 파일은 후보에서 제외."""
        _write(str(tmp_path / ".harness_backups/main.sh.bak"),
               "#!/usr/bin/env bash\necho backup\n")
        _write(str(tmp_path / "src/tool.sh"),
               "#!/usr/bin/env bash\necho real\n")
        result = find_bash_entry(str(tmp_path))
        assert ".harness_backups" not in (result or "")


# ── Bash 진입점 점수 평가 ─────────────────────────────────────────────────────

class TestScoreBashEntry:
    def test_main_with_shebang_and_call_scores_high(self, tmp_path):
        p = tmp_path / "main.sh"
        p.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "main() { echo hi; }\nmain \"$@\"\n"
        )
        assert _score_bash_entry(p) >= 7

    def test_utility_library_scores_low(self, tmp_path):
        p = tmp_path / "utils.sh"
        p.write_text(
            "#!/usr/bin/env bash\n"
            "is_valid_ip() { return 0; }\n"
            "is_valid_port() { return 0; }\n"
        )
        # shebang 있으면 4점, 직접 실행 명령 없으면 추가점 없음
        score = _score_bash_entry(p)
        assert score <= 5

    def test_direct_commands_add_score(self, tmp_path):
        p = tmp_path / "run.sh"
        p.write_text(
            "#!/usr/bin/env bash\n"
            "echo 'Starting...'\n"
            "iptables -L\n"
        )
        assert _score_bash_entry(p) >= 6


# ── Python 진입점 탐색 ────────────────────────────────────────────────────────

class TestFindPythonEntry:
    def test_fixed_candidate_src_main(self, tmp_path):
        _write(str(tmp_path / "src/main.py"),
               '"""Main."""\nif __name__ == "__main__":\n    print("ok")\n')
        result = find_python_entry(str(tmp_path))
        assert result == str(tmp_path / "src/main.py")

    def test_dunder_main_detection(self, tmp_path):
        """파일명이 main.py 가 아니어도 __name__ == "__main__" 패턴으로 탐지."""
        _write(str(tmp_path / "src/analyzer.py"),
               '"""Analyzer."""\nif __name__ == "__main__":\n    print("ok")\n')
        result = find_python_entry(str(tmp_path))
        assert result is not None
        assert "analyzer.py" in result

    def test_excludes_test_files(self, tmp_path):
        """테스트 파일은 진입점 후보에서 제외."""
        _write(str(tmp_path / "test_main.py"),
               'if __name__ == "__main__":\n    pass\n')
        _write(str(tmp_path / "src/app.py"),
               '"""App."""\nx = 1\n')
        result = find_python_entry(str(tmp_path))
        # test_main.py 가 아닌 app.py 가 선택되어야 함
        assert result is None or "test_main" not in result

    def test_no_py_returns_none(self, tmp_path):
        result = find_python_entry(str(tmp_path))
        assert result is None


# ── Validator 통합: 동적 탐색 적용 ────────────────────────────────────────────

class TestBashValidatorDynamicEntry:
    """BashExecutionValidator 가 실제로 동적 탐색을 사용하는지 확인."""

    def test_finds_non_standard_entry_name(self, tmp_path):
        """main.sh 대신 iptables_analyzer.sh 가 진입점인 경우."""
        import shutil
        if not shutil.which("bash"):
            pytest.skip("bash not available")

        _write(str(tmp_path / "src/utils.sh"),
               "#!/usr/bin/env bash\nset -euo pipefail\n"
               "log() { echo \"[LOG] $*\"; }\n")
        _write(str(tmp_path / "src/iptables_analyzer.sh"),
               "#!/usr/bin/env bash\nset -euo pipefail\n"
               "# shellcheck source=src/utils.sh\n"
               "source \"$(dirname \"$0\")/utils.sh\"\n"
               "main() {\n"
               "    log 'Analysis complete'\n"
               "    echo 'DONE'\n"
               "}\n"
               "if [[ \"${1:-}\" == '--help' ]]; then\n"
               "    echo 'Usage: iptables_analyzer.sh [options]'\n"
               "    exit 0\n"
               "fi\n"
               "main \"$@\"\n")

        from agent.infrastructure.validators.bash_validator import BashExecutionValidator
        v = BashExecutionValidator(run_timeout=10, run_flags=["--help"])

        build = v.build(str(tmp_path), dry_run=False)
        assert build.passed, build.errors

        run = v.run_smoke_test(str(tmp_path), dry_run=False)
        assert run.passed, run.error_summary
