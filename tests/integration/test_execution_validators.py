"""
ExecutionValidator 통합 테스트.

실제 python3 / bash / gcc / g++ 를 호출합니다.
각 도구가 없는 환경에서는 pytest.skip 으로 건너뜁니다.
"""
from __future__ import annotations

import os
import shutil
import textwrap
import stat

import pytest

from agent.infrastructure.validators.python_validator import PythonExecutionValidator
from agent.infrastructure.validators.bash_validator import BashExecutionValidator
from agent.infrastructure.validators.c_cpp_validator import (
    CExecutionValidator,
    CppExecutionValidator,
)


def _write(path, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(text))


# ── Python ────────────────────────────────────────────────────────────────────

class TestPythonValidator:
    @pytest.fixture(autouse=True)
    def _require_python(self):
        if not shutil.which("python3"):
            pytest.skip("python3 not available")

    def test_build_pass_valid_code(self, tmp_path):
        _write(str(tmp_path / "src/main.py"),
               '"""Main module."""\n\n\ndef main() -> None:\n    """Entry."""\n    print("hello")\n\n\nif __name__ == "__main__":\n    main()\n')
        result = PythonExecutionValidator().build(str(tmp_path), dry_run=False)
        assert result.passed, result.errors

    def test_build_fail_syntax_error(self, tmp_path):
        _write(str(tmp_path / "src/main.py"), "def foo(\n")
        result = PythonExecutionValidator().build(str(tmp_path), dry_run=False)
        assert not result.passed
        assert result.errors

    def test_run_pass_valid_entry(self, tmp_path):
        _write(str(tmp_path / "src/main.py"),
               '"""Main."""\n\nif __name__ == "__main__":\n    print("ok")\n')
        v = PythonExecutionValidator(run_timeout=10)
        build = v.build(str(tmp_path), dry_run=False)
        assert build.passed
        run = v.run_smoke_test(str(tmp_path), dry_run=False)
        assert run.passed, run.error_summary
        assert "ok" in run.stdout

    def test_run_fail_no_entry_point(self, tmp_path):
        # .py 파일이 아예 없는 경우 진입점을 찾을 수 없어야 한다
        _write(str(tmp_path / "src/README.md"), "# docs\n")
        v = PythonExecutionValidator()
        run = v.run_smoke_test(str(tmp_path), dry_run=False)
        assert not run.passed
        assert "진입점" in run.error_summary

    def test_run_fail_nonzero_exit(self, tmp_path):
        _write(str(tmp_path / "src/main.py"),
               'import sys\nsys.exit(1)\n')
        v = PythonExecutionValidator(run_timeout=10)
        v.build(str(tmp_path), dry_run=False)
        run = v.run_smoke_test(str(tmp_path), dry_run=False)
        assert not run.passed
        assert run.exit_code == 1

    def test_dry_run_always_passes(self, tmp_path):
        v = PythonExecutionValidator()
        assert v.build(str(tmp_path), dry_run=True).passed
        assert v.run_smoke_test(str(tmp_path), dry_run=True).passed


# ── Bash ──────────────────────────────────────────────────────────────────────

class TestBashValidator:
    @pytest.fixture(autouse=True)
    def _require_bash(self):
        if not shutil.which("bash"):
            pytest.skip("bash not available")

    def test_build_pass_valid_script(self, tmp_path):
        _write(str(tmp_path / "src/main.sh"),
               '#!/usr/bin/env bash\nset -euo pipefail\necho "hello"\n')
        result = BashExecutionValidator().build(str(tmp_path), dry_run=False)
        assert result.passed, result.errors

    def test_build_fail_syntax_error(self, tmp_path):
        _write(str(tmp_path / "src/main.sh"),
               '#!/usr/bin/env bash\nif [[\n')
        result = BashExecutionValidator().build(str(tmp_path), dry_run=False)
        assert not result.passed

    def test_run_pass_with_help_flag(self, tmp_path):
        script = (
            '#!/usr/bin/env bash\n'
            'set -euo pipefail\n'
            'if [[ "${1:-}" == "--help" ]]; then\n'
            '    echo "Usage: main.sh"\n'
            '    exit 0\n'
            'fi\n'
            'echo "running"\n'
        )
        _write(str(tmp_path / "src/main.sh"), script)
        os.chmod(str(tmp_path / "src/main.sh"),
                 os.stat(str(tmp_path / "src/main.sh")).st_mode | stat.S_IXUSR)
        v = BashExecutionValidator(run_timeout=10, run_flags=["--help"])
        v.build(str(tmp_path), dry_run=False)
        run = v.run_smoke_test(str(tmp_path), dry_run=False)
        assert run.passed, run.error_summary

    def test_run_fail_no_entry(self, tmp_path):
        # .sh 파일이 아예 없는 경우 진입점을 찾을 수 없어야 한다
        _write(str(tmp_path / "src/README.md"), "# docs\n")
        run = BashExecutionValidator().run_smoke_test(str(tmp_path), dry_run=False)
        assert not run.passed
        assert "진입점" in run.error_summary

    def test_dry_run_always_passes(self, tmp_path):
        v = BashExecutionValidator()
        assert v.build(str(tmp_path), dry_run=True).passed
        assert v.run_smoke_test(str(tmp_path), dry_run=True).passed


# ── C ─────────────────────────────────────────────────────────────────────────

class TestCValidator:
    @pytest.fixture(autouse=True)
    def _require_gcc(self):
        if not (shutil.which("gcc") or shutil.which("cc")):
            pytest.skip("gcc/cc not available")

    def test_build_and_run_pass(self, tmp_path):
        _write(str(tmp_path / "src/main.c"),
               '#include <stdio.h>\nint main(void) { printf("ok\\n"); return 0; }\n')
        v = CExecutionValidator(run_timeout=10)
        build = v.build(str(tmp_path), dry_run=False)
        assert build.passed, build.errors
        run = v.run_smoke_test(str(tmp_path), dry_run=False)
        assert run.passed, run.error_summary
        assert "ok" in run.stdout

    def test_build_fail_syntax_error(self, tmp_path):
        _write(str(tmp_path / "src/main.c"),
               'int main(void) { int x = ; return 0; }\n')
        result = CExecutionValidator().build(str(tmp_path), dry_run=False)
        assert not result.passed

    def test_run_fail_nonzero_exit(self, tmp_path):
        _write(str(tmp_path / "src/main.c"),
               'int main(void) { return 42; }\n')
        v = CExecutionValidator(run_timeout=10)
        v.build(str(tmp_path), dry_run=False)
        run = v.run_smoke_test(str(tmp_path), dry_run=False)
        assert not run.passed
        assert run.exit_code == 42

    def test_dry_run_always_passes(self, tmp_path):
        v = CExecutionValidator()
        assert v.build(str(tmp_path), dry_run=True).passed
        assert v.run_smoke_test(str(tmp_path), dry_run=True).passed


# ── C++ ───────────────────────────────────────────────────────────────────────

class TestCppValidator:
    @pytest.fixture(autouse=True)
    def _require_gpp(self):
        if not (shutil.which("g++") or shutil.which("c++")):
            pytest.skip("g++/c++ not available")

    def test_build_and_run_pass(self, tmp_path):
        _write(str(tmp_path / "src/main.cpp"),
               '#include <iostream>\nint main() { std::cout << "ok" << std::endl; return 0; }\n')
        v = CppExecutionValidator(run_timeout=10)
        build = v.build(str(tmp_path), dry_run=False)
        assert build.passed, build.errors
        run = v.run_smoke_test(str(tmp_path), dry_run=False)
        assert run.passed, run.error_summary
        assert "ok" in run.stdout

    def test_build_fail_syntax_error(self, tmp_path):
        _write(str(tmp_path / "src/main.cpp"),
               'int main() { int x = ; return 0; }\n')
        result = CppExecutionValidator().build(str(tmp_path), dry_run=False)
        assert not result.passed

    def test_dry_run_always_passes(self, tmp_path):
        v = CppExecutionValidator()
        assert v.build(str(tmp_path), dry_run=True).passed
        assert v.run_smoke_test(str(tmp_path), dry_run=True).passed
