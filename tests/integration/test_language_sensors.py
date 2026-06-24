"""
언어별 센서 통합 테스트.

실제 lint 도구(shellcheck, gcc, g++, ruff)를 호출하므로
해당 도구가 없는 환경에서는 pytest.skip 으로 건너뜁니다.
"""
from __future__ import annotations

import os
import shutil
import stat
import textwrap

import pytest

from agent.domain.entities import LintResult
from agent.infrastructure.sensors.bash_sensor import ShellCheckSensorAdapter
from agent.infrastructure.sensors.c_sensor import CSensorAdapter
from agent.infrastructure.sensors.cpp_sensor import CppSensorAdapter
from agent.infrastructure.sensors.lint_sensor import SmartLintSensor
from agent.infrastructure.sensors.router import LanguageSensorRouter

# ── 공통 헬퍼 ─────────────────────────────────────────────────────────────────

def _write(path, text: str) -> str:
    """tmp_path 하위에 파일을 쓰고 절대 경로를 반환."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return str(path)


# ── Python (ruff/flake8) ──────────────────────────────────────────────────────

class TestPythonSensor:
    @pytest.fixture(autouse=True)
    def _require_ruff_or_flake8(self):
        if not (shutil.which("ruff") or shutil.which("flake8")):
            pytest.skip("ruff / flake8 not installed")

    def test_valid_python_passes(self, tmp_path):
        f = _write(tmp_path / "ok.py", """\
            \"\"\"Module docstring.\"\"\"


            def add(a: int, b: int) -> int:
                \"\"\"Add two numbers.\"\"\"
                return a + b
        """)
        result = SmartLintSensor().verify_code(f, dry_run=False)
        assert result.passed, result.errors

    def test_invalid_python_fails(self, tmp_path):
        f = _write(tmp_path / "bad.py", "def foo(\n")
        result = SmartLintSensor().verify_code(f, dry_run=False)
        assert not result.passed

    def test_dry_run_always_passes(self, tmp_path):
        f = _write(tmp_path / "any.py", "def foo(\n")   # 문법 오류
        result = SmartLintSensor().verify_code(f, dry_run=True)
        assert result.passed


# ── Bash (shellcheck) ─────────────────────────────────────────────────────────

class TestBashSensor:
    @pytest.fixture(autouse=True)
    def _require_shellcheck(self):
        if not shutil.which("shellcheck"):
            pytest.skip("shellcheck not installed")

    def test_valid_bash_passes(self, tmp_path):
        f = _write(tmp_path / "ok.sh", """\
            #!/usr/bin/env bash
            set -euo pipefail
            main() {
                echo "hello world"
            }
            main "$@"
        """)
        os.chmod(f, os.stat(f).st_mode | stat.S_IXUSR)
        result = ShellCheckSensorAdapter().verify_code(f, dry_run=False)
        assert result.passed, result.errors

    def test_local_outside_function_fails(self, tmp_path):
        # SC2168: local is only valid inside functions – shellcheck error
        f = _write(tmp_path / "bad.sh", """\
            #!/usr/bin/env bash
            set -euo pipefail
            local myvar="hello"
            echo "$myvar"
        """)
        result = ShellCheckSensorAdapter().verify_code(f, dry_run=False)
        assert not result.passed
    def test_dry_run_always_passes(self, tmp_path):
        f = _write(tmp_path / "any.sh", "#!/usr/bin/env bash\necho $UNQUOTED\n")
        result = ShellCheckSensorAdapter().verify_code(f, dry_run=True)
        assert result.passed


# ── C (gcc) ───────────────────────────────────────────────────────────────────

class TestCSensor:
    @pytest.fixture(autouse=True)
    def _require_gcc(self):
        if not (shutil.which("gcc") or shutil.which("cc")):
            pytest.skip("gcc/cc not installed")

    def test_valid_c_passes(self, tmp_path):
        f = _write(tmp_path / "ok.c", """\
            /**
             * \\file ok.c
             * \\brief Simple valid C file.
             */
            #include <stdio.h>

            int main(void) {
                printf("hello\\n");
                return 0;
            }
        """)
        # use_tidy=False: clang-tidy 없는 환경에서도 통과
        result = CSensorAdapter(use_tidy=False).verify_code(f, dry_run=False)
        assert result.passed, result.errors

    def test_syntax_error_fails(self, tmp_path):
        f = _write(tmp_path / "bad.c", """\
            int main(void) {
                int x = ;
                return 0;
            }
        """)
        result = CSensorAdapter(use_tidy=False).verify_code(f, dry_run=False)
        assert not result.passed

    def test_header_file_passes(self, tmp_path):
        f = _write(tmp_path / "mylib.h", """\
            #ifndef MYLIB_H
            #define MYLIB_H
            int add(int a, int b);
            #endif
        """)
        result = CSensorAdapter(use_tidy=False).verify_code(f, dry_run=False)
        assert result.passed, result.errors

    def test_dry_run_always_passes(self, tmp_path):
        f = _write(tmp_path / "any.c", "int main(void){ int x = ; }\n")
        result = CSensorAdapter(use_tidy=False).verify_code(f, dry_run=True)
        assert result.passed


# ── C++ (g++) ─────────────────────────────────────────────────────────────────

class TestCppSensor:
    @pytest.fixture(autouse=True)
    def _require_gpp(self):
        if not (shutil.which("g++") or shutil.which("c++")):
            pytest.skip("g++/c++ not installed")

    def test_valid_cpp_passes(self, tmp_path):
        f = _write(tmp_path / "ok.cpp", """\
            /**
             * \\file ok.cpp
             * \\brief Simple valid C++ file.
             */
            #include <iostream>

            int main() {
                std::cout << "hello" << std::endl;
                return 0;
            }
        """)
        result = CppSensorAdapter(use_tidy=False).verify_code(f, dry_run=False)
        assert result.passed, result.errors

    def test_syntax_error_fails(self, tmp_path):
        f = _write(tmp_path / "bad.cpp", """\
            int main() {
                int x = ;
                return 0;
            }
        """)
        result = CppSensorAdapter(use_tidy=False).verify_code(f, dry_run=False)
        assert not result.passed

    def test_header_file_passes(self, tmp_path):
        f = _write(tmp_path / "stack.hpp", """\
            #pragma once
            #include <vector>

            template <typename T>
            class Stack {
            public:
                void push(const T& val) { data_.push_back(val); }
                void pop()              { data_.pop_back(); }
                const T& top() const   { return data_.back(); }
                bool empty() const     { return data_.empty(); }
            private:
                std::vector<T> data_;
            };
        """)
        result = CppSensorAdapter(use_tidy=False).verify_code(f, dry_run=False)
        assert result.passed, result.errors

    def test_dry_run_always_passes(self, tmp_path):
        f = _write(tmp_path / "any.cpp", "int main(){ int x = ; }\n")
        result = CppSensorAdapter(use_tidy=False).verify_code(f, dry_run=True)
        assert result.passed


# ── LanguageSensorRouter ──────────────────────────────────────────────────────

class TestLanguageSensorRouterIntegration:
    def test_router_dispatches_python(self, tmp_path):
        if not (shutil.which("ruff") or shutil.which("flake8")):
            pytest.skip("no python linter")
        f = _write(tmp_path / "mod.py", '"""ok."""\n\nx = 1\n')
        result = LanguageSensorRouter().verify_code(f, dry_run=False)
        assert result.passed

    def test_router_dispatches_bash(self, tmp_path):
        if not shutil.which("shellcheck"):
            pytest.skip("shellcheck not installed")
        f = _write(tmp_path / "run.sh",
                   "#!/usr/bin/env bash\nset -euo pipefail\necho hi\n")
        result = LanguageSensorRouter().verify_code(f, dry_run=False)
        assert result.passed

    def test_router_dispatches_c(self, tmp_path):
        if not (shutil.which("gcc") or shutil.which("cc")):
            pytest.skip("gcc not installed")
        f = _write(tmp_path / "hello.c",
                   "#include <stdio.h>\nint main(void){return 0;}\n")
        result = LanguageSensorRouter().verify_code(f, dry_run=False)
        assert result.passed

    def test_router_dispatches_cpp(self, tmp_path):
        if not (shutil.which("g++") or shutil.which("c++")):
            pytest.skip("g++ not installed")
        f = _write(tmp_path / "hello.cpp",
                   "#include <iostream>\nint main(){return 0;}\n")
        result = LanguageSensorRouter().verify_code(f, dry_run=False)
        assert result.passed
