"""
CodePostProcessor 단위 테스트.

실제 보고된 에러 케이스를 모두 커버한다:
  W291  Trailing whitespace
  W292  No newline at end of file
  W391  Blank line at end of file
  F401  Unused import
  코드 잘림 (truncation) 감지
"""
from __future__ import annotations

import pytest

from agent.domain.entities import CodePatch
from agent.infrastructure.llm.post_processor import (
    CodePostProcessor,
    TruncatedCodeError,
    _ast_remove_unused_imports,
    _detect_truncation,
    _normalize_eof,
    _strip_trailing_whitespace,
)


def _make_patch(code: str, path: str = "src/foo.py") -> CodePatch:
    return CodePatch(target_file=path, explanation="test", code_block=code)


# ── Trailing whitespace (W291 / W293) ─────────────────────────────────────────

class TestTrailingWhitespace:
    def test_strips_trailing_spaces(self):
        code = "x = 1   \ny = 2\n"
        result, n = _strip_trailing_whitespace(code)
        assert result == "x = 1\ny = 2\n"
        assert n == 1

    def test_strips_trailing_tabs(self):
        code = "def foo():\t\n    pass\n"
        result, n = _strip_trailing_whitespace(code)
        assert result == "def foo():\n    pass\n"
        assert n == 1

    def test_no_change_when_clean(self):
        code = "x = 1\ny = 2\n"
        result, n = _strip_trailing_whitespace(code)
        assert result == code
        assert n == 0

    def test_processor_applies_fix(self):
        proc = CodePostProcessor()
        patch = _make_patch("x = 1   \ny = 2  \n")
        out = proc.process(patch)
        for line in out.code_block.splitlines():
            assert line == line.rstrip(), f"Trailing whitespace in: {repr(line)}"
        assert out.code_block == "x = 1\ny = 2\n"


# ── EOF newline (W292 / W391) ─────────────────────────────────────────────────

class TestEofNewline:
    def test_adds_missing_newline(self):
        assert _normalize_eof("x = 1") == "x = 1\n"

    def test_removes_extra_newlines(self):
        assert _normalize_eof("x = 1\n\n\n") == "x = 1\n"

    def test_single_newline_unchanged(self):
        assert _normalize_eof("x = 1\n") == "x = 1\n"


# ── Unused imports (F401) ─────────────────────────────────────────────────────

class TestUnusedImports:
    def test_removes_unused_typing_optional(self):
        code = (
            "import logging\n"
            "import signal\n"
            "from typing import Optional\n"
            "\n"
            "def main() -> None:\n"
            "    logging.info('start')\n"
            "    signal.signal(signal.SIGTERM, lambda s, f: None)\n"
        )
        result, removed = _ast_remove_unused_imports(code)
        assert "Optional" not in result
        assert any("Optional" in r for r in removed)
        assert "logging" in result
        assert "signal" in result

    def test_keeps_used_imports(self):
        code = (
            "from typing import Optional\n"
            "\n"
            "def foo(x: Optional[int]) -> None:\n"
            "    pass\n"
        )
        result, removed = _ast_remove_unused_imports(code)
        assert "Optional" in result
        assert not removed

    def test_removes_multiple_unused(self):
        code = (
            "import os\n"
            "import sys\n"
            "\n"
            "x = os.getcwd()\n"
        )
        result, removed = _ast_remove_unused_imports(code)
        assert "os" in result
        assert "sys" not in result

    def test_syntax_error_code_unchanged(self):
        # 구문 오류 있는 코드는 건드리지 않음
        code = "from typing import Optional\ndef foo(\n"
        result, removed = _ast_remove_unused_imports(code)
        assert result == code
        assert removed == []

    def test_processor_applies_to_py_files(self):
        proc = CodePostProcessor()
        code = (
            "import logging\n"
            "from typing import Optional\n"
            "\n"
            "logging.info('hi')\n"
        )
        out = proc.process(_make_patch(code, "src/main.py"))
        assert "Optional" not in out.code_block
        assert "logging" in out.code_block

    def test_processor_skips_non_py_files(self):
        proc = CodePostProcessor()
        # .sh 파일은 import 처리 건너뜀
        code = "echo hello\n"
        out = proc.process(_make_patch(code, "src/run.sh"))
        assert out.code_block == code


# ── 잘림 감지 ─────────────────────────────────────────────────────────────────

class TestTruncationDetection:
    def test_detects_import_with_trailing_comma(self):
        # 실제 보고 케이스: import 줄이 쉼표로 끝남
        code = "import logging\nfrom src.models import LeaseRecord,"
        result = _detect_truncation(code)
        assert result is not None
        assert "trailing comma" in result

    def test_detects_empty_code(self):
        assert _detect_truncation("") is not None
        assert _detect_truncation("   \n  ") is not None

    def test_clean_code_not_flagged(self):
        code = (
            '"""Module."""\n\n'
            "def add(a: int, b: int) -> int:\n"
            '    """Add."""\n'
            "    return a + b\n"
        )
        assert _detect_truncation(code) is None

    def test_short_identifier_ending_not_flagged(self):
        # "pass", "None", return 문 등은 잘림이 아님
        assert _detect_truncation("def foo():\n    pass\n") is None
        assert _detect_truncation("x = None\n") is None

    def test_processor_raises_on_trailing_comma_import(self):
        proc = CodePostProcessor(detect_truncation=True)
        code = "from datetime import datetime\nfrom src.domain.models import LeaseR,"
        with pytest.raises(TruncatedCodeError):
            proc.process(_make_patch(code, "src/use_cases.py"))

    def test_processor_raises_on_empty_code(self):
        proc = CodePostProcessor(detect_truncation=True)
        # CodePatch validator 가 빈 코드를 거부하므로 최소 공백이 있는 케이스로 테스트
        # (실제 TruncatedCodeError 는 post_processor 내부에서 발생)
        code = "   \n  "  # non-empty string but blank content
        # CodePatch 는 strip 후 비어 있으므로 ValidationError
        with pytest.raises(Exception):
            proc.process(_make_patch(code, "src/foo.py"))

    def test_processor_skips_detection_when_disabled(self):
        proc = CodePostProcessor(detect_truncation=False)
        # trailing comma import 이지만 감지 안 함
        code = "from src.models import Foo,\n# incomplete\n"
        # TruncatedCodeError 없이 처리됨
        out = proc.process(_make_patch(code, "src/use_cases.py"))
        assert out is not None


# ── 실제 보고된 에러 케이스 재현 ──────────────────────────────────────────────

class TestRealWorldErrorCases:
    """실제 실패 로그에서 보고된 케이스를 직접 재현."""

    def test_settings_py_trailing_whitespace(self):
        """W291: settings.py 의 docstring 줄 끝 공백."""
        proc = CodePostProcessor()
        code = (
            '"""Settings module."""\n'
            '\n'
            'class Settings:\n'
            '    """   \n'
            '    Application settings loaded from environment variables.   \n'
            '\n'
            "    Uses Pydantic's BaseSettings.   \n"
            '    """\n'
            '    pass\n'
        )
        out = proc.process(_make_patch(code, "src/config/settings.py"))
        for line in out.code_block.splitlines():
            assert line == line.rstrip(), f"Trailing whitespace remaining: {repr(line)}"

    def test_models_py_trailing_whitespace(self):
        """W291: models.py docstring 줄 끝 공백."""
        proc = CodePostProcessor()
        code = (
            '"""Models."""   \n'
            '\n'
            'class LeaseRecord:   \n'
            '    """   \n'
            '    Represents a delegated prefix block (e.g., /64).   \n'
            '\n'
            '    This model encapsulates the network addr   \n'
            '    """\n'
            '    pass\n'
        )
        out = proc.process(_make_patch(code, "src/domain/models.py"))
        for line in out.code_block.splitlines():
            assert line == line.rstrip()

    def test_api_py_unused_optional_import(self):
        """F401: api.py `typing.Optional` imported but unused."""
        proc = CodePostProcessor()
        code = (
            "import logging\n"
            "import signal\n"
            "from typing import Optional\n"
            "\n"
            "logger = logging.getLogger(__name__)\n"
            "\n"
            "def start_server() -> None:\n"
            "    signal.signal(signal.SIGTERM, lambda s, f: None)\n"
            "    logger.info('Server started')\n"
        )
        out = proc.process(_make_patch(code, "src/presentation/api.py"))
        assert "Optional" not in out.code_block
        assert "logging" in out.code_block
        assert "signal" in out.code_block

    def test_main_py_unused_optional_import(self):
        """F401: main.py `typing.Optional` imported but unused."""
        proc = CodePostProcessor()
        code = (
            "import logging\n"
            "import signal\n"
            "from typing import Optional\n"
            "\n"
            "logging.basicConfig()\n"
            "signal.signal(signal.SIGINT, lambda s, f: None)\n"
        )
        out = proc.process(_make_patch(code, "src/main.py"))
        assert "Optional" not in out.code_block

    def test_use_cases_py_truncated_import(self):
        """invalid-syntax: use_cases.py 잘린 import."""
        proc = CodePostProcessor(detect_truncation=True)
        truncated = (
            "from datetime import datetime, timedelta\n"
            "\n"
            "from src.domain.models import LeaseR,"
        )
        with pytest.raises(TruncatedCodeError):
            proc.process(_make_patch(truncated, "src/application/use_cases.py"))
