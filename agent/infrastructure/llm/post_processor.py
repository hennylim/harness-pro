"""
CodePostProcessor – LLM 이 생성한 코드를 디스크에 쓰기 전에
자동으로 수정 가능한 사소한 lint 문제를 제거한다.

처리 항목 (언어 무관)
----------------------
1. 줄 끝 trailing whitespace 제거  (W291 / W293)
2. 파일 끝 빈 줄 보장              (W292)
3. 파일 끝 과도한 빈 줄 제거       (W391)

Python 전용 후처리
------------------
4. 미사용 import 탐지 및 제거      (F401)

잘림(Truncation) 감지
---------------------
5. 파일 끝이 미완성 import 줄(trailing comma)로 끝나는 경우
6. 빈 코드
"""
from __future__ import annotations

import ast
import re

import structlog

from agent.domain.entities import CodePatch

log = structlog.get_logger(__name__)

_PYTHON_EXTENSIONS = frozenset({".py", ".pyw"})


class TruncatedCodeError(Exception):
    """LLM 이 생성한 코드가 잘렸을 때 발생."""


class CodePostProcessor:
    """
    생성된 코드를 디스크 저장 전에 정제한다.

    Parameters
    ----------
    fix_trailing_whitespace:
        줄 끝 공백 자동 제거 (기본 True).
    fix_unused_imports:
        Python 미사용 import 자동 제거 (기본 True).
    detect_truncation:
        코드 잘림 감지 시 TruncatedCodeError 발생 (기본 True).
    """

    def __init__(
        self,
        fix_trailing_whitespace: bool = True,
        fix_unused_imports: bool = True,
        detect_truncation: bool = True,
    ) -> None:
        self._fix_ws = fix_trailing_whitespace
        self._fix_imports = fix_unused_imports
        self._detect_truncation = detect_truncation

    def process(self, patch: CodePatch) -> CodePatch:
        """
        CodePatch 를 받아 정제된 새 CodePatch 를 반환한다.

        처리 순서
        ---------
        1. 잘림 감지 (빈 코드, trailing comma import)
        2. Trailing whitespace 제거
        3. EOF 정규화
        4. 미사용 import 제거 (Python only)

        Raises
        ------
        TruncatedCodeError
            빈 코드 또는 명백한 잘림이 감지된 경우.
        """
        code = patch.code_block
        file_path = patch.target_file
        ext = ("." + file_path.rsplit(".", 1)[-1].lower()) if "." in file_path else ""

        fixes: list[str] = []

        # ── 1. 잘림 감지 ───────────────────────────────────────────────────────
        if self._detect_truncation:
            reason = _detect_truncation(code)
            if reason:
                raise TruncatedCodeError(
                    f"코드 잘림 감지 ({file_path}): {reason}\n"
                    f"마지막 50자: {repr(code[-50:] if len(code) >= 50 else code)}"
                )

        # ── 2. Trailing whitespace 제거 ────────────────────────────────────────
        if self._fix_ws:
            new_code, n = _strip_trailing_whitespace(code)
            if n > 0:
                fixes.append(f"trailing_whitespace:{n}lines")
                code = new_code

        # ── 3. EOF 정규화 ──────────────────────────────────────────────────────
        new_code = _normalize_eof(code)
        if new_code != code:
            fixes.append("eof_newline")
            code = new_code

        # ── 4. 미사용 import 제거 (Python only) ───────────────────────────────
        if self._fix_imports and ext in _PYTHON_EXTENSIONS:
            new_code, removed = _remove_unused_imports(code)
            if removed:
                fixes.append(f"unused_imports:{','.join(removed)}")
                code = new_code

        if fixes:
            log.info("postprocessor.fixed", file=file_path, fixes=fixes)

        return CodePatch(
            target_file=patch.target_file,
            explanation=patch.explanation,
            code_block=code,
        )


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _detect_truncation(code: str) -> str | None:
    """
    명백한 잘림 징후만 감지한다. 오탐(false positive)을 최소화하도록
    범위를 좁게 잡는다.

    감지 조건
    ---------
    1. 코드가 비어 있음
    2. 마지막 비공백 줄이 trailing comma 로 끝나는 import 문
       (예: "from src.models import LeaseRecord,")
    """
    stripped = code.strip()
    if not stripped:
        return "코드가 비어 있음"

    # 마지막 의미 있는 줄 추출
    last_line = ""
    for line in reversed(stripped.splitlines()):
        if line.strip():
            last_line = line
            break

    # trailing comma import: "from X import Y," 또는 "import A,"
    if re.search(r',\s*$', last_line) and \
       re.search(r'^\s*(from\s+\S+\s+import|import)\s', last_line):
        return f"import 문이 trailing comma 로 끝남: {repr(last_line.strip()[-40:])}"

    return None


def _strip_trailing_whitespace(code: str) -> tuple[str, int]:
    """줄 끝 공백/탭 제거. (수정 코드, 수정된 줄 수) 반환."""
    lines = code.split("\n")
    fixed = [line.rstrip() for line in lines]
    n = sum(1 for a, b in zip(lines, fixed) if a != b)
    return "\n".join(fixed), n


def _normalize_eof(code: str) -> str:
    """파일이 줄바꿈 1개로 끝나도록 보장."""
    return code.rstrip("\n") + "\n"


def _remove_unused_imports(code: str) -> tuple[str, list[str]]:
    """autoflake 가 있으면 사용, 없으면 AST 기반 처리."""
    try:
        import autoflake  # type: ignore
        new_code = autoflake.fix_code(
            code, remove_all_unused_imports=True, remove_unused_variables=False
        )
        if new_code == code:
            return code, []
        removed = _diff_imports(code, new_code)
        return new_code, removed
    except ImportError:
        pass
    return _ast_remove_unused_imports(code)


def _diff_imports(original: str, fixed: str) -> list[str]:
    orig_lines = set(original.splitlines())
    fixed_lines = set(fixed.splitlines())
    removed = []
    for line in orig_lines - fixed_lines:
        s = line.strip()
        if s.startswith(("import ", "from ")):
            removed.append(s[:60])
    return removed


def _ast_remove_unused_imports(code: str) -> tuple[str, list[str]]:
    """AST 기반 간이 미사용 import 제거."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, []

    # 코드에서 사용된 이름 수집
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value  # type: ignore[assignment]
            if isinstance(root, ast.Name):
                used_names.add(root.id)

    lines = code.splitlines(keepends=True)
    unused_line_nos: set[int] = set()  # 1-indexed
    removed: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        line_no = node.lineno

        names_in_import: list[str] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                names_in_import.append(alias.asname or alias.name.split(".")[0])
        else:
            for alias in node.names:
                if alias.name == "*":
                    names_in_import.append("*")
                else:
                    names_in_import.append(alias.asname or alias.name)

        any_used = any(n == "*" or n in used_names for n in names_in_import)
        if not any_used:
            unused_line_nos.add(line_no)
            removed.append(lines[line_no - 1].strip()[:60])

    if not unused_line_nos:
        return code, []

    new_lines = [l for i, l in enumerate(lines, 1) if i not in unused_line_nos]
    return "".join(new_lines), removed
