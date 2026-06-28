"""
entry_point_finder.py – 워크스페이스에서 진입점 파일을 동적으로 탐색한다.

탐색 전략 (우선순위 순)
-----------------------
1. 고정 후보 목록 확인 (src/main.sh, main.sh 등)
2. 파일명에 'main' 이 포함된 파일 탐색
3. 'main' 을 포함하지 않지만 파일이 1개인 경우 그것을 진입점으로 사용
4. 파일이 여러 개면 shebang + 실행 가능 여부로 최적 후보 선택

언어별 진입점 특성
-----------------
Bash : shebang(#!/) 으로 시작하는 .sh 파일
Python: if __name__ == "__main__" 패턴이 있는 .py 파일
C/C++ : 컴파일된 바이너리 (.build/program)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def find_bash_entry(
    workspace_dir: str,
    candidates: list[str] | None = None,
) -> str | None:
    """
    Bash 프로젝트의 진입점 .sh 파일을 찾는다.

    Parameters
    ----------
    workspace_dir:
        검색 루트 디렉터리.
    candidates:
        우선 확인할 고정 경로 목록.

    Returns
    -------
    str | None
        절대 경로, 또는 탐색 실패 시 None.
    """
    fixed = candidates or [
        "src/main.sh", "main.sh",
        "src/run.sh",  "run.sh",
        "src/start.sh","start.sh",
    ]

    # ── 1단계: 고정 후보 확인 ─────────────────────────────────────────────────
    for rel in fixed:
        full = os.path.join(workspace_dir, rel)
        if os.path.isfile(full):
            log.debug("entry_finder.fixed_match", path=rel)
            return full

    # ── 2단계: 워크스페이스 전체 .sh 파일 수집 (.build, .harness_backups 제외) ─
    sh_files = [
        p for p in sorted(Path(workspace_dir).rglob("*.sh"))
        if not any(skip in str(p) for skip in [".build", ".harness_backups", "__pycache__"])
    ]

    if not sh_files:
        log.warning("entry_finder.no_sh_files", workspace=workspace_dir)
        return None

    log.debug("entry_finder.found_sh_files",
              files=[str(p) for p in sh_files])

    # ── 3단계: 파일명에 'main' 포함 ───────────────────────────────────────────
    main_files = [p for p in sh_files if "main" in p.stem.lower()]
    if len(main_files) == 1:
        log.info("entry_finder.name_match", path=str(main_files[0]))
        return str(main_files[0])

    # ── 4단계: shebang + 직접 실행 가능 구조 탐색 ──────────────────────────────
    # shebang 이 있고, 함수를 정의하지 않고 바로 실행하는 패턴
    scored: list[tuple[int, Path]] = []
    for sh in sh_files:
        score = _score_bash_entry(sh)
        if score > 0:
            scored.append((score, sh))

    if scored:
        scored.sort(key=lambda x: (-x[0], str(x[1])))
        best = scored[0][1]
        log.info("entry_finder.score_match",
                 path=str(best), score=scored[0][0])
        return str(best)

    # ── 5단계: 파일이 1개면 그것을 사용 ──────────────────────────────────────
    if len(sh_files) == 1:
        log.info("entry_finder.single_file", path=str(sh_files[0]))
        return str(sh_files[0])

    # ── 6단계: 가장 짧은 경로(루트에 가까운) 파일 선택 ────────────────────────
    shallowest = min(sh_files, key=lambda p: len(p.parts))
    log.info("entry_finder.shallowest", path=str(shallowest))
    return str(shallowest)


def find_python_entry(
    workspace_dir: str,
    candidates: list[str] | None = None,
) -> str | None:
    """Python 프로젝트의 진입점 .py 파일을 찾는다."""
    fixed = candidates or [
        "src/main.py", "main.py",
        "src/app.py",  "app.py",
        "src/run.py",  "run.py",
        "src/cli.py",  "cli.py",
    ]

    # 1. 고정 후보
    for rel in fixed:
        full = os.path.join(workspace_dir, rel)
        if os.path.isfile(full):
            return full

    # 2. 전체 .py 파일 수집
    py_files = [
        p for p in sorted(Path(workspace_dir).rglob("*.py"))
        if "__pycache__" not in str(p)
        and ".harness_backups" not in str(p)
        and not p.stem.startswith("test_")
        and not p.stem.endswith("_test")
    ]
    if not py_files:
        return None

    # 3. if __name__ == "__main__" 패턴
    main_pattern = re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']')
    main_files = [
        p for p in py_files
        if _file_contains(p, main_pattern)
    ]
    if len(main_files) == 1:
        return str(main_files[0])
    if main_files:
        # 여러 개면 'main' 이름 우선
        named = [p for p in main_files if "main" in p.stem.lower()]
        return str(named[0] if named else main_files[0])

    # 4. 파일명에 main 포함
    named = [p for p in py_files if "main" in p.stem.lower()]
    if named:
        return str(named[0])

    # 5. 단일 파일
    if len(py_files) == 1:
        return str(py_files[0])

    return None


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

_SHEBANG_RE = re.compile(r'^#!\s*/usr/(bin/env\s+)?(ba)?sh')
_MAIN_CALL_RE = re.compile(r'^\s*(main\s*\$?@?|main\s*"?\$@"?)\s*$', re.MULTILINE)
_DIRECT_CMD_RE = re.compile(
    r'^\s*(echo|printf|read|source|\.|iptables|ip|ping|curl|wget)',
    re.MULTILINE,
)


def _score_bash_entry(path: Path) -> int:
    """진입점 가능성을 0-10 점수로 평가. 0 이면 진입점 아님."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0

    score = 0

    # shebang 있으면 +4
    if _SHEBANG_RE.match(content):
        score += 4

    # main 함수 호출 패턴 있으면 +3
    if _MAIN_CALL_RE.search(content):
        score += 3

    # 직접 실행 명령 있으면 +2
    if _DIRECT_CMD_RE.search(content):
        score += 2

    # 파일명에 main 포함 +1
    if "main" in path.stem.lower():
        score += 1

    return score


def _file_contains(path: Path, pattern: re.Pattern) -> bool:
    try:
        return bool(pattern.search(
            path.read_text(encoding="utf-8", errors="ignore")
        ))
    except OSError:
        return False
