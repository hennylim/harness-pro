"""
Local file-system adapter.

Security
--------
* All paths are resolved inside a fixed ``workspace_dir``; any attempt to
  escape (via ``..`` or absolute paths) raises ``PathEscapeError``.
* ``write_file`` creates parent directories automatically.

Dry-run
-------
When ``dry_run=True``, writes and deletes are skipped and the intended
content is logged instead (useful for CI preview runs).
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from typing import Optional

import structlog

from agent.domain.exceptions import FileSystemError, PathEscapeError
from agent.domain.interfaces import IFileSystemAdapter

log = structlog.get_logger(__name__)


class LocalFileSystemAdapter(IFileSystemAdapter):
    """
    File-system operations scoped to a single workspace directory.

    Parameters
    ----------
    workspace_dir:
        Absolute or relative path to the root of the generated project.
        Created on first use if it does not exist.
    backup_dir:
        Sub-directory (relative to workspace) where backups are stored.
        Defaults to ``.harness_backups``.
    """

    def __init__(
        self,
        workspace_dir: str,
        backup_dir: str = ".harness_backups",
    ) -> None:
        self._base = os.path.realpath(os.path.abspath(workspace_dir))
        self._backup_root = os.path.join(self._base, backup_dir)
        os.makedirs(self._base, exist_ok=True)
        os.makedirs(self._backup_root, exist_ok=True)
        log.info("fs.init", workspace=self._base)

    # ── IFileSystemAdapter ────────────────────────────────────────────────────

    def get_absolute_path(self, relative_path: str) -> str:
        return self._safe_join(relative_path)

    def write_file(self, relative_path: str, content: str, dry_run: bool) -> None:
        abs_path = self._safe_join(relative_path)
        if dry_run:
            preview = content[:300] + ("…" if len(content) > 300 else "")
            log.info(
                "fs.write.dry_run",
                path=relative_path,
                size_chars=len(content),
                preview=preview,
            )
            return
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        try:
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(content)
            log.info("fs.write.ok", path=relative_path, size_chars=len(content))
        except OSError as exc:
            raise FileSystemError(f"Failed to write {relative_path!r}: {exc}") from exc

    def read_file(self, relative_path: str) -> str:
        abs_path = self._safe_join(relative_path)
        try:
            with open(abs_path, encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError as exc:
            raise FileSystemError(f"File not found: {relative_path!r}") from exc
        except OSError as exc:
            raise FileSystemError(f"Failed to read {relative_path!r}: {exc}") from exc

    def delete_file(self, relative_path: str, dry_run: bool) -> None:
        abs_path = self._safe_join(relative_path)
        if dry_run:
            log.info("fs.delete.dry_run", path=relative_path)
            return
        if not os.path.exists(abs_path):
            log.warning("fs.delete.not_found", path=relative_path)
            return
        try:
            os.remove(abs_path)
            log.info("fs.delete.ok", path=relative_path)
        except OSError as exc:
            raise FileSystemError(f"Failed to delete {relative_path!r}: {exc}") from exc

    def get_workspace_skeleton(
        self,
        accepted_extensions: frozenset | None = None,
    ) -> str:
        """
        Return a compact text snapshot of workspace source files.

        The snapshot is intentionally richer than a file-name skeleton: generated
        source files often depend on declarations from earlier headers. Keeping
        enough of each small file in the prompt prevents the LLM from inventing
        struct fields, typedefs, and function names that do not exist.

        Parameters
        ----------
        accepted_extensions:
            파일 확장자 필터 (점 포함, 소문자). None 이면 .py 만 포함 (하위 호환).
        """
        exts = accepted_extensions or frozenset({".py"})
        max_lines_per_file = 200
        max_chars_per_file = 20000
        max_total_chars = 80000

        chunks: list[str] = []
        total_chars = 0

        for root, dirs, files in os.walk(self._base):
            dirs[:] = [
                d for d in sorted(dirs)
                if not d.startswith(".") and d != "__pycache__"
            ]
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() not in exts:
                    continue

                full = os.path.join(root, fname)
                rel = os.path.relpath(full, self._base)
                block = self._preview_text_file(
                    full,
                    rel,
                    max_lines=max_lines_per_file,
                    max_chars=max_chars_per_file,
                )

                if total_chars + len(block) > max_total_chars:
                    remaining = max_total_chars - total_chars
                    if remaining > 0:
                        chunks.append(block[:remaining])
                    chunks.append("\n... <workspace snapshot truncated>\n")
                    return "".join(chunks)

                chunks.append(block)
                total_chars += len(block)

        return "".join(chunks)

    @staticmethod
    def _preview_text_file(
        absolute_path: str,
        relative_path: str,
        max_lines: int,
        max_chars: int,
    ) -> str:
        """Return a bounded preview block for one UTF-8 text file."""
        header = f"\n--- [{relative_path}] ---\n"
        lines: list[str] = []
        chars = 0
        truncated = False

        try:
            with open(absolute_path, encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if idx >= max_lines or chars + len(line) > max_chars:
                        truncated = True
                        break
                    lines.append(line)
                    chars += len(line)
        except UnicodeDecodeError:
            return header + "  <unreadable: non-UTF-8 text>\n"
        except OSError as exc:
            return header + f"  <unreadable: {exc}>\n"

        if not lines:
            lines.append("  <empty>\n")
        if truncated:
            lines.append(
                f"... <truncated after {len(lines)} lines / {chars} chars>\n"
            )
        return header + "".join(lines)

    def backup_file(self, relative_path: str) -> Optional[str]:
        """
        Copy the file to the backup directory with a timestamp suffix.
        Returns the backup path, or None if the source file does not exist.
        """
        abs_path = self._safe_join(relative_path)
        if not os.path.exists(abs_path):
            return None
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_name = relative_path.replace(os.sep, "_").replace("/", "_")
        backup_path = os.path.join(self._backup_root, f"{safe_name}.{ts}.bak")
        try:
            shutil.copy2(abs_path, backup_path)
            log.debug("fs.backup.ok", original=relative_path, backup=backup_path)
            return backup_path
        except OSError as exc:
            log.warning("fs.backup.failed", path=relative_path, error=str(exc))
            return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _safe_join(self, relative_path: str) -> str:
        """
        Join *relative_path* with the workspace root and verify the result
        stays inside the workspace (no path traversal).
        """
        joined = os.path.realpath(os.path.join(self._base, relative_path))
        if not joined.startswith(self._base + os.sep) and joined != self._base:
            raise PathEscapeError(
                f"Path {relative_path!r} escapes workspace {self._base!r}."
            )
        return joined
