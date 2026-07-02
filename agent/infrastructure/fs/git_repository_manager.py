"""
Git repository input support.

Clone repositories listed in environment variables into the workspace,
compute per-repo analysis summaries, and persist those summaries to disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone

from agent.domain.exceptions import FileSystemError


class GitRepositoryManager:
    """Clone git repos into the workspace and maintain analysis state."""

    def __init__(
        self,
        workspace_dir: str,
        repo_urls: list[str],
        repos_dir: str = "repositories",
        analysis_dir: str = ".repo_analysis",
    ) -> None:
        self._workspace_dir = os.path.realpath(os.path.abspath(workspace_dir))
        self._repos_root = os.path.join(self._workspace_dir, repos_dir)
        self._analysis_root = os.path.join(self._workspace_dir, analysis_dir)
        self._repo_urls = [url.strip() for url in repo_urls if url and url.strip()]
        self._summaries: list[dict] = []

        os.makedirs(self._repos_root, exist_ok=True)
        os.makedirs(self._analysis_root, exist_ok=True)

    def prepare_repositories(self) -> None:
        """Clone / update all configured repositories and write analysis state."""
        for url in self._repo_urls:
            try:
                self._prepare_repository(url)
            except Exception as exc:
                raise FileSystemError(
                    f"Failed to prepare git repository {url}: {exc}"
                ) from exc

    def get_repo_analysis_summary(self) -> str:
        """Return a human-readable summary of all managed repositories."""
        if not self._summaries:
            return ""

        parts: list[str] = []
        for summary in self._summaries:
            parts.append(
                f"Repository: {summary['repo_name']} ({summary['repo_url']})"
            )
            parts.append(f"Path: {summary['path']}")
            parts.append(f"Commit: {summary['commit'] or 'unknown'}")
            parts.append(f"Files analysed: {summary['file_count']}")
            changed_files = summary.get("changed_files", [])
            if changed_files:
                parts.append("Changed files since last analysis:")
                parts.extend(f"- {path}" for path in changed_files)
            else:
                parts.append("No changes since last analysis.")
            parts.append("")
        return "\n".join(parts).strip()

    def _prepare_repository(self, repo_url: str) -> None:
        repo_name = self._normalize_repo_name(repo_url)
        repo_path = os.path.join(self._repos_root, repo_name)
        worktree_path = os.path.join(self._repos_root, f"{repo_name}-worktree")

        if not os.path.isdir(os.path.join(repo_path, ".git")):
            self._git_clone(repo_url, repo_path)
        else:
            self._git_fetch(repo_path)

        self._ensure_worktree(repo_path, worktree_path)
        summary = self._analyze_repository(repo_url, repo_name, worktree_path)
        self._summaries.append(summary)

    def _normalize_repo_name(self, repo_url: str) -> str:
        base = os.path.basename(repo_url.rstrip("/"))
        name = os.path.splitext(base)[0] if base else "repo"
        name = re.sub(r"[^A-Za-z0-9._-]", "-", name).strip("-_.")
        if not name:
            name = "repo"
        suffix = hashlib.sha1(repo_url.encode("utf-8")).hexdigest()[:8]
        return f"{name}-{suffix}"

    def _git_clone(self, repo_url: str, destination: str) -> None:
        try:
            subprocess.run(
                ["git", "clone", repo_url, destination],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"git clone failed: {exc.stderr.strip() or exc.stdout.strip()}"
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError("git executable not found") from exc

    def _git_fetch(self, repo_path: str) -> None:
        try:
            subprocess.run(
                ["git", "fetch", "--all", "--prune"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"git fetch failed: {exc.stderr.strip() or exc.stdout.strip()}"
            ) from exc

    def _git_current_commit(self, repo_path: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def _analyze_repository(
        self,
        repo_url: str,
        repo_name: str,
        repo_path: str,
    ) -> dict:
        file_hashes = self._compute_file_hashes(repo_path)
        previous = self._load_analysis(repo_name)
        changed_files = self._compute_changed_files(file_hashes, previous.get("file_hashes", {}))
        summary = {
            "repo_name": repo_name,
            "repo_url": repo_url,
            "path": os.path.relpath(repo_path, self._workspace_dir),
            "commit": self._git_current_commit(repo_path),
            "file_count": len(file_hashes),
            "changed_files": changed_files,
            "file_hashes": file_hashes,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_analysis(repo_name, summary)
        return summary

    def _compute_file_hashes(self, repo_path: str) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d != ".git" and d != "__pycache__"]
            for filename in sorted(files):
                if filename == ".git":
                    continue
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, repo_path)
                try:
                    with open(abs_path, "rb") as fh:
                        checksum = hashlib.sha256(fh.read()).hexdigest()
                except OSError:
                    continue
                hashes[rel_path] = checksum
        return hashes

    def _compute_changed_files(
        self,
        current_hashes: dict[str, str],
        previous_hashes: dict[str, str],
    ) -> list[str]:
        if not previous_hashes:
            return []

        changed: list[str] = []
        for path, checksum in current_hashes.items():
            if previous_hashes.get(path) != checksum:
                changed.append(path)
        for path in previous_hashes:
            if path not in current_hashes:
                changed.append(path)
        return sorted(changed)

    def _analysis_path(self, repo_name: str) -> str:
        return os.path.join(self._analysis_root, f"{repo_name}.json")

    def _load_analysis(self, repo_name: str) -> dict:
        path = self._analysis_path(repo_name)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except OSError as exc:
            raise FileSystemError(f"Unable to read analysis state: {exc}") from exc
        except json.JSONDecodeError:
            return {}

    def _save_analysis(self, repo_name: str, summary: dict) -> None:
        path = self._analysis_path(repo_name)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise FileSystemError(f"Unable to write analysis state: {exc}") from exc

    def _ensure_worktree(self, repo_path: str, worktree_path: str) -> None:
        worktree_exists = os.path.exists(worktree_path)
        if worktree_exists:
            self._remove_worktree(repo_path, worktree_path)

        target_ref = self._git_worktree_target_ref(repo_path)
        self._git_worktree_add(repo_path, worktree_path, target_ref)

    def _remove_worktree(self, repo_path: str, worktree_path: str) -> None:
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_path],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            if os.path.isdir(worktree_path):
                shutil.rmtree(worktree_path)
            elif os.path.exists(worktree_path):
                os.remove(worktree_path)

    def _git_worktree_target_ref(self, repo_path: str) -> str:
        try:
            result = subprocess.run(
                ["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return result.stdout.strip()
            except subprocess.CalledProcessError:
                return "HEAD"

    def _git_worktree_add(self, repo_path: str, worktree_path: str, target_ref: str) -> None:
        try:
            subprocess.run(
                ["git", "worktree", "add", "--force", worktree_path, target_ref],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"git worktree add failed: {exc.stderr.strip() or exc.stdout.strip()}"
            ) from exc
