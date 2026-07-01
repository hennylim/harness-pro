import os
import subprocess
from pathlib import Path

from agent.infrastructure.fs.git_repository_manager import GitRepositoryManager
from config.settings import Settings


def test_settings_parses_multiple_git_repository_urls(monkeypatch):
    monkeypatch.setenv(
        "GIT_REPOSITORY_URLS",
        "https://example.com/repo1.git;https://example.com/repo2.git\nhttps://example.com/repo3.git",
    )
    settings = Settings()
    assert settings.git_repository_list == [
        "https://example.com/repo1.git",
        "https://example.com/repo2.git",
        "https://example.com/repo3.git",
    ]


def test_git_repository_manager_clones_repository(tmp_path):
    source_repo = tmp_path / "source_repo"
    source_repo.mkdir()
    subprocess.run(
        ["git", "init"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tester"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (source_repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    workspace = tmp_path / "workspace"
    manager = GitRepositoryManager(
        workspace_dir=str(workspace),
        repo_urls=[str(source_repo)],
        repos_dir="repos",
        analysis_dir="analysis",
    )
    manager.prepare_repositories()

    assert (workspace / "repos").exists()
    assert (workspace / "analysis").exists()
    assert list((workspace / "analysis").glob("*.json"))

    summary = manager.get_repo_analysis_summary()
    assert "Repository:" in summary
    assert "Files analysed:" in summary
    assert "No changes since last analysis." in summary
