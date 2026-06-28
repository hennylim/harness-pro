"""Integration tests for LocalFileSystemAdapter."""
import os
import pytest

from agent.domain.exceptions import PathEscapeError
from agent.infrastructure.fs.local_adapter import LocalFileSystemAdapter


@pytest.fixture
def adapter(tmp_path):
    return LocalFileSystemAdapter(workspace_dir=str(tmp_path))


class TestWriteAndRead:
    def test_write_creates_file(self, adapter, tmp_path):
        adapter.write_file("hello.py", "print('hi')\n", dry_run=False)
        assert os.path.exists(tmp_path / "hello.py")

    def test_write_creates_subdirs(self, adapter, tmp_path):
        adapter.write_file("src/deep/mod.py", "x = 1\n", dry_run=False)
        assert (tmp_path / "src" / "deep" / "mod.py").exists()

    def test_read_file(self, adapter, tmp_path):
        (tmp_path / "a.py").write_text("x = 42\n")
        assert adapter.read_file("a.py") == "x = 42\n"

    def test_dry_run_does_not_write(self, adapter, tmp_path):
        adapter.write_file("ghost.py", "x = 1\n", dry_run=True)
        assert not (tmp_path / "ghost.py").exists()


class TestSecurity:
    def test_path_traversal_rejected(self, adapter):
        with pytest.raises(PathEscapeError):
            adapter.write_file("../../etc/passwd", "bad", dry_run=False)

    def test_absolute_path_rejected(self, adapter):
        with pytest.raises(PathEscapeError):
            adapter.get_absolute_path("/etc/passwd")


class TestBackup:
    def test_backup_created_for_existing_file(self, adapter, tmp_path):
        (tmp_path / "existing.py").write_text("old content\n")
        bak = adapter.backup_file("existing.py")
        assert bak is not None
        assert os.path.exists(bak)

    def test_backup_returns_none_for_missing_file(self, adapter):
        result = adapter.backup_file("nonexistent.py")
        assert result is None


class TestSkeleton:
    def test_skeleton_lists_py_files(self, adapter, tmp_path):
        (tmp_path / "mod.py").write_text("def foo(): pass\n")
        skeleton = adapter.get_workspace_skeleton()
        assert "mod.py" in skeleton

    def test_skeleton_excludes_non_py(self, adapter, tmp_path):
        (tmp_path / "data.json").write_text("{}")
        skeleton = adapter.get_workspace_skeleton()
        assert "data.json" not in skeleton

    def test_skeleton_header_full_content(self, tmp_path):
        """헤더 파일(.h)은 줄 제한 없이 전체 포함."""
        from agent.infrastructure.fs.local_adapter import LocalFileSystemAdapter
        adapter = LocalFileSystemAdapter(str(tmp_path))
        import os
        os.makedirs(str(tmp_path / "include"), exist_ok=True)
        # 50줄짜리 헤더
        header = "\n".join([f"/* line {i} */" for i in range(50)]) + "\n"
        (tmp_path / "include/lib.h").write_text(header)
        # 40줄짜리 소스 (30줄 제한 적용)
        os.makedirs(str(tmp_path / "src"), exist_ok=True)
        src = "\n".join([f"// line {i}" for i in range(40)]) + "\n"
        (tmp_path / "src/main.c").write_text(src)

        skeleton = adapter.get_workspace_skeleton(
            accepted_extensions=frozenset({".h", ".c"}),
            header_extensions=frozenset({".h"}),
        )
        # 헤더는 전체 50줄
        assert skeleton.count("/* line ") == 50
        # 소스는 30줄 제한
        assert skeleton.count("// line ") == 30
        assert "more lines" in skeleton

    def test_skeleton_excludes_build_dir(self, tmp_path):
        """`.build` 디렉터리는 skeleton에서 제외."""
        from agent.infrastructure.fs.local_adapter import LocalFileSystemAdapter
        import os
        adapter = LocalFileSystemAdapter(str(tmp_path))
        os.makedirs(str(tmp_path / ".build"))
        (tmp_path / ".build/program.c").write_text("// generated\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src/main.c").write_text("// real\n")
        skeleton = adapter.get_workspace_skeleton(
            accepted_extensions=frozenset({".c"}),
        )
        assert "program.c" not in skeleton
        assert "main.c" in skeleton
