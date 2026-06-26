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

    def test_skeleton_includes_short_file_content(self, adapter, tmp_path):
        """짧은 파일도 StopIteration 때문에 unreadable 로 처리되면 안 된다."""
        (tmp_path / "short.py").write_text("line1\nline2\n")
        skeleton = adapter.get_workspace_skeleton()
        assert "short.py" in skeleton
        assert "line1" in skeleton
        assert "line2" in skeleton
        assert "<unreadable>" not in skeleton

    def test_skeleton_includes_declarations_after_first_ten_lines(self, adapter, tmp_path):
        """C 헤더의 struct/typedef 선언이 10줄 뒤에 있어도 컨텍스트에 포함한다."""
        header = "\n".join(f"// filler {i}" for i in range(15))
        header += "\ntypedef struct {\n    char *interface_name;\n} config_t;\n"
        (tmp_path / "config.h").write_text(header)
        skeleton = adapter.get_workspace_skeleton(frozenset({".h"}))
        assert "config.h" in skeleton
        assert "interface_name" in skeleton
        assert "config_t" in skeleton
