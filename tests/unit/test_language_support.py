"""
언어 프로필, 프롬프트 빌더, 센서 라우터 단위 테스트.
"""
import os
import stat
import textwrap

import pytest

from agent.domain.profiles import PROFILE_REGISTRY
from agent.domain.profiles.python_profile import PythonProfile
from agent.domain.profiles.bash_profile import BashProfile
from agent.domain.profiles.c_profile import CProfile
from agent.domain.profiles.cpp_profile import CppProfile
from agent.infrastructure.llm.prompt_builder import PromptBuilder
from agent.infrastructure.sensors.router import EXTENSION_MAP, LanguageSensorRouter


# ── Profile registry ──────────────────────────────────────────────────────────

class TestProfileRegistry:
    def test_all_languages_registered(self):
        for key in ("python", "bash", "c", "cpp"):
            assert key in PROFILE_REGISTRY, f"'{key}' not in PROFILE_REGISTRY"

    def test_profiles_have_nonempty_fields(self):
        for key, profile in PROFILE_REGISTRY.items():
            assert profile.name, f"{key}.name is empty"
            assert profile.extensions, f"{key}.extensions is empty"
            assert profile.code_style_rules, f"{key}.code_style_rules is empty"
            assert profile.plan_file_extension_hint, f"{key}.plan_file_extension_hint is empty"

    def test_extensions_are_lowercase_with_dot(self):
        for key, profile in PROFILE_REGISTRY.items():
            for ext in profile.extensions:
                assert ext.startswith("."), f"{key}: extension '{ext}' missing dot"
                assert ext == ext.lower(), f"{key}: extension '{ext}' not lowercase"


# ── Profile specifics ─────────────────────────────────────────────────────────

class TestPythonProfile:
    def test_requires_compilation_false(self):
        assert PythonProfile().requires_compilation is False

    def test_extension(self):
        assert ".py" in PythonProfile().extensions


class TestBashProfile:
    def test_shebang_in_header(self):
        assert "#!/usr/bin/env bash" in BashProfile().file_header_hint

    def test_extension(self):
        assert ".sh" in BashProfile().extensions


class TestCProfile:
    def test_requires_compilation(self):
        assert CProfile().requires_compilation is True

    def test_extensions_include_header(self):
        p = CProfile()
        assert ".c" in p.extensions
        assert ".h" in p.extensions

    def test_skeleton_includes_headers(self):
        p = CProfile()
        assert ".h" in p.skeleton_extensions


class TestCppProfile:
    def test_requires_compilation(self):
        assert CppProfile().requires_compilation is True

    def test_extensions(self):
        p = CppProfile()
        assert ".cpp" in p.extensions

    def test_skeleton_includes_headers(self):
        p = CppProfile()
        assert ".hpp" in p.skeleton_extensions


# ── PromptBuilder ─────────────────────────────────────────────────────────────

class TestPromptBuilder:
    @pytest.fixture(params=list(PROFILE_REGISTRY.keys()))
    def builder(self, request):
        return PromptBuilder(PROFILE_REGISTRY[request.param])

    def test_plan_system_contains_language_name(self, builder):
        prompt = builder.build_plan_system()
        assert builder._profile.name in prompt

    def test_plan_system_contains_ext_hint(self, builder):
        prompt = builder.build_plan_system()
        assert builder._profile.plan_file_extension_hint in prompt

    def test_code_system_contains_language_name(self, builder):
        prompt = builder.build_code_system()
        assert builder._profile.name in prompt

    def test_code_system_contains_style_rules(self, builder):
        prompt = builder.build_code_system()
        # 스타일 규칙의 첫 줄이 포함되는지 확인
        first_rule = builder._profile.code_style_rules.splitlines()[0].strip()
        assert first_rule in prompt

    def test_summarise_system_is_static(self):
        s1 = PromptBuilder.build_summarise_system()
        s2 = PromptBuilder.build_summarise_system()
        assert s1 == s2
        assert len(s1) > 10


# ── LanguageSensorRouter ──────────────────────────────────────────────────────

class TestLanguageSensorRouter:
    def test_all_registered_extensions_route(self):
        router = LanguageSensorRouter()
        for ext in EXTENSION_MAP:
            # dry_run=True なので実際のツールは不要
            result = router.verify_code(f"/fake/file{ext}", dry_run=True)
            assert result.passed, f"dry_run should pass for ext '{ext}'"

    def test_unknown_extension_returns_no_op(self):
        router = LanguageSensorRouter()
        result = router.verify_code("/fake/file.rs", dry_run=False)
        assert result.passed
        assert result.tool == "no-op"
        assert any("No sensor" in w for w in result.warnings)

    def test_extension_map_covers_expected_keys(self):
        expected = {".py", ".sh", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp"}
        for ext in expected:
            assert ext in EXTENSION_MAP, f"'{ext}' not in EXTENSION_MAP"


# ── FS skeleton extension filter ──────────────────────────────────────────────

class TestSkeletonExtensionFilter:
    def test_skeleton_filters_by_extension(self, tmp_path):
        from agent.infrastructure.fs.local_adapter import LocalFileSystemAdapter
        adapter = LocalFileSystemAdapter(str(tmp_path))
        (tmp_path / "main.py").write_text("x = 1\n")
        (tmp_path / "main.c").write_text("int main(){}\n")
        (tmp_path / "run.sh").write_text("#!/usr/bin/env bash\n")

        py_skel = adapter.get_workspace_skeleton(frozenset({".py"}))
        assert "main.py" in py_skel
        assert "main.c" not in py_skel

        c_skel = adapter.get_workspace_skeleton(frozenset({".c"}))
        assert "main.c" in c_skel
        assert "main.py" not in c_skel

        sh_skel = adapter.get_workspace_skeleton(frozenset({".sh"}))
        assert "run.sh" in sh_skel
