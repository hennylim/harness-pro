"""
적응형 플래너 및 청크 생성 단위/통합 테스트.

핵심 검증:
  1. EnhancedProjectPlan 생성 (normal + chunked 분류)
  2. FileSection 컨텍스트 누적 (이전 섹션이 다음에 전달되는지)
  3. 섹션 조립 → lint 파이프라인
  4. 빌드 에러 자동 수정 (auto-fix) 로직
  5. 강화 계획 실패 시 일반 계획 폴백
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from agent.domain.entities import (
    BuildResult,
    ChunkedTask,
    ChunkStatus,
    CodePatch,
    EnhancedProjectPlan,
    ExecutionResult,
    FileSection,
    LintResult,
    ProjectPlan,
    RunStatus,
    Task,
    TaskAction,
    TaskStatus,
)
from agent.domain.exceptions import LLMParseError, PlanValidationError
from agent.usecase.orchestrator import HarnessOrchestrator
from agent.usecase.adaptive_planner import AdaptivePlanner
from config.logging import configure_logging

configure_logging(level="ERROR", fmt="console")


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _make_task(tid=1, path="src/a.c") -> Task:
    return Task(task_id=tid, file_path=path,
                action=TaskAction.CREATE, description="create")


def _make_section(idx: int, name: str, desc: str = "section") -> FileSection:
    return FileSection(
        section_index=idx,
        section_name=name,
        description=desc,
    )


def _make_chunked_task(path="src/big.c", n_sections=3) -> ChunkedTask:
    return ChunkedTask(
        task_id=1,
        file_path=path,
        action=TaskAction.CREATE,
        description="big file",
        sections=[
            _make_section(i, f"section_{i}") for i in range(n_sections)
        ],
    )


def _make_orch(
    llm=None, fs=None, sensor=None, run_repo=None,
    execution_validator=None,
    max_self_heal=2,
    chunk_threshold=80,
    dry_run=False,
) -> HarnessOrchestrator:
    llm = llm or MagicMock()
    fs = fs or MagicMock()
    sensor = sensor or MagicMock()
    run_repo = run_repo or MagicMock()
    fs.backup_file.return_value = None
    fs.get_workspace_skeleton.return_value = ""
    fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"
    sensor.verify_code.return_value = LintResult(passed=True, tool="gcc")
    return HarnessOrchestrator(
        llm=llm, fs=fs, sensor=sensor, run_repo=run_repo,
        run_id="test", execution_validator=execution_validator,
        max_self_heal=max_self_heal,
        chunk_threshold_lines=chunk_threshold,
        dry_run=dry_run,
    )


# ── EnhancedProjectPlan 엔티티 ───────────────────────────────────────────────

class TestEnhancedProjectPlan:
    def test_all_tasks_count(self):
        plan = EnhancedProjectPlan(
            tasks=[_make_task(1), _make_task(2)],
            chunked_tasks=[_make_chunked_task()],
        )
        assert plan.all_tasks_count == 3

    def test_empty_plan(self):
        plan = EnhancedProjectPlan()
        assert plan.all_tasks_count == 0


class TestChunkedTask:
    def test_assembled_code_joins_generated_sections(self):
        ct = _make_chunked_task(n_sections=3)
        ct.sections[0].generated_code = "int a = 1;\n"
        ct.sections[0].status = ChunkStatus.GENERATED
        ct.sections[1].generated_code = "int b = 2;\n"
        ct.sections[1].status = ChunkStatus.GENERATED
        ct.sections[2].status = ChunkStatus.PENDING  # 미생성

        # GENERATED 섹션만 조립
        assembled = ct.assembled_code
        assert "int a = 1;" in assembled
        assert "int b = 2;" in assembled

    def test_is_complete_all_generated(self):
        ct = _make_chunked_task(n_sections=2)
        for s in ct.sections:
            s.status = ChunkStatus.GENERATED
            s.generated_code = "x = 1;\n"
        assert ct.is_complete is True

    def test_is_complete_false_when_pending(self):
        ct = _make_chunked_task(n_sections=2)
        ct.sections[0].status = ChunkStatus.GENERATED
        ct.sections[0].generated_code = "x = 1;\n"
        # sections[1] is still PENDING
        assert ct.is_complete is False

    def test_mark_done_sets_status(self):
        ct = _make_chunked_task()
        ct.mark_done()
        assert ct.status == TaskStatus.DONE

    def test_mark_failed_appends_error(self):
        ct = _make_chunked_task()
        ct.mark_failed("lint error")
        assert ct.status == TaskStatus.FAILED
        assert ct.last_error == "lint error"


# ── AdaptivePlanner ───────────────────────────────────────────────────────────

class TestAdaptivePlanner:
    def _make_planner(self, llm=None) -> AdaptivePlanner:
        return AdaptivePlanner(
            llm_adapter=llm or MagicMock(),
            chunk_threshold_lines=80,
        )

    def test_previous_context_empty_when_no_sections(self):
        result = AdaptivePlanner._build_previous_context([])
        assert result == ""

    def test_previous_context_includes_generated_sections(self):
        sections = [
            FileSection(section_index=0, section_name="includes",
                        description="", generated_code="#include <stdio.h>\n",
                        status=ChunkStatus.GENERATED),
            FileSection(section_index=1, section_name="types",
                        description="", generated_code="typedef int MyInt;\n",
                        status=ChunkStatus.GENERATED),
        ]
        ctx = AdaptivePlanner._build_previous_context(sections)
        assert "#include <stdio.h>" in ctx
        assert "typedef int MyInt;" in ctx
        assert "Section 0: includes" in ctx
        assert "Section 1: types" in ctx

    def test_previous_context_skips_empty_sections(self):
        sections = [
            FileSection(section_index=0, section_name="empty",
                        description="", generated_code="",
                        status=ChunkStatus.PENDING),
            FileSection(section_index=1, section_name="real",
                        description="", generated_code="int x = 1;\n",
                        status=ChunkStatus.GENERATED),
        ]
        ctx = AdaptivePlanner._build_previous_context(sections)
        assert "int x = 1;" in ctx
        assert "empty" not in ctx

    def test_generate_section_passes_previous_context_to_llm(self):
        """generate_section 이 이전 섹션 코드를 user 프롬프트에 포함하는지 확인."""
        llm = MagicMock()

        # GeminiAdapter 처럼 동작하는 mock
        class FakeLLM:
            _code_max_tokens = 8192
            calls = []

            def _call(self, system, user, max_tokens):
                self.calls.append(user)
                return json.dumps({"section_code": "int result = 42;\n"})

        fake_llm = FakeLLM()
        planner = AdaptivePlanner(llm_adapter=fake_llm, chunk_threshold_lines=80)

        prev = FileSection(
            section_index=0, section_name="includes",
            description="", generated_code="#include <stdio.h>\n",
            status=ChunkStatus.GENERATED,
        )
        current = FileSection(
            section_index=1, section_name="main_function",
            description="main function",
        )

        # GeminiAdapter 로 패칭
        from agent.infrastructure.llm.gemini_adapter import GeminiAdapter
        with patch(
            "agent.usecase.adaptive_planner.isinstance",
            side_effect=lambda obj, cls: cls == GeminiAdapter,
        ):
            code = planner.generate_section(
                file_path="src/main.c",
                section=current,
                previous_sections=[prev],
                workspace_skeleton="",
                error_feedback="",
                language_name="C",
            )

        assert code.strip() == "int result = 42;"
        # 이전 섹션이 user 프롬프트에 포함됐는지
        assert len(fake_llm.calls) == 1
        assert "#include <stdio.h>" in fake_llm.calls[0]


# ── HarnessOrchestrator + 청크 모드 ─────────────────────────────────────────

class TestOrchestratorChunkedMode:
    def test_execute_with_chunked_task_completes(self):
        """청크 태스크가 포함된 계획을 실행하면 COMPLETED 반환."""
        llm = MagicMock()
        fs = MagicMock()
        sensor = MagicMock()

        fs.backup_file.return_value = None
        fs.get_workspace_skeleton.return_value = ""
        fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"
        sensor.verify_code.return_value = LintResult(passed=True, tool="gcc")

        # 강화 계획: 1 normal + 1 chunked
        normal_task = _make_task(1, "include/types.h")
        chunked = _make_chunked_task("src/engine.c", n_sections=2)

        # LLM mock: generate_plan → 일반 플랜 (폴백용)
        llm.generate_plan.return_value = ProjectPlan(tasks=[normal_task])
        # generate_code: 일반 태스크용
        llm.generate_code.return_value = CodePatch(
            target_file="include/types.h",
            explanation="Header.",
            code_block="#ifndef TYPES_H\n#define TYPES_H\ntypedef int MyInt;\n#endif\n",
        )

        orch = _make_orch(llm=llm, fs=fs, sensor=sensor)

        # _planner.generate_enhanced_plan 을 직접 패치
        enhanced_plan = EnhancedProjectPlan(
            tasks=[normal_task],
            chunked_tasks=[chunked],
        )

        # _planner.generate_section 을 패치해 각 섹션 코드 반환
        section_codes = [
            "#include <stdio.h>\n#include \"types.h\"\n",
            "int engine_run(void) { return 0; }\n",
        ]
        call_idx = {"i": 0}

        def fake_generate_section(**kwargs):
            code = section_codes[call_idx["i"]]
            call_idx["i"] += 1
            return code

        orch._planner.generate_enhanced_plan = MagicMock(
            return_value=enhanced_plan
        )
        orch._planner.generate_section = MagicMock(
            side_effect=fake_generate_section
        )

        summary = orch.execute("build engine")

        assert summary.status == RunStatus.COMPLETED
        assert summary.failed_tasks == 0
        # 섹션이 2번 호출됐는지
        assert orch._planner.generate_section.call_count == 2

    def test_section_context_accumulates(self):
        """섹션 생성 시 이전 섹션 코드가 다음 섹션에 전달되는지 검증."""
        llm = MagicMock()
        fs = MagicMock()
        sensor = MagicMock()

        fs.backup_file.return_value = None
        fs.get_workspace_skeleton.return_value = ""
        fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"
        sensor.verify_code.return_value = LintResult(passed=True, tool="gcc")

        ct = ChunkedTask(
            task_id=1, file_path="src/big.c",
            action=TaskAction.CREATE,
            description="big C file",
            sections=[
                _make_section(0, "includes", "include headers"),
                _make_section(1, "functions", "define functions"),
                _make_section(2, "main", "main function"),
            ],
        )

        captured_prev = []

        def fake_generate_section(**kwargs):
            captured_prev.append(
                [s.section_name for s in kwargs["previous_sections"]]
            )
            return f"// section {kwargs['section'].section_index}\n"

        orch = _make_orch(llm=llm, fs=fs, sensor=sensor)
        orch._planner.generate_section = MagicMock(
            side_effect=fake_generate_section
        )

        # 청크 태스크만 실행
        summary_tasks_proxy: list[Task] = []
        from agent.domain.entities import RunSummary
        summary = RunSummary(run_id="t", started_at=datetime.now(timezone.utc))
        summary.tasks = []

        orch._execute_chunked_tasks([ct], summary, [])

        # section 0: 이전 없음
        assert captured_prev[0] == []
        # section 1: section 0 이 이전
        assert captured_prev[1] == ["includes"]
        # section 2: section 0, 1 이 이전
        assert captured_prev[2] == ["includes", "functions"]

    def test_chunked_lint_fail_falls_back_to_normal(self):
        """청크 조립 후 lint 실패 시 일반 태스크 모드로 폴백."""
        llm = MagicMock()
        fs = MagicMock()
        sensor = MagicMock()

        fs.backup_file.return_value = None
        fs.get_workspace_skeleton.return_value = ""
        fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"

        # 첫 번째 lint(청크 조립): 실패
        # 두 번째 lint(폴백 일반 모드): 성공
        sensor.verify_code.side_effect = [
            LintResult(passed=False, errors=["E: syntax"], tool="gcc"),
            LintResult(passed=True, tool="gcc"),
        ]

        llm.generate_code.return_value = CodePatch(
            target_file="src/big.c",
            explanation="Fixed.",
            code_block="int main(void) { return 0; }\n",
        )

        ct = _make_chunked_task("src/big.c", n_sections=1)

        orch = _make_orch(llm=llm, fs=fs, sensor=sensor)
        orch._planner.generate_section = MagicMock(
            return_value="int main(void) { return 0; }\n"
        )

        from agent.domain.entities import RunSummary
        summary = RunSummary(run_id="t", started_at=datetime.now(timezone.utc))
        summary.tasks = []

        orch._execute_chunked_tasks([ct], summary, [])

        # 청크 조립 lint 실패 → 일반 generate_code 호출됨
        assert llm.generate_code.called


# ── 빌드 에러 자동 수정 ──────────────────────────────────────────────────────

class TestBuildAutoFix:
    def test_extract_files_from_errors(self):
        error = (
            "/ws/src/report_generator.c:17:50: error: unknown type\n"
            "/ws/include/packet_analyzer.h:28:5: note: declared here\n"
        )
        files = HarnessOrchestrator._extract_files_from_errors(error, "/ws")
        assert "src/report_generator.c" in files

    def test_extract_files_no_match_returns_empty(self):
        files = HarnessOrchestrator._extract_files_from_errors(
            "generic error message", "/ws"
        )
        assert files == []

    def test_build_fail_triggers_auto_fix(self):
        """빌드 실패 시 _try_fix_build_error 가 호출되는지 확인."""
        llm = MagicMock()
        fs = MagicMock()
        sensor = MagicMock()
        validator = MagicMock()

        fs.backup_file.return_value = None
        fs.get_workspace_skeleton.return_value = ""
        fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"
        sensor.verify_code.return_value = LintResult(passed=True)

        # Build: 첫 번째 실패, 두 번째 성공
        validator.build.side_effect = [
            BuildResult(
                passed=False,
                errors=["/ws/src/main.c:10: error: something"],
                tool="gcc",
            ),
            BuildResult(passed=True, tool="gcc"),
        ]
        validator.run_smoke_test.return_value = ExecutionResult(
            passed=True, exit_code=0, command="./program"
        )

        # generate_code: 수정된 코드 반환
        llm.generate_code.return_value = CodePatch(
            target_file="src/main.c",
            explanation="Fixed.",
            code_block="int main(void) { return 0; }\n",
        )
        llm.generate_plan.return_value = ProjectPlan(tasks=[_make_task(1, "src/a.c")])

        orch = _make_orch(
            llm=llm, fs=fs, sensor=sensor,
            execution_validator=validator,
        )
        orch._planner.generate_enhanced_plan = MagicMock(
            return_value=EnhancedProjectPlan(
                tasks=[_make_task(1, "src/a.c")],
            )
        )
        llm.generate_code.return_value = CodePatch(
            target_file="src/a.c",
            explanation="ok",
            code_block="int x = 1;\n",
        )

        summary = orch.execute("build something")
        # 빌드가 2번 호출됐어야 함 (1회 실패 + 1회 성공)
        assert validator.build.call_count == 2
        assert summary.build_passed is True


# ── 강화 계획 실패 → 폴백 ─────────────────────────────────────────────────────

class TestEnhancedPlanFallback:
    def test_falls_back_to_normal_plan_on_error(self):
        """강화 계획 생성 실패 시 일반 LLM 플랜으로 폴백."""
        llm = MagicMock()
        fs = MagicMock()
        sensor = MagicMock()

        fs.backup_file.return_value = None
        fs.get_workspace_skeleton.return_value = ""
        fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"
        sensor.verify_code.return_value = LintResult(passed=True)

        normal_task = _make_task(1, "src/main.c")
        llm.generate_plan.return_value = ProjectPlan(tasks=[normal_task])
        llm.generate_code.return_value = CodePatch(
            target_file="src/main.c",
            explanation="ok",
            code_block="int main(void){return 0;}\n",
        )

        orch = _make_orch(llm=llm, fs=fs, sensor=sensor)
        # 강화 계획 생성이 실패하도록 패치
        orch._planner.generate_enhanced_plan = MagicMock(
            side_effect=Exception("LLM 에러")
        )

        summary = orch.execute("build something")

        # 폴백: 일반 계획 사용 → 완료
        assert summary.status == RunStatus.COMPLETED
        assert summary.completed_tasks == 1


# ── 기존 오케스트레이터 호환성 ───────────────────────────────────────────────

class TestBackwardCompatibility:
    """기존 단순 태스크 흐름이 계속 동작하는지 확인."""

    def test_single_normal_task_completes(self):
        llm = MagicMock()
        fs = MagicMock()
        sensor = MagicMock()

        fs.backup_file.return_value = None
        fs.get_workspace_skeleton.return_value = ""
        fs.get_absolute_path.side_effect = lambda p: f"/ws/{p}"
        sensor.verify_code.return_value = LintResult(passed=True)

        llm.generate_plan.return_value = ProjectPlan(
            tasks=[_make_task(1, "src/a.py")]
        )
        llm.generate_code.return_value = CodePatch(
            target_file="src/a.py",
            explanation="ok",
            code_block='"""Module."""\n\nx = 1\n',
        )

        orch = _make_orch(llm=llm, fs=fs, sensor=sensor)
        orch._planner.generate_enhanced_plan = MagicMock(
            return_value=EnhancedProjectPlan(
                tasks=[_make_task(1, "src/a.py")]
            )
        )

        summary = orch.execute("simple")
        assert summary.status == RunStatus.COMPLETED
        assert summary.completed_tasks == 1
        assert summary.failed_tasks == 0
