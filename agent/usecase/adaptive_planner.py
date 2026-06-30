"""
adaptive_planner.py – 파일 복잡도를 예측해 큰 파일을 섹션으로 자동 분할하는 플래너.

핵심 전략
---------
1. 강화 계획 생성 (generate_enhanced_plan)
   - LLM 에게 각 파일의 예상 줄 수와 섹션 분할 계획을 요청
   - 예상 줄 수 > chunk_threshold_lines 인 파일은 ChunkedTask 로 분류
   - 단순한 파일(헤더, 설정 등)은 일반 Task 로 유지

2. 섹션별 생성 (generate_section)
   - 각 섹션마다 이전 섹션 코드를 컨텍스트로 전달
   - 섹션은 컴파일 가능한 단위(함수 전체, 구조체 정의 등)로 분할
   - 섹션 경계에서 잘림이 발생해도 다음 섹션에서 이어받음

3. 맥락 유지 (context accumulation)
   - 생성된 섹션 코드를 누적해 다음 섹션에 전달
   - "이미 작성된 코드" 로 처리해 중복/불일치 방지
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog
from pydantic import ValidationError

from agent.domain.entities import (
    ChunkedTask,
    EnhancedProjectPlan,
    FileSection,
    ProjectPlan,
    Task,
    TaskAction,
)
from agent.domain.exceptions import LLMParseError
from agent.domain.interfaces import IAdaptivePlanner, ILLMAdapter
from agent.infrastructure.llm.json_recovery import recover_code_patch_json

log = structlog.get_logger(__name__)

# ── 강화 계획 생성 프롬프트 ────────────────────────────────────────────────────

_ENHANCED_PLAN_SYSTEM = """You are an expert software architect specializing in \
planning code generation for small LLMs.

Analyze the requirements and produce a detailed JSON plan.
Return ONLY a valid JSON object with NO markdown, NO commentary:
{{
  "files": [
    {{
      "task_id": <int>,
      "file_path": "<relative path>",
      "action": "<create|modify|delete>",
      "description": "<max 20 words>",
      "estimated_lines": <int>,
      "complexity": "<simple|medium|complex>",
      "sections": [
        {{
          "section_index": <int>,
          "section_name": "<imports|types|constants|functions|main|etc>",
          "description": "<max 20 words what this section contains>",
          "depends_on": [<section_index>, ...]
        }}
      ]
    }}
  ]
}}

Rules:
- task_id: sequential starting from 1.
- file_path: relative path, no leading slash, no "..".
- List files in dependency order (dependencies first).
- estimated_lines: your best estimate of source lines of code.
- complexity: simple(<50 lines), medium(50-150), complex(>150).
- sections: REQUIRED for complex files. Split by logical unit:
    * Header files: include_guards, type_definitions, function_declarations
    * C/C++ source: includes_globals, data_structures, helper_functions, core_logic, main_function
    * Python: imports, classes/types, helper_functions, main_logic, entry_point
    * Bash: header_utils, validation_functions, core_functions, main_handler
- sections for simple/medium files: single section covering the whole file.
- Keep each section implementable in {threshold} lines or less.
"""

# ── 섹션 생성 프롬프트 ────────────────────────────────────────────────────────

_SECTION_SYSTEM = """You are a senior software engineer generating ONE section of \
a larger source file.

CRITICAL: You are generating ONLY the section described below.
The surrounding context (previous sections) is provided for reference.

Return ONLY a JSON object:
{{
  "section_code": "<complete code for THIS section only>"
}}

Rules:
1. section_code must be COMPLETE and COMPILABLE for the described section.
2. Do NOT repeat code from previous sections.
3. Do NOT add file headers/includes that belong to other sections.
4. section_code ends with a newline character.
5. Escape double-quotes as \\".
6. No trailing whitespace on any line.
7. section_code must be syntactically complete (no dangling brackets/quotes).
"""


class AdaptivePlanner(IAdaptivePlanner):
    """
    LLM 을 사용해 요구사항을 세분화된 계획으로 변환하고
    큰 파일을 섹션별로 생성하는 강화 플래너.

    Parameters
    ----------
    llm_adapter:
        실제 LLM 호출을 담당하는 어댑터.
    chunk_threshold_lines:
        이 줄 수 이상으로 예상되는 파일은 청크 분할.
    section_max_lines:
        각 섹션의 최대 목표 줄 수.
    """

    def __init__(
        self,
        llm_adapter: ILLMAdapter,
        chunk_threshold_lines: int = 80,
        section_max_lines: int = 60,
    ) -> None:
        self._llm = llm_adapter
        self._threshold = chunk_threshold_lines
        self._section_max = section_max_lines

    # ── IAdaptivePlanner ──────────────────────────────────────────────────────

    def generate_enhanced_plan(
        self,
        requirements: str,
        chunk_threshold_lines: int,
    ) -> EnhancedProjectPlan:
        """
        요구사항 → 세분화된 계획 생성.

        complex 파일은 ChunkedTask, simple/medium 은 Task 로 분류.
        """
        log.info("adaptive_planner.plan.start", threshold=chunk_threshold_lines)

        system = _ENHANCED_PLAN_SYSTEM.format(
            threshold=chunk_threshold_lines
        )
        user = f"Requirements:\n{requirements}"

        raw = self._call_llm_plan(system, user)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMParseError(
                f"강화 계획 JSON 파싱 실패: {exc}\nRaw:\n{raw[:500]}"
            ) from exc

        files = data.get("files", [])
        if not files:
            raise LLMParseError("강화 계획에 파일 목록이 없습니다.")

        normal_tasks: list[Task] = []
        chunked_tasks: list[ChunkedTask] = []

        for item in files:
            task_id = item.get("task_id", 0)
            file_path = item.get("file_path", "")
            action_str = item.get("action", "create")
            description = item.get("description", "")
            estimated_lines = item.get("estimated_lines", 50)
            complexity = item.get("complexity", "simple")
            sections_data = item.get("sections", [])

            try:
                action = TaskAction(action_str)
            except ValueError:
                action = TaskAction.CREATE

            # 복잡한 파일 또는 예상 줄 수 초과 → ChunkedTask
            is_chunked = (
                complexity == "complex"
                or estimated_lines > chunk_threshold_lines
                or len(sections_data) > 1
            )

            if is_chunked and sections_data:
                sections = [
                    FileSection(
                        section_index=s.get("section_index", i),
                        section_name=s.get("section_name", f"section_{i}"),
                        description=s.get("description", ""),
                        depends_on=s.get("depends_on", []),
                    )
                    for i, s in enumerate(sections_data)
                ]
                chunked_tasks.append(ChunkedTask(
                    task_id=task_id,
                    file_path=file_path,
                    action=action,
                    description=description,
                    sections=sections,
                ))
                log.info(
                    "adaptive_planner.task.chunked",
                    file=file_path,
                    sections=len(sections),
                    estimated_lines=estimated_lines,
                )
            else:
                # 단일 섹션 → 일반 Task
                normal_tasks.append(Task(
                    task_id=task_id,
                    file_path=file_path,
                    action=action,
                    description=description,
                ))
                log.info(
                    "adaptive_planner.task.normal",
                    file=file_path,
                    estimated_lines=estimated_lines,
                )

        plan = EnhancedProjectPlan(
            tasks=normal_tasks,
            chunked_tasks=chunked_tasks,
        )
        log.info(
            "adaptive_planner.plan.ready",
            normal=len(normal_tasks),
            chunked=len(chunked_tasks),
            total=plan.all_tasks_count,
        )
        return plan

    def generate_section(
        self,
        file_path: str,
        section: FileSection,
        previous_sections: list[FileSection],
        workspace_skeleton: str,
        error_feedback: str,
        language_name: str,
    ) -> str:
        """
        청크 파일의 한 섹션을 생성한다.
        이전 섹션 코드를 컨텍스트로 포함해 맥락을 유지한다.
        """
        log.info(
            "adaptive_planner.section.generate",
            file=file_path,
            section=section.section_name,
            index=section.section_index,
        )

        # 이전 섹션 코드 누적
        prev_context = self._build_previous_context(previous_sections)

        user_parts = [
            f"File: {file_path}",
            f"Language: {language_name}",
            f"Section {section.section_index}: {section.section_name}",
            f"Description: {section.description}",
        ]
        if prev_context:
            user_parts.append(
                f"\n=== Already written code (previous sections) ===\n"
                f"{prev_context}\n"
                "=== End of previous sections ==="
            )
        if workspace_skeleton:
            user_parts.append(
                f"\n=== Workspace context ===\n{workspace_skeleton}\n"
                "=== End workspace ==="
            )
        if error_feedback:
            user_parts.append(
                f"\n⚠️  Previous attempt failed. Fix these issues:\n{error_feedback}"
            )

        user_parts.append(
            f"\nNow write ONLY the '{section.section_name}' section code."
        )

        raw = self._call_llm_section("\n".join(user_parts))

        try:
            data = json.loads(raw)
            code = data.get("section_code", "")
        except json.JSONDecodeError:
            # 복구 시도: JSON 아닌 순수 코드로 반환된 경우
            try:
                recovered = recover_code_patch_json(raw)
                code = recovered.get("code_block", raw)
            except Exception:
                code = raw

        if not code.strip():
            raise LLMParseError(
                f"섹션 '{section.section_name}' 코드가 비어 있습니다."
            )

        return code

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call_llm_plan(self, system: str, user: str) -> str:
        """LLM 어댑터의 generate_plan 을 직접 호출하되 raw JSON 반환."""
        # ILLMAdapter.generate_plan 은 ProjectPlan 을 반환하므로
        # 강화 계획을 위해 내부적으로 generate_code 를 재활용하거나
        # 어댑터의 내부 _call 메서드를 사용한다.
        # 여기서는 어댑터 유형에 따라 분기한다.
        from agent.infrastructure.llm.gemini_adapter import GeminiAdapter
        from agent.infrastructure.llm.openai_adapter import OpenAICompatibleAdapter

        if isinstance(self._llm, GeminiAdapter):
            return self._llm._call(
                system=system,
                user=user,
                max_tokens=self._llm._max_tokens,
            )
        elif isinstance(self._llm, OpenAICompatibleAdapter):
            return self._llm._call(system=system, user=user)
        else:
            # 폴백: generate_plan 의 결과를 JSON 으로 직렬화
            plan = self._llm.generate_plan(user)
            return plan.model_dump_json()

    def _call_llm_section(self, user: str) -> str:
        """섹션 생성 LLM 호출."""
        from agent.infrastructure.llm.gemini_adapter import GeminiAdapter
        from agent.infrastructure.llm.openai_adapter import OpenAICompatibleAdapter

        if isinstance(self._llm, GeminiAdapter):
            return self._llm._call(
                system=_SECTION_SYSTEM,
                user=user,
                max_tokens=self._llm._code_max_tokens,
            )
        elif isinstance(self._llm, OpenAICompatibleAdapter):
            return self._llm._call(
                system=_SECTION_SYSTEM,
                user=user,
                override_max_tokens=self._llm._code_max_tokens,
            )
        else:
            # 폴백: generate_code 를 흉내 낸 Task 로 호출
            from agent.domain.entities import Task, TaskAction
            fake_task = Task(
                task_id=0,
                file_path="section",
                action=TaskAction.CREATE,
                description=user[:200],
            )
            patch = self._llm.generate_code(
                task=fake_task,
                memory_summary="",
                workspace_skeleton="",
                error_feedback="",
            )
            return json.dumps({"section_code": patch.code_block})

    @staticmethod
    def _build_previous_context(sections: list[FileSection]) -> str:
        """이전 섹션 코드를 하나의 문자열로 조립."""
        if not sections:
            return ""
        parts = []
        for s in sections:
            if s.generated_code.strip():
                parts.append(
                    f"/* --- Section {s.section_index}: {s.section_name} --- */\n"
                    f"{s.generated_code}"
                )
        return "\n".join(parts)
