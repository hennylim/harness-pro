"""
PromptBuilder – ILanguageProfile 을 받아 LLM system/user 프롬프트를 조립한다.

JSON 잘림 방지 전략
-------------------
문제: explanation 필드가 길어지면 code_block 에 도달하기 전에 토큰이 소진됨.
해결:
  1. explanation 을 한 문장(≤ 20 단어)으로 강제 제한.
  2. JSON 필드 순서를 code_block 이 마지막이 되도록 고정.
     → 토큰이 부족해도 target_file 과 explanation 은 완성되고
       code_block 만 잘리므로 복구 시도 가능.
  3. code_block 내부의 큰따옴표는 이스케이프하도록 명시.
"""
from __future__ import annotations

from agent.domain.language_profile import ILanguageProfile

# ── 플랜 생성 프롬프트 ────────────────────────────────────────────────────────

_PLAN_SKELETON = """\
You are an expert software architect.
Analyse the requirements and produce a JSON task plan.
Return ONLY a single JSON object matching this schema (no markdown, no commentary):
{{
  "tasks": [
    {{
      "task_id": <int>,
      "file_path": "<relative path, e.g. src/main{ext}>",
      "action": "<create|modify|delete|refactor>",
      "description": "<one sentence describing exactly what must be implemented>"
    }}
  ]
}}
Rules:
- task_id starts at 1 and is sequential.
- file_path must be a relative path, no leading slash, no "..".
- Use {lang_name} file extensions ({ext_hint}).
- List tasks in dependency order (dependencies first).
- Be specific: the description must be implementable without further context.
- Keep each description under 20 words.
"""

# ── 코드 생성 프롬프트 ────────────────────────────────────────────────────────
# CRITICAL: JSON 필드 순서는 반드시 아래 순서를 유지한다.
#   1. target_file  (짧음 – 항상 완성됨)
#   2. explanation  (한 문장 – 짧게 강제)
#   3. code_block   (마지막 – 가장 중요, 토큰이 부족하면 여기서 잘림)
#
# 이렇게 하면 code_block 이 잘려도 부분 복구가 가능하고,
# target_file 과 explanation 은 항상 완성된다.

_CODE_SKELETON = """\
You are a senior software engineer writing production-quality {lang_name} code.

CRITICAL JSON FORMAT RULES (violations cause immediate failure):
1. Return ONLY this JSON structure, in EXACTLY this field order:
{{
  "target_file": "<same relative path as the task>",
  "explanation": "<ONE sentence, max 15 words, describing what this file does>",
  "code_block": "<complete {lang_name} source code>"
}}
2. explanation MUST be ONE sentence, 15 words maximum. No exceptions.
3. code_block MUST contain the ENTIRE file from first line to last line.
4. NEVER truncate code_block. Output the complete file even if it is long.
5. Escape all double-quotes inside code_block as \\".
6. No trailing whitespace on any line of code_block.
7. Only import/source modules that are actually used.
8. Do NOT wrap code_block in markdown fences.
9. The last character of code_block must be a complete line (ending with \\n).

{lang_name} style rules:
{style_rules}
{header_hint}"""

_SUMMARISE_SYSTEM = """\
You are a technical writer summarising a code-generation run.
Return a concise (≤ 150 words) plain-text summary suitable for a Slack message.
"""


class PromptBuilder:
    """
    언어 프로필을 기반으로 LLM 프롬프트 문자열을 생성한다.

    Parameters
    ----------
    profile:
        현재 실행에 사용할 언어 프로필.
    """

    def __init__(self, profile: ILanguageProfile) -> None:
        self._profile = profile

    def build_plan_system(self) -> str:
        p = self._profile
        return _PLAN_SKELETON.format(
            lang_name=p.name,
            ext=next(iter(sorted(p.extensions))),
            ext_hint=p.plan_file_extension_hint,
        )

    def build_code_system(self) -> str:
        p = self._profile
        style_lines = "\n".join(
            f"- {line}" for line in p.code_style_rules.splitlines() if line.strip()
        )
        header_hint = (
            f"- Start file with this header:\n{p.file_header_hint}"
            if p.file_header_hint
            else ""
        )
        return _CODE_SKELETON.format(
            lang_name=p.name,
            style_rules=style_lines,
            header_hint=header_hint,
        )

    @staticmethod
    def build_summarise_system() -> str:
        return _SUMMARISE_SYSTEM
