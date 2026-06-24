"""
PromptBuilder – ILanguageProfile 을 받아 LLM system/user 프롬프트를 조립한다.

설계 원칙
---------
* LLM 어댑터(OpenAI, Gemini …)는 이 클래스를 사용한다.
* 언어 추가 시 이 파일은 수정하지 않는다 – 언어별 규칙은 ILanguageProfile 에 있다.
* 어댑터는 build_plan_system() / build_code_system() 을 호출해 프롬프트를 얻는다.
"""
from __future__ import annotations

from agent.domain.language_profile import ILanguageProfile

# ── 언어 무관 공통 뼈대 ───────────────────────────────────────────────────────

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
"""

_CODE_SKELETON = """\
You are a senior software engineer writing production-quality {lang_name} code.
Return ONLY a single JSON object (no markdown, no commentary):
{{
  "target_file": "<same relative path as the task>",
  "explanation": "<one paragraph: what this file does and key design decisions>",
  "code_block": "<complete, runnable {lang_name} source code as a single string>"
}}
Rules:
- code_block must contain the ENTIRE file, not just a snippet.
{style_rules}
- Do NOT wrap code_block in markdown fences.
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

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_plan_system(self) -> str:
        """플랜 생성 system 프롬프트 반환."""
        p = self._profile
        return _PLAN_SKELETON.format(
            lang_name=p.name,
            ext=next(iter(sorted(p.extensions))),   # 첫 번째 확장자 예시
            ext_hint=p.plan_file_extension_hint,
        )

    def build_code_system(self) -> str:
        """코드 생성 system 프롬프트 반환."""
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
        """진행 요약 system 프롬프트 반환 (언어 무관)."""
        return _SUMMARISE_SYSTEM
