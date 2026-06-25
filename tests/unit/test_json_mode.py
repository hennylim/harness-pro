"""
OpenAICompatibleAdapter JSON 모드 협상 단위 테스트.

실제 HTTP 호출 없이 OpenAI 클라이언트를 mock 합니다.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import httpx
import pytest
from openai import APIStatusError

from agent.domain.exceptions import LLMError, LLMParseError
from agent.infrastructure.llm.openai_adapter import (
    JsonMode,
    OpenAICompatibleAdapter,
    _strip_markdown_fence,
)
from config.logging import configure_logging

configure_logging(level="ERROR", fmt="console")


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _make_adapter(json_mode: JsonMode = JsonMode.AUTO) -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(
        model="test-model",
        base_url="http://localhost:1234/v1",
        api_key="test",
        json_mode=json_mode,
        max_retries=2,
    )


def _ok_response(content: str) -> MagicMock:
    """성공 응답 mock."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 20
    return resp


def _status_error(status: int, message: str) -> APIStatusError:
    req = httpx.Request("POST", "http://localhost/v1/chat/completions")
    resp = httpx.Response(status, json={"error": message}, request=req)
    return APIStatusError(message, response=resp, body={"error": message})


_PLAN_JSON = json.dumps({"tasks": [
    {"task_id": 1, "file_path": "src/a.py",
     "action": "create", "description": "test"}
]})

_CODE_JSON = json.dumps({
    "target_file": "src/a.py",
    "explanation": "ok",
    "code_block": "x = 1\n",
})


# ── _strip_markdown_fence ─────────────────────────────────────────────────────

class TestStripMarkdownFence:
    def test_strips_json_fence(self):
        assert _strip_markdown_fence("```json\n{}\n```") == "{}"

    def test_strips_plain_fence(self):
        assert _strip_markdown_fence("```\n{}\n```") == "{}"

    def test_passthrough_raw_json(self):
        assert _strip_markdown_fence('{"a": 1}') == '{"a": 1}'

    def test_strips_whitespace(self):
        assert _strip_markdown_fence("  ```json\n{}\n```  ") == "{}"


# ── AUTO 모드: json_object 성공 ───────────────────────────────────────────────

class TestAutoModeJsonObject:
    def test_selects_json_object_on_first_success(self):
        adapter = _make_adapter(JsonMode.AUTO)
        with patch.object(adapter._client.chat.completions, "create",
                          return_value=_ok_response(_PLAN_JSON)) as mock_create:
            plan = adapter.generate_plan("build something")

        assert adapter._json_mode == JsonMode.JSON_OBJECT
        assert len(plan.tasks) == 1
        # 협상 1회 + 이후 호출 없음 (이중 호출 없음)
        assert mock_create.call_count == 1

    def test_mode_cached_on_second_call(self):
        adapter = _make_adapter(JsonMode.AUTO)
        ok = _ok_response(_PLAN_JSON)
        with patch.object(adapter._client.chat.completions, "create",
                          return_value=ok) as mock_create:
            adapter.generate_plan("first")
            adapter.generate_plan("second")

        # 두 번 호출되지만 협상은 첫 번째 1회만
        assert mock_create.call_count == 2
        assert adapter._json_mode == JsonMode.JSON_OBJECT


# ── AUTO 모드: json_object 실패 → json_schema 폴백 ───────────────────────────

class TestAutoModeFallbackToJsonSchema:
    def test_falls_back_to_json_schema(self):
        adapter = _make_adapter(JsonMode.AUTO)

        def _side_effect(**kwargs):
            rf = kwargs.get("response_format", {})
            if rf.get("type") == "json_object":
                raise _status_error(
                    400,
                    "'response_format.type' must be 'json_schema' or 'text'"
                )
            return _ok_response(_PLAN_JSON)

        with patch.object(adapter._client.chat.completions, "create",
                          side_effect=_side_effect) as mock_create:
            plan = adapter.generate_plan("build something")

        assert adapter._json_mode == JsonMode.JSON_SCHEMA
        assert len(plan.tasks) == 1
        # json_object 탐색(실패) 1회 + json_schema 성공 1회 = 2회
        assert mock_create.call_count == 2

    def test_no_double_call_after_schema_selected(self):
        adapter = _make_adapter(JsonMode.AUTO)

        call_count = {"n": 0}

        def _side_effect(**kwargs):
            call_count["n"] += 1
            rf = kwargs.get("response_format", {})
            if rf.get("type") == "json_object":
                raise _status_error(400, "response_format unsupported")
            return _ok_response(_PLAN_JSON)

        with patch.object(adapter._client.chat.completions, "create",
                          side_effect=_side_effect):
            adapter.generate_plan("first")   # 협상: 2회 (1 fail + 1 ok)
            n_after_first = call_count["n"]
            adapter.generate_plan("second")  # 캐시된 json_schema 직접 사용: 1회

        assert n_after_first == 2
        assert call_count["n"] == 3   # 2 + 1


# ── AUTO 모드: json_schema 도 실패 → plain_prompt ────────────────────────────

class TestAutoModeFallbackToPlainPrompt:
    def test_falls_back_to_plain_prompt(self):
        adapter = _make_adapter(JsonMode.AUTO)

        def _side_effect(**kwargs):
            rf = kwargs.get("response_format", {})
            if rf and rf.get("type") in ("json_object", "json_schema"):
                raise _status_error(400, "response_format not supported")
            # plain_prompt: response_format 없음 → 성공
            return _ok_response(_PLAN_JSON)

        with patch.object(adapter._client.chat.completions, "create",
                          side_effect=_side_effect):
            plan = adapter.generate_plan("build something")

        assert adapter._json_mode == JsonMode.PLAIN_PROMPT
        assert len(plan.tasks) == 1

    def test_plain_prompt_strips_markdown_fence(self):
        adapter = _make_adapter(JsonMode.PLAIN_PROMPT)
        fenced = f"```json\n{_PLAN_JSON}\n```"

        with patch.object(adapter._client.chat.completions, "create",
                          return_value=_ok_response(fenced)):
            plan = adapter.generate_plan("build something")

        assert len(plan.tasks) == 1


# ── 강제 지정 모드 ────────────────────────────────────────────────────────────

class TestForcedMode:
    @pytest.mark.parametrize("mode", [
        JsonMode.JSON_OBJECT, JsonMode.JSON_SCHEMA, JsonMode.PLAIN_PROMPT
    ])
    def test_forced_mode_skips_negotiation(self, mode):
        adapter = _make_adapter(mode)

        call_log = []

        def _side_effect(**kwargs):
            call_log.append(kwargs.get("response_format"))
            return _ok_response(_PLAN_JSON)

        with patch.object(adapter._client.chat.completions, "create",
                          side_effect=_side_effect):
            adapter.generate_plan("build something")

        assert len(call_log) == 1
        # 모드는 협상 없이 유지
        assert adapter._json_mode == mode


# ── 에러 처리 ─────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_non_format_400_raises_immediately(self):
        adapter = _make_adapter(JsonMode.AUTO)

        with patch.object(adapter._client.chat.completions, "create",
                          side_effect=_status_error(400, "model not found")):
            with pytest.raises(LLMError, match="API error 400"):
                adapter.generate_plan("build something")

    def test_invalid_json_raises_parse_error(self):
        adapter = _make_adapter(JsonMode.JSON_OBJECT)

        with patch.object(adapter._client.chat.completions, "create",
                          return_value=_ok_response("not valid json")):
            with pytest.raises(LLMParseError):
                adapter.generate_plan("build something")

    def test_all_modes_exhausted_raises_llm_error(self):
        adapter = _make_adapter(JsonMode.AUTO)

        with patch.object(adapter._client.chat.completions, "create",
                          side_effect=_status_error(400, "response_format unsupported")):
            with pytest.raises(LLMError, match="No supported JSON mode"):
                adapter.generate_plan("build something")
