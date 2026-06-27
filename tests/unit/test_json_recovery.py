"""
json_recovery 단위 테스트.

실제 로그에서 채취한 잘린 JSON 패턴으로 복구 로직을 검증한다.
"""
from __future__ import annotations

import json

import pytest

from agent.infrastructure.llm.json_recovery import (
    recover_code_patch_json,
    _unescape_json_string,
)


# ── 정상 케이스 ───────────────────────────────────────────────────────────────

class TestNormalParsing:
    def test_valid_json_returned_as_is(self):
        raw = json.dumps({
            "target_file": "src/main.sh",
            "explanation": "Entry point.",
            "code_block": "#!/usr/bin/env bash\necho hello\n",
        })
        result = recover_code_patch_json(raw)
        assert result["target_file"] == "src/main.sh"
        assert result["code_block"] == "#!/usr/bin/env bash\necho hello\n"

    def test_valid_json_with_escaped_content(self):
        raw = json.dumps({
            "target_file": "src/a.sh",
            "explanation": "Does stuff.",
            "code_block": '#!/usr/bin/env bash\necho "hello world"\n',
        })
        result = recover_code_patch_json(raw)
        assert '"hello world"' in result["code_block"]


# ── code_block 잘림 복구 ──────────────────────────────────────────────────────

class TestCodeBlockTruncation:
    def test_recovers_truncated_code_block(self):
        # 실제 로그 패턴: code_block 값이 중간에 잘림
        raw = (
            '{\n'
            '  "target_file": "src/engine.sh",\n'
            '  "explanation": "Core engine.",\n'
            '  "code_block": "#!/usr/bin/env bash\\nset -euo pipefail\\n\\n'
            '# Function to parse iptables rules\\nparse_rules() {\\n'
            '    local chain=\\"$1\\"\\n    iptab'  # ← 여기서 잘림
        )
        result = recover_code_patch_json(raw)
        assert result["target_file"] == "src/engine.sh"
        assert result["explanation"] == "Core engine."
        assert "#!/usr/bin/env bash" in result["code_block"]
        assert "parse_rules" in result["code_block"]

    def test_recovers_empty_code_block_when_field_missing(self):
        # code_block 필드 자체가 없는 경우
        raw = (
            '{\n'
            '  "target_file": "src/engine.sh",\n'
            '  "explanation": "Core engine that implements'  # explanation 도 잘림
        )
        result = recover_code_patch_json(raw)
        assert result["target_file"] == "src/engine.sh"
        assert result["code_block"] == ""

    def test_real_log_pattern_engine_sh(self):
        """실제 로그에서 채취한 잘린 JSON."""
        # 로그: char 566 에서 잘림, "src/uti" 에서 끊김
        raw = (
            '{\n'
            '  "target_file": "src/engine.sh",\n'
            '  "explanation": "This script implements the core logic for parsing '
            'iptables-save output and simulating packet flow through chains. It uses '
            'associative arrays to store rules and policies, and a recursive function '
            'to traverse chains based on matching criteria (source IP and destination '
            'port). Design decisions include using Bash regex for rule parsing and '
            'ensuring SC2155 compliance by separating declaration and assignment. '
            'The script also integrates with src/uti'
        )
        result = recover_code_patch_json(raw)
        assert result["target_file"] == "src/engine.sh"
        # explanation 이 잘렸고 code_block 없음 → 빈 code_block
        assert result["code_block"] == ""

    def test_real_log_pattern_with_code_started(self):
        """code_block 이 시작됐지만 잘린 경우."""
        raw = (
            '{\n'
            '  "target_file": "src/analyzer.sh",\n'
            '  "explanation": "Analyzes iptables rules.",\n'
            '  "code_block": "#!/usr/bin/env bash\\nset -euo pipefail\\n\\n'
            'source \\"$(dirname \\"$0\\")/utils.sh\\"\\n\\n'
            'analyze_packet() {\\n    local src_ip=\\"$1\\"\\n    local dst_port=\\"$2\\"\\n'
            '    echo \\"Analyzing $src_ip:$dst_port\\"\\n'
            '    iptables -L INPUT -n --line-numbers'  # ← 잘림
        )
        result = recover_code_patch_json(raw)
        assert result["target_file"] == "src/analyzer.sh"
        assert "#!/usr/bin/env bash" in result["code_block"]
        assert "analyze_packet" in result["code_block"]
        assert len(result["code_block"]) > 10


# ── explanation 잘림 복구 ────────────────────────────────────────────────────

class TestExplanationTruncation:
    def test_recovers_when_explanation_truncated(self):
        raw = (
            '{\n'
            '  "target_file": "src/main.sh",\n'
            '  "explanation": "Entry point that orchestrat'  # 잘림
        )
        result = recover_code_patch_json(raw)
        assert result["target_file"] == "src/main.sh"
        assert result["code_block"] == ""


# ── 복구 불가 케이스 ─────────────────────────────────────────────────────────

class TestUnrecoverable:
    def test_raises_on_garbage_input(self):
        with pytest.raises((json.JSONDecodeError, Exception)):
            recover_code_patch_json("not json at all")

    def test_raises_on_empty_string(self):
        with pytest.raises((json.JSONDecodeError, Exception)):
            recover_code_patch_json("")


# ── _unescape_json_string ─────────────────────────────────────────────────────

class TestUnescapeJsonString:
    def test_unescape_newline(self):
        assert _unescape_json_string("line1\\nline2") == "line1\nline2"

    def test_unescape_tab(self):
        assert _unescape_json_string("col1\\tcol2") == "col1\tcol2"

    def test_unescape_quote(self):
        assert _unescape_json_string('say \\"hello\\"') == 'say "hello"'

    def test_unescape_backslash(self):
        assert _unescape_json_string("path\\\\to") == "path\\to"

    def test_passthrough_normal(self):
        assert _unescape_json_string("hello world") == "hello world"


# ── Gemini 어댑터 통합: code_max_tokens 검증 ─────────────────────────────────

class TestGeminiAdapterTokens:
    def test_code_tokens_double_of_max(self):
        from unittest.mock import MagicMock, patch
        from agent.infrastructure.llm.gemini_adapter import GeminiAdapter

        with patch("google.genai.Client"):
            adapter = GeminiAdapter(api_key="test", max_tokens=4096)
        assert adapter._code_max_tokens == 8192

    def test_code_tokens_capped_at_32768(self):
        from unittest.mock import patch
        from agent.infrastructure.llm.gemini_adapter import GeminiAdapter

        with patch("google.genai.Client"):
            adapter = GeminiAdapter(api_key="test", max_tokens=20000)
        assert adapter._code_max_tokens == 32768
