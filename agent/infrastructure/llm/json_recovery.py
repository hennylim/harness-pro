"""
json_recovery.py – 잘린 LLM JSON 응답에서 최대한 데이터를 복구한다.

Gemini / LM Studio 등이 max_output_tokens 한도에서 JSON 을 중간에 끊을 때,
code_block 값이 마지막 필드이면 다음 전략으로 복구를 시도한다:

전략 1 – 정상 파싱
  json.loads() 가 성공하면 그대로 반환.

전략 2 – code_block 잘림 복구
  "code_block": "..." 가 닫히지 않은 경우:
  a) code_block 값이 시작됐으면 끊긴 위치까지만 추출하고 "}}" 로 닫음.
  b) code_block 이 아예 없으면 빈 문자열로 채워 닫음.

전략 3 – explanation 잘림 복구
  explanation 이 닫히지 않은 경우:
  explanation 을 잘린 위치에서 강제로 닫고 code_block 을 빈 문자열로 채움.

복구된 응답은 LLMParseError 대신 경고 로그와 함께 반환된다.
code_block 이 비어 있는 복구 결과는 LLMParseError 를 발생시켜 재시도를 유도한다.
"""
from __future__ import annotations

import json
import re

import structlog

log = structlog.get_logger(__name__)


def recover_code_patch_json(raw: str) -> dict:
    """
    잘린 JSON 응답에서 CodePatch 딕셔너리를 복구한다.

    Parameters
    ----------
    raw:
        LLM 이 반환한 원본 문자열 (잘렸을 수 있음).

    Returns
    -------
    dict
        {"target_file": ..., "explanation": ..., "code_block": ...}

    Raises
    ------
    json.JSONDecodeError
        복구 불가능한 경우.
    """
    # ── 전략 1: 정상 파싱 ─────────────────────────────────────────────────────
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # ── 전략 2: code_block 잘림 복구 ──────────────────────────────────────────
    # target_file 과 explanation 이 완성됐는지 확인
    tf_match = re.search(r'"target_file"\s*:\s*"([^"]*)"', raw)
    ex_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', raw)

    if tf_match and ex_match:
        target_file = tf_match.group(1)
        explanation = ex_match.group(1)

        # code_block 필드가 시작됐는지 확인
        cb_start = re.search(r'"code_block"\s*:\s*"', raw)
        if cb_start:
            # code_block 값의 시작 위치
            cb_value_start = cb_start.end()
            # code_block 값: 닫는 따옴표 없이 끝까지 추출
            partial_code = raw[cb_value_start:]
            # 이스케이프 시퀀스 처리 (\n → 실제 개행, \" → ")
            partial_code = _unescape_json_string(partial_code)

            if partial_code.strip():
                log.warning(
                    "json_recovery.code_block_truncated",
                    target_file=target_file,
                    recovered_chars=len(partial_code),
                )
                return {
                    "target_file": target_file,
                    "explanation": explanation,
                    "code_block": partial_code,
                }

        # code_block 이 아예 없는 경우 → 빈 코드 반환 (재시도 유도)
        log.warning(
            "json_recovery.code_block_missing",
            target_file=target_file,
        )
        return {
            "target_file": target_file,
            "explanation": explanation,
            "code_block": "",
        }

    # ── 전략 3: explanation 도 잘린 경우 ──────────────────────────────────────
    if tf_match:
        target_file = tf_match.group(1)
        # explanation 값이 시작됐는지 확인
        ex_start = re.search(r'"explanation"\s*:\s*"', raw)
        if ex_start:
            partial_ex = raw[ex_start.end():]
            partial_ex = _unescape_json_string(partial_ex).strip()
            log.warning(
                "json_recovery.explanation_truncated",
                target_file=target_file,
            )
            return {
                "target_file": target_file,
                "explanation": partial_ex[:100],
                "code_block": "",   # 비어 있으면 재시도 유도
            }

    # ── 복구 불가 ─────────────────────────────────────────────────────────────
    log.error("json_recovery.unrecoverable", raw_preview=raw[:200])
    raise json.JSONDecodeError("Unrecoverable truncated JSON", raw, len(raw))


def _unescape_json_string(s: str) -> str:
    """
    JSON 문자열 값(따옴표로 닫히지 않은)의 이스케이프 시퀀스를 처리한다.
    \\n → 실제 개행, \\" → ", \\\\ → \\
    """
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == 'n':
                result.append('\n')
                i += 2
            elif nxt == 't':
                result.append('\t')
                i += 2
            elif nxt == '"':
                result.append('"')
                i += 2
            elif nxt == '\\':
                result.append('\\')
                i += 2
            elif nxt == 'r':
                result.append('\r')
                i += 2
            else:
                result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)
