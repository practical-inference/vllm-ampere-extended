# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unclosed tool-region leak tests for the GLM-4.7 parser.

When generation ends inside an unclosed `<tool_call>` region (length
cap), `validate_tool_names` never accepts the name, the call registers
nothing — and the whole turn used to vanish (empty content AND empty
tool_calls), poisoning the model's own history. The fix leaks the
swallowed region text into content so the turn is never empty."""

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.parser.glm47_moe import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    Glm47MoeParser,
)

WRITE_TOOL = ChatCompletionToolsParam(
    type="function",
    function={
        "name": "Write",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
        },
    },
)
TOOL_S, TOOL_E, USER = 1002, 1003, 1006
VOCAB = {TOOL_CALL_START: TOOL_S, TOOL_CALL_END: TOOL_E, "<|user|>": USER}
ARG = "<arg_key>file_path</arg_key><arg_value>/repo/app.py</arg_value>"
BODY = "def main():\n    print('hi')\n"


@pytest.fixture
def parser():
    return Glm47MoeParser(make_mock_tokenizer(VOCAB))


class TestUnclosedToolRegion:
    def test_truncated_invalid_name_leaks_to_content(self, parser, mock_request):
        mock_request.tools = [WRITE_TOOL]
        text = f"{TOOL_CALL_START}WriteNot\n{ARG}\n{BODY}"
        reasoning, content, tool_calls = parser.parse(text, mock_request)
        assert content is not None
        assert "WriteNot" in content
        assert "def main():" in content
        assert not tool_calls

    def test_truncated_valid_name_registers_call(self, parser, mock_request):
        mock_request.tools = [WRITE_TOOL]
        text = f"{TOOL_CALL_START}Write\n{ARG}"
        reasoning, content, tool_calls = parser.parse(text, mock_request)
        assert tool_calls and tool_calls[0].name == "Write"

    def test_closed_rejected_name_leaks(self, parser, mock_request):
        # call closes cleanly but the name is not a real tool: the turn
        # must leak the region text, not vanish
        mock_request.tools = [WRITE_TOOL]
        text = f"{TOOL_CALL_START}WriteNot\n{ARG}{TOOL_CALL_END}"
        reasoning, content, tool_calls = parser.parse(text, mock_request)
        assert not tool_calls
        assert content is not None
        assert "WriteNot" in content

    def test_mixed_turn_accepted_then_rejected_no_leak(self, parser, mock_request):
        # an accepted call followed by a rejected region: the turn-level
        # guard must not fire; content stays as before (no leaked text)
        mock_request.tools = [WRITE_TOOL]
        text = (
            f"{TOOL_CALL_START}Write\n<arg_key>file_path</arg_key>"
            "<arg_value>/a.py</arg_value>"
            f"{TOOL_CALL_END}{TOOL_CALL_START}Nope\n"
            "<arg_key>file_path</arg_key><arg_value>/b.py</arg_value>"
        )
        reasoning, content, tool_calls = parser.parse(text, mock_request)
        assert tool_calls and tool_calls[0].name == "Write"
        assert not content

    def test_streaming_mixed_turn_no_leak(self, parser, mock_request):
        # accepted call in an earlier chunk must suppress a later leak
        mock_request.tools = [WRITE_TOOL]
        chunks = [
            f"{TOOL_CALL_START}Write\n<arg_key>file_path</arg_key>"
            "<arg_value>/a.py</arg_value>"
            f"{TOOL_CALL_END}{TOOL_CALL_START}Nope\n"
            "<arg_key>file_path</arg_key><arg_value>/b",
            "<arg_value> unfinished",
        ]
        contents = []
        for i, chunk in enumerate(chunks):
            delta = parser.parse_delta(chunk, [], mock_request, finished=i == 1)
            contents.append((delta and delta.content) or "")
        assert not any(contents)

    def test_rejected_before_accepted_mixed_turn_leaks(self, parser, mock_request):
        # rejected region closes before the accepted call registers:
        # the leak fires (recoverable text instead of a swallowed turn)
        # and the accepted call still registers
        mock_request.tools = [WRITE_TOOL]
        text = (
            f"{TOOL_CALL_START}Nope\n<arg_key>file_path</arg_key>"
            "<arg_value>/b.py</arg_value>"
            f"{TOOL_CALL_END}{TOOL_CALL_START}Write\n"
            "<arg_key>file_path</arg_key><arg_value>/a.py</arg_value>"
            f"{TOOL_CALL_END}"
        )
        reasoning, content, tool_calls = parser.parse(text, mock_request)
        assert tool_calls and tool_calls[0].name == "Write"
        assert content is not None and "Nope" in content

    def test_closed_call_still_extracts(self, parser, mock_request):
        mock_request.tools = [WRITE_TOOL]
        text = f"{TOOL_CALL_START}Write\n{ARG}{TOOL_CALL_END}"
        reasoning, content, tool_calls = parser.parse(text, mock_request)
        assert tool_calls and tool_calls[0].name == "Write"
        assert not content
