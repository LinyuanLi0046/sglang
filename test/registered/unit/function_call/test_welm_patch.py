import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def tools():
    return [Tool(function=Function(name="weather", parameters={"type": "object"}))]


@pytest.mark.parametrize("forward", [False, True])
def test_unknown_tool_forwarding_stream_and_nonstream(tools, forward):
    text = (
        '<tool_call>\n{"name":"unregistered","arguments":{"city":"北京"}}\n</tool_call>'
    )
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(forward):
        parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")
        _, calls = parser.parse_non_stream(text)
        assert len(calls) == int(forward)
        if forward:
            assert calls[0].name == "unregistered"
            assert calls[0].tool_index == -1
            assert json.loads(calls[0].parameters) == {"city": "北京"}

        streamed = []
        for char in text:
            _, delta = parser.parse_stream_chunk(char)
            streamed.extend(delta)
        if forward:
            assert [call.name for call in streamed if call.name] == ["unregistered"]
            assert {call.tool_index for call in streamed} == {0}
            assert json.loads("".join(call.parameters for call in streamed)) == {
                "city": "北京"
            }
        else:
            assert streamed == []


def test_stream_mixed_known_and_unknown_tools(tools):
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")
        calls = {}
        for name in ("weather", "unregistered", "weather"):
            text = f'<tool_call>\n{{"name":"{name}","arguments":{{"city":"北京"}}}}\n</tool_call>\n'
            for char in text:
                _, delta = parser.parse_stream_chunk(char)
                for call in delta:
                    entry = calls.setdefault(call.tool_index, ["", ""])
                    entry[0] += call.name or ""
                    entry[1] += call.parameters
        assert [calls[i][0] for i in range(3)] == ["weather", "unregistered", "weather"]
        assert all(json.loads(args) == {"city": "北京"} for _, args in calls.values())


@pytest.mark.parametrize(
    "payload",
    [
        "{'name': 'weather', 'arguments': {'city': '北京'}}",
        '{"name":"weather","arguments":{"city":"北京",}}',
        '{"name":"weather","arguments":{"city":"北京"}',
    ],
)
def test_repair_malformed_nonstream_json(tools, payload):
    result = Qwen25Detector().detect_and_parse(
        f"<tool_call>\n{payload}\n</tool_call>", tools
    )
    assert len(result.calls) == 1
    assert result.calls[0].name == "weather"
    assert json.loads(result.calls[0].parameters) == {"city": "北京"}


def test_unrepairable_block_does_not_hide_following_call(tools):
    text = (
        "<tool_call>\nnot a tool object\n</tool_call>\n"
        '<tool_call>\n{"name":"weather","arguments":{}}\n</tool_call>'
    )
    result = Qwen25Detector().detect_and_parse(text, tools)
    assert [call.name for call in result.calls] == ["weather"]


def test_repair_respects_unknown_tool_setting(tools):
    text = "<tool_call>\n{'name':'unregistered','arguments':{}}\n</tool_call>"
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        assert not Qwen25Detector().detect_and_parse(text, tools).calls
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        assert (
            Qwen25Detector().detect_and_parse(text, tools).calls[0].name
            == "unregistered"
        )


def test_registry_retains_structural_tag_interfaces(tools):
    # Importing the registry also imports GLM47, which requires these exports.
    from sglang.srt.function_call.base_format_detector import (
        StructuralTag,
        get_model_structural_tag,
    )

    assert StructuralTag is not None
    assert get_model_structural_tag is None or callable(get_model_structural_tag)
    parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")
    assert parser.get_structure_constraint("auto") is None
    constraint_type, constraint = parser.get_structure_constraint("required")
    assert constraint_type == "structural_tag"
    assert constraint.at_least_one
    assert not parser.detector.parses_required_natively()
