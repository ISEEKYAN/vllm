# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from pathlib import Path

import pytest

from tests.tokenizers_.test_deepseek_v4 import FakeHfTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest
from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer

FIXTURES = Path(__file__).parent / "fixtures"


def render(messages, **kwargs):
    return get_deepseek_v41_tokenizer(FakeHfTokenizer()).apply_chat_template(
        messages, tokenize=False, **kwargs
    )


@pytest.mark.parametrize("case_id", [1, 2])
def test_reference_encoder_fixtures(case_id):
    # Expected prompts come from the DeepSeek V4.1 reference encoder.
    data = json.loads(
        (FIXTURES / "deepseek_v4" / f"test_input_{case_id}.json").read_text()
    )
    messages = data["messages"] if isinstance(data, dict) else data
    tools = data.get("tools") if isinstance(data, dict) else None
    expected = (FIXTURES / "deepseek_v41" / f"test_output_{case_id}.txt").read_text()
    assert render(messages, tools=tools) == expected


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        (None, 75),
        ("low", 50),
        ("high", 75),
        ("xhigh", 75),
        ("max", 100),
        (1, 1),
        (42, 42),
        (100, 100),
    ],
)
def test_numeric_reasoning_effort(effort, budget):
    assert render(
        [{"role": "user", "content": "question"}], reasoning_effort=effort
    ) == (
        "<｜begin▁of▁sentence｜><｜System｜>"
        f"Reasoning Effort: {budget} (range 1-100, the higher the value, "
        "the more thorough the reasoning)\n\n"
        "<｜User｜>question<｜Assistant｜><think>"
    )


@pytest.mark.parametrize("effort", ["medium", "minimal", -1, 0, 101, True, 1.5, []])
def test_invalid_effort_is_a_request_error(effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        render([{"role": "user", "content": "question"}], reasoning_effort=effort)


@pytest.mark.parametrize(
    "controls",
    [
        {"thinking": False},
        {"enable_thinking": False},
        {"thinking": True, "reasoning_effort": "none"},
        {"enable_thinking": True, "reasoning_effort": "none"},
    ],
)
def test_chat_mode_has_closed_thinking_prefix(controls):
    assert render([{"role": "user", "content": "question"}], **controls) == (
        "<｜begin▁of▁sentence｜><｜User｜>question<｜Assistant｜></think>"
    )


def test_raw_text_parts_preserve_reference_separator():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    ]
    original = copy.deepcopy(messages)
    assert (
        render(
            messages,
            conversation=[{"role": "user", "content": "first\nsecond"}],
            thinking=False,
        )
        == "<｜begin▁of▁sentence｜><｜User｜>first\n\nsecond<｜Assistant｜></think>"
    )
    assert messages == original


@pytest.mark.parametrize(
    ("role", "responses_type"),
    [("user", "input_text"), ("assistant", "output_text")],
)
def test_responses_text_parts_match_chat_text_parts(role, responses_type):
    responses_message = {
        "role": role,
        "content": [{"type": responses_type, "text": "hello"}],
    }
    chat_message = {
        "role": role,
        "content": [{"type": "text", "text": "hello"}],
    }
    assert render([responses_message], thinking=False) == render(
        [chat_message], thinking=False
    )


def test_mid_system_gets_its_own_marker_and_generation_header():
    assert render(
        [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "answer briefly"},
        ],
        thinking=False,
    ) == (
        "<｜begin▁of▁sentence｜><｜User｜>question"
        "<｜System｜>answer briefly<｜Assistant｜></think>"
    )


def test_top_level_effort_overrides_template_effort():
    request = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        messages=[{"role": "user", "content": "question"}],
        reasoning_effort="low",
        chat_template_kwargs={"reasoning_effort": 100},
    )
    kwargs = request.build_chat_params(None, "auto").get_apply_chat_template_kwargs()
    assert "Reasoning Effort: 50 " in render(request.messages, **kwargs)


def test_encode_uses_one_bos_and_forwards_truncation():
    tokenizer = get_deepseek_v41_tokenizer(FakeHfTokenizer())
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "question"}],
        thinking=False,
        truncation=True,
        max_length=16,
    )
    text, add_special_tokens, kwargs = tokenizer.last_encode
    assert text.startswith("<｜begin▁of▁sentence｜>")
    assert add_special_tokens is False
    assert kwargs == {"truncation": True, "max_length": 16}


@pytest.mark.parametrize("image_type", ["image_url", "input_image", "image_pil"])
def test_images_preserve_content_order_and_reference_separator(image_type):
    from PIL import Image

    def image_part(color):
        if image_type == "image_pil":
            return {"type": image_type, "image_pil": Image.new("RGB", (2, 2), color)}
        url = f"https://example.com/{color}.png"
        return {
            "type": image_type,
            "image_url": {"url": url} if image_type == "image_url" else url,
        }

    parts = [
        {"type": "text", "text": "first"},
        image_part("red"),
        {"type": "text", "text": "second"},
        image_part("blue"),
        {"type": "text", "text": "compare"},
    ]
    assert render([{"role": "user", "content": parts}], thinking=False) == (
        "<｜begin▁of▁sentence｜><｜User｜>first\n\n<｜deepseek_image｜>\n\n"
        "second\n\n<｜deepseek_image｜>\n\ncompare<｜Assistant｜></think>"
    )
    assert parts[1]["type"] == parts[3]["type"] == image_type


def test_unsupported_media_is_a_request_error():
    with pytest.raises(ValueError, match="text and image content only"):
        render([{"role": "user", "content": [{"type": "input_audio"}]}])


@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content", "both"])
@pytest.mark.parametrize("tool_calls_type", [list, tuple, iter])
def test_tokenize_reasoning_tool_calls_match_chat(reasoning_field, tool_calls_type):
    def payload():
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_type(
                [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calculate", "arguments": '{"value": 2}'},
                    }
                ]
            ),
        }
        if reasoning_field in ("reasoning", "both"):
            assistant["reasoning"] = "I should calculate first."
        if reasoning_field in ("reasoning_content", "both"):
            assistant["reasoning_content"] = (
                "deprecated reasoning"
                if reasoning_field == "both"
                else "I should calculate first."
            )
        return {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Calculate two."},
                assistant,
                {"role": "tool", "tool_call_id": "call_1", "content": "2"},
            ],
        }

    token_request = TokenizeChatRequest.model_validate(payload())
    chat_request = ChatCompletionRequest.model_validate(payload())
    assert token_request.messages[1]["reasoning"] == "I should calculate first."
    assert isinstance(token_request.messages[1]["tool_calls"], list)
    assert "reasoning_content" not in token_request.messages[1]
    assert render(token_request.messages) == render(chat_request.messages)
    tokenizer = get_deepseek_v41_tokenizer(FakeHfTokenizer())
    assert tokenizer.apply_chat_template(
        token_request.messages
    ) == tokenizer.apply_chat_template(chat_request.messages)
    assert (
        json.loads(token_request.model_dump_json())["messages"]
        == (json.loads(chat_request.model_dump_json())["messages"])
    )


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
    ],
)
def test_tokenize_existing_text_messages_unchanged(messages):
    original = copy.deepcopy(messages)
    request = TokenizeChatRequest(messages=copy.deepcopy(messages))
    assert request.messages == original
    assert render(request.messages) == render(original)
