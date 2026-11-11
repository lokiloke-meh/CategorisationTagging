from unittest.mock import MagicMock, patch

import pytest

from utils.llm_models import LLMResult
from utils.myllamacpp import llamaCppChat


@pytest.fixture()
def chat():
    return llamaCppChat(model="gemma-4-12b-it-qat-q4_0")


class TestLlamaCppChatBehavior:
    """Tests that assert observable behavior — what a user sees.

    A test that asserts which HTTP params were sent, or restates mock data,
    is deleted per principle-test-behavior-not-implementation.
    """

    def test_successful_chat_returns_result_and_metrics(self, chat):
        with (
            patch("utils.myllamacpp.requests.post") as mock_post,
            patch.object(LLMResult, "to_otel_attributes", return_value={}),
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                _content=b'{"choices":[{"message":{"role":"assistant","content":"Quantum computing is cool"}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15},"timings":{"predicted_per_second":65.0,"prompt_ms":800}}',
            )

            result = chat.chat_with_expected_type_result(
                chat_payload={
                    "messages": [
                        {"role": "system", "content": "Be helpful"},
                        {"role": "user", "content": "What is quantum computing?"},
                    ],
                    "temperature": 0.7,
                },
                result_type="string",
            )

        assert result.success is True
        assert result.model_result == "Quantum computing is cool"
        assert result.prompt_token_count == 10
        assert result.completion_token_count == 5
        assert result.total_token_count == 15
        assert result.response_tokens_per_second == pytest.approx(65.0)
        assert result.prompt_eval_duration_secs == pytest.approx(0.8, abs=0.01)

    def test_json_result_type_parses_content(self, chat):
        with (
            patch("utils.myllamacpp.requests.post") as mock_post,
            patch.object(LLMResult, "to_otel_attributes", return_value={}),
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                _content=b'{"choices":[{"message":{"role":"assistant","content":"{\\"key\\": \\"value\\"}"}}],"usage":{}}',
            )

            result = chat.chat_with_expected_type_result(
                chat_payload={
                    "messages": [{"role": "user", "content": "test"}],
                    "response_format": {"type": "json_object"},
                },
                result_type="dict",
            )

        assert result.success is True
        assert result.model_result == {"key": "value"}

    def test_json_without_response_format_raises(self, chat):
        with pytest.raises(OSError, match="Expected to pass response_format"):
            chat.chat_with_expected_type_result(
                chat_payload={"messages": [{"role": "user", "content": "test"}]},
                result_type="dict",
            )

    def test_ollama_format_key_is_normalized(self, chat):
        with (
            patch("utils.myllamacpp.requests.post") as mock_post,
            patch.object(LLMResult, "to_otel_attributes", return_value={}),
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                _content=b'{"choices":[{"message":{"role":"assistant","content":"{\"key\": \\"value\\"}"}}],"usage":{}}',
            )

            schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            chat.chat_with_expected_type_result(
                chat_payload={
                    "messages": [{"role": "user", "content": "test"}],
                    "format": schema,
                },
                result_type="dict",
            )

        # Verify the normalized payload was sent to the API
        call_args = mock_post.call_args
        sent_payload = call_args.kwargs["json"]
        assert sent_payload.get("response_format") == {
            "type": "json_object",
            "schema": schema,
        }
        assert "format" not in sent_payload  # original key removed

    def test_invalid_result_type_raises(self, chat):
        with pytest.raises(ValueError, match="Not supported result_type"):
            chat.chat_with_expected_type_result(
                chat_payload={"messages": [{"role": "user", "content": "test"}]},
                result_type="blob",
            )

    def test_non_200_response_returns_failure(self, chat):
        with (
            patch.object(LLMResult, "to_otel_attributes", return_value={}),
            patch("utils.myllamacpp.requests.post") as mock_post,
        ):
            mock_post.return_value = MagicMock(status_code=500)

            result = chat.chat_with_expected_type_result(
                chat_payload={"messages": [{"role": "user", "content": "test"}]},
                result_type="string",
            )

        assert result.success is False
        assert "HTTPError 500" in result.error_message

    def test_connection_error_returns_failure(self, chat):
        with (
            patch.object(LLMResult, "to_otel_attributes", return_value={}),
            patch("utils.myllamacpp.requests.post") as mock_post,
        ):
            mock_post.side_effect = Exception("Connection refused")

            result = chat.chat_with_expected_type_result(
                chat_payload={"messages": [{"role": "user", "content": "test"}]},
                result_type="string",
            )

        assert result.success is False
        assert "Connection refused" in result.error_message


@pytest.mark.integration
def test_json_result_parses_real_categorisation_schema():
    """Real integration test against live llama.cpp server: TopicBucketing pattern.

    Mirrors the real usage in tagging/independent_category_tagging.py (TopicBucketing):
    - Literal[*target_buckets] constrains category to one of predefined options
    - response_format.schema enforces exact JSON shape at the server level
    - Pydantic validates both structure AND value constraints
    """
    from typing import Literal

    import pydantic

    target_categories = [
        "MarketWideBoardLotSizeFrameworkChanges",
        "IPAddressChangesbytheExchange",
        "MockDRDisasterRecoverySession",
    ]

    TagsModel = pydantic.create_model(
        "TagsModel",
        **{
            tag: (Literal["Low", "Medium", "High"], ...)
            for tag in target_categories
        }
    )

    ResponseModel = pydantic.create_model("ResponseModel", tags=TagsModel)

    chat = llamaCppChat(model="gemma-4-12b-it-qat-q4_0")

    result = chat.chat_with_expected_type_result(
        chat_payload={
            "messages": [
                {"role": "system", "content": (
                    "You are in charge of a trading system. Categorize the relevance of each topic into one of: Low, Medium, or High."
                )},
                {"role": "user", "content": (
                    "The exchange is changing network IP addresses for their matching engine. "
                    "Board lot size framework changes are also being proposed."  
                )},
            ],
            "temperature": 0.2,
            "format": ResponseModel.model_json_schema(),
        },
        result_type="dict",
    )

    assert result.success is True, f"Request failed: {result.error_message}"

    # Validate against the real Pydantic model used by TopicBucketing chains
    try:
        validated = ResponseModel(**result.model_result)
    except pydantic.ValidationError as e:
        pytest.fail(f"Model result does not match expected schema:\n{e}\nRaw: {result.model_result}")

    # Assert valid category values — each tag must be one of Low/Medium/High
    for tag in target_categories:
        assert validated.tags.__dict__[tag] in ("Low", "Medium", "High")
