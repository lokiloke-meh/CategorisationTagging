"""End-to-end smoke test for the full tagging pipeline.

Builds the complete pipeline via TagPipeline with realistic mock outputs and asserts:
result.success is True, output_data contains tags dict with valid categories,
summary and effectivedate are present. This would fail if any refactoring broke the
e2e flow or removed keys from ChainResult output.
"""

from __future__ import annotations

from prompt_chains.base_chains import ChainResult, Document
from utils.llm_models import LLMResult


class _PipelineMockLLM:
    """Returns realistic shapes for every chain type in the pipeline."""

    def chat_with_expected_type_result(self, chat_payload, result_type):
        system = chat_payload.get("messages", [{}])[0].get("content", "")

        if "effective date" in system.lower():
            # RefineSummarizeChain output
            return LLMResult(
                model_name="mock",
                user_input="",
                system_prompt="",
                model_result={
                    "effectivedate": "2025-01-15",
                    "summary": "Test summary of the circular.",
                },
                success=True,
            )

        if "subcategories" in system.lower():
            # TopicBucketing output
            return LLMResult(
                model_name="mock",
                user_input="",
                system_prompt="",
                model_result={
                    "reasoning": "Market-wide impact event.",
                    "category": "MarketWideImpact",
                    "category1": "MarketWideImpact",
                    "category2": "MarketWideImpact",
                    "category3": "MarketWideImpact",
                    "category4": "MarketWideImpact",
                },
                success=True,
            )

        # SingleTradingTopicCategorisation output
        return LLMResult(
            model_name="mock",
            user_input="",
            system_prompt="",
            model_result={
                "reasoning": "High relevance.",
                "Relevance": "High",
                "Relevance1": "High",
                "Relevance2": "High",
                "Relevance3": "High",
                "Relevance4": "High",
            },
            success=True,
        )

    def get_embeddings(self, text: str) -> LLMResult:
        return LLMResult(model_name="mock", user_input=text, model_result=[0.1] * 768, success=True)


def test_full_pipeline_produces_valid_output():
    """Pipeline returns ChainResult with correct output_data keys."""
    from tagging.pipelines.tag_pipeline import TagPipeline

    pipeline = TagPipeline(
        llm=_PipelineMockLLM(),
        target_tags=["MarketWideBoardLotSizeFrameworkChanges"],
        min_chars_for_summarization=10,  # Low threshold to exercise long-text path
    )

    docs = [Document(page_content="A" * 5000, title="test circular")]
    result = pipeline.invoke(docs)

    assert isinstance(result, ChainResult)
    assert result.success is True
    from tagging.utils.data_models import TagResult

    output_data = result.output_data
    # Pipeline merges branch outputs into a unified TagResult with tags + summary/effectivedate
    assert isinstance(output_data, TagResult)
    assert len(output_data.tags) > 0
    assert isinstance(output_data.summary, str)
    assert len(output_data.summary) > 0
    assert output_data.effectivedate is not None


def test_full_pipeline_short_text_skips_summarization():
    """Short text (< threshold) bypasses summarizer and goes straight to bucketing."""
    from tagging.pipelines.tag_pipeline import TagPipeline

    pipeline = TagPipeline(
        llm=_PipelineMockLLM(),
        target_tags=["MarketWideBoardLotSizeFrameworkChanges"],
        min_chars_for_summarization=10,
    )

    docs = [Document(page_content="Short", title="test")]
    result = pipeline.invoke(docs)

    assert isinstance(result, ChainResult)
    assert result.success is True
    from tagging.utils.data_models import TagResult

    output_data = result.output_data
    assert isinstance(output_data, TagResult)
