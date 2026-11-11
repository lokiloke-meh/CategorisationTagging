"""Specialized pipeline for DecisionModel categorisation + ChatAdapter summarisation.

Inherits from TagPipeline but overrides _build_chain() to route adapters:
  - DecisionModel → categorisation paths (TopicBucketing, TradingTopicsDistiller)
  - ChatAdapter   → all generative paths (summarisers — DecisionModel has no generative output)
"""

from __future__ import annotations

from prompt_chains.base_chains import BaseChain
from prompt_chains.long_text_summariser import (
    MapReduceSummarizeChain,
    RefineSummarizeChain,
    SentenceTextSplitter,
)
from prompt_chains.routing_chains_bool import (
    BoolRoute,
    BoolRouterChain,
    TextLengthCondition,
)
from prompt_chains.routing_chains_categorical import CategoricalRouterChain
from prompt_chains.sequential_chains import SequentialChain
from tagging.category_tagging import DistillerAdapter
from tagging.pipelines.tag_pipeline import (
    BucketingCondition,
    NoOpPassThrough,
    TagPipeline,
)
from utils.llm_models import ChatAdapter, DecisionModel


class TagPipelineDecisionModel(TagPipeline):
    """TagPipeline optimized for mixed-adapter usage.

    Parameters:
        llm: ChatAdapter required for all generative output (summarisation).
        decision_model: DecisionModel used for categorisation tasks (structured decisions).
        All other params inherited from TagPipeline.
    """

    def __init__(
        self,
        llm: ChatAdapter,
        decision_model: DecisionModel,
        target_tags: list[str],
        target_tag_examples: dict[str, str] | None = None,
        evaluators: list[object] | None = None,
        text_chunk_size: int = 800,
        min_chars_for_summarization: int = 4000,
        impact_categories: list[str] | None = None,
        prompt_samples: dict[str, str] | None = None,
        summarizer_chain: type[
            RefineSummarizeChain | MapReduceSummarizeChain
        ] = RefineSummarizeChain,
        category_tag_config: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            target_tags=target_tags,
            target_tag_examples=target_tag_examples,
            evaluators=evaluators,
            text_chunk_size=text_chunk_size,
            min_chars_for_summarization=min_chars_for_summarization,
            impact_categories=impact_categories,
            prompt_samples=prompt_samples,
            summarizer_chain=summarizer_chain,
            category_tag_config=category_tag_config,
        )
        self.decision_model = decision_model

    def _build_category_destinations_mixed(self) -> dict[str, object]:
        """Build destinations with decision_model for categorisation + llm for summarisation."""
        from tagging.category_tagging import TradingTopicsDistiller

        impact_categories = self.impact_categories or [
            "SpecificSecurityImpact",
            "SegmentAssetLevelImpact",
            "MarketWideImpact",
        ]
        config = self.category_tag_config
        destination_map: dict[str, object] = {}
        for category in impact_categories:
            tags = config.get(category) if config else self.target_tags
            if not tags:
                destination_map[category] = NoOpPassThrough(self.target_tags)
            else:
                distiller = TradingTopicsDistiller(
                    llm=self.llm,
                    target_tags=tags,
                    target_tag_examples=self.target_tag_examples,
                    decision_model=self.decision_model,
                    evaluators=self.evaluators,
                )
                destination_map[category] = DistillerAdapter(
                    distiller, self.target_tags
                )
        return destination_map

    def _build_chain(self) -> BaseChain:
        """Override to use DecisionModel for categorisation, ChatAdapter for summarisation."""
        impact_categories = self.impact_categories or [
            "SpecificSecurityImpact",
            "SegmentAssetLevelImpact",
            "MarketWideImpact",
        ]

        # Build distillers with decision_model for categorisation + llm for summarisation.
        destination_map = self._build_category_destinations_mixed()

        impact_router = CategoricalRouterChain(
            condition=BucketingCondition(
                self.decision_model, impact_categories, self.prompt_samples
            ),
            destination_map=destination_map,
            custom_name="ImpactRouter",
        )

        summarizer_cls = self.summarizer_chain
        if issubclass(summarizer_cls, RefineSummarizeChain):
            summarize_chain: BaseChain = RefineSummarizeChain(
                llm=self.llm,
                text_splitter=SentenceTextSplitter(chunk_size=self.text_chunk_size),
            )
        else:
            summarize_chain = MapReduceSummarizeChain(
                llm=self.llm,
                text_splitter=SentenceTextSplitter(chunk_size=self.text_chunk_size),
            )

        summarise_then_categorise = SequentialChain(
            chains=[summarize_chain, impact_router],
        )

        long_text_route = BoolRoute(
            name="LongText",
            condition=TextLengthCondition(min_chars=self.min_chars_for_summarization),
            destination_chain=summarise_then_categorise,
        )

        return BoolRouterChain(
            routes=[long_text_route],
            default_chain=impact_router,
        )
