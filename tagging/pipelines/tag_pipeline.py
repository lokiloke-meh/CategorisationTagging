"""General-purpose document tagging pipeline.

Two-tier architecture: power users build custom chains from primitives in
prompt_chains/; main.py uses this high-level pipeline as a thin orchestrator.

Flow (short text < 4000 chars):
    BoolRouter(default) → CategoricalRouter(impact) → TradingTopicsDistiller

Flow (long text >= 4000 chars):
    BoolRouter(LongText) → SequentialChain([RefineSummarizeChain, CategoricalRouter(impact)]) → TradingTopicsDistiller
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from evals.base_evals import BaseEvaluator
from prompt_chains.base_chains import BaseChain, ChainResult
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
from prompt_chains.routing_chains_categorical import (
    CategoricalCondition,
    CategoricalRouterChain,
)
from prompt_chains.sequential_chains import SequentialChain
from tagging.category_tagging import DistillerAdapter, TradingTopicsDistiller
from tagging.independent_category_tagging import TopicBucketing
from tagging.utils.data_models import TagResult
from utils.llm_models import ChatAdapter, DecisionModel


class NoOpPassThrough:
    """Skip distillation but mark all target tags as 'None' to prevent re-evaluation."""

    name = "NoOpPassThrough"

    def __init__(self, target_tags: list[str]) -> None:
        self.target_tags = target_tags

    def invoke(self, docs: list[Any]) -> ChainResult:
        tags = {tag: "None" for tag in self.target_tags}
        return ChainResult(
            chain_name="NoOpPassThrough",
            input_data=f"{len(docs)} documents",
            output_data=TagResult(tags=tags),
            success=True,
        )


class BucketingCondition(CategoricalCondition):
    """Routes impact text to one of the known buckets via TopicBucketing."""

    def __init__(
        self, llm: ChatAdapter | DecisionModel, buckets: list[str], samples: str | None
    ) -> None:
        super().__init__()
        self.llm = llm
        self.buckets = buckets
        self.samples = samples

    def evaluate(self, docs: list[Any]) -> str | None:
        bucketing_result = TopicBucketing(
            llm=self.llm,
            target_buckets=self.buckets,
            eval_max_retries=3,
            prompt_samples=self.samples,
        ).invoke(docs)

        if not bucketing_result.success or bucketing_result.output_data is None:
            raise RuntimeError(
                f"Bucketing failed: {bucketing_result.error_message}"
                if bucketing_result.error_message
                else "Bucketing returned no output"
            )
        return bucketing_result.output_data.get("category")  # type: ignore[return-value]


class TagPipeline(BaseChain):
    """Encapsulates the standard document tagging pipeline.

    Parameters:
        llm: LLM adapter used for all chains.
        target_tags: Tags to distill against in TradingTopicsDistiller.
        target_tag_examples: Examples for each tag (optional).
        evaluators: Evaluators to run on final output (optional).
        text_chunk_size: Chunk size for SentenceTextSplitter in RefineSummarizeChain.
        min_chars_for_summarization: Threshold for the BoolRouter length condition.
        impact_categories: Impact bucket categories for TopicBucketing.
        prompt_samples: Optional samples appended to the impact bucketing system prompt.
        summarizer_chain: Which summarizer to use (RefineSummarizeChain or MapReduceSummarizeChain).
    """

    def __init__(
        self,
        llm: ChatAdapter | DecisionModel,
        target_tags: list[str],
        target_tag_examples: dict[str, str] | None = None,
        evaluators: list[Any] | None = None,
        text_chunk_size: int = 800,
        min_chars_for_summarization: int = 4000,
        impact_categories: list[str] | None = None,
        prompt_samples: dict[str, str] | None = None,
        summarizer_chain: type[
            RefineSummarizeChain | MapReduceSummarizeChain
        ] = RefineSummarizeChain,
        category_tag_config: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.target_tags = target_tags
        self.target_tag_examples = target_tag_examples
        self.evaluators = evaluators
        self.text_chunk_size = text_chunk_size
        self.min_chars_for_summarization = min_chars_for_summarization
        self.impact_categories = impact_categories
        self.prompt_samples = prompt_samples
        self.summarizer_chain = summarizer_chain
        self.category_tag_config = category_tag_config

    @staticmethod
    def _serialize_value(obj):
        """Recursively serialize any value to JSON-native types.

        Branches:
          BaseChain       → recurse into chain graph   (via _serialize_chain)
          BaseEvaluator   → skip, replace with name    (not part of identity)
          has __dict__    → flatten its __dict__       (e.g. LLM adapters, conditions)
          else            → keep as-is                 (str, int, float, bool)

        Ensures nested non-serializable objects (llamaCppChat, ollamaChat) are
        fully flattened into their constituent JSON-safe attributes.
        """
        if isinstance(obj, BaseChain):
            return TagPipeline._serialize_chain(obj)
        if isinstance(obj, BaseEvaluator):
            return f"{type(obj).__name__}()"
        if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool)):
            return {
                k: TagPipeline._serialize_value(v)
                for k, v in vars(obj).items()
                if not callable(v)
            }
        return obj

    @staticmethod
    def _serialize_chain(chain: BaseChain) -> dict:
        """Recursively serialize a chain graph into a JSON-serializable dict.

        Branches per attribute:
          BaseChain       → recurse via _serialize_chain     (e.g. default_chain)
          has __dict__    → flatten + recurse values         (e.g. conditions)
          list            → recurse items if they're chains  (e.g. routes)
          dict            → recurse values if they're chains (e.g. destination_map)
          else            → keep as-is                       (str, int, float, bool)

        Each chain node gets a `type` key with its class name for identification.
        """
        result = {"type": type(chain).__name__}
        for attr_name in dir(chain):
            if attr_name.startswith("_") or callable(getattr(chain, attr_name)):
                continue
            value = getattr(chain, attr_name)
            if isinstance(value, BaseChain):
                result[attr_name] = TagPipeline._serialize_chain(value)
            elif hasattr(value, "__dict__") and not isinstance(
                value, (str, int, float, bool, list, dict)
            ):
                result[attr_name] = {
                    k: TagPipeline._serialize_value(v)
                    for k, v in vars(value).items()
                    if not callable(v)
                }
            elif isinstance(value, list):
                result[attr_name] = [
                    (
                        TagPipeline._serialize_chain(item)
                        if isinstance(item, BaseChain)
                        else TagPipeline._serialize_value(item)
                    )
                    for item in value
                ]
            elif isinstance(value, dict):
                result[attr_name] = {
                    k: (
                        TagPipeline._serialize_chain(v)
                        if isinstance(v, BaseChain)
                        else TagPipeline._serialize_value(v)
                    )
                    for k, v in value.items()
                }
            else:
                result[attr_name] = value
        return result

    def pipeline_identity(self) -> tuple[str, dict]:
        """Serialize pipeline config + chain topology, compute hash.

        Returns (hash_string, {pipeline: {...}, chain_graph: {...})).
        Pipeline section captures all instance attributes.
        Chain graph captures the full nested structure built by _build_chain().
        """
        graph = {
            "pipeline": self._serialize_chain(self),
            "chain_graph": self._serialize_chain(self._build_chain()),
        }
        json_str = json.dumps(graph, sort_keys=True, default=str)
        hash_string = hashlib.sha256(json_str.encode()).hexdigest()[:16]
        return hash_string, graph

    def _build_category_destinations(
        self, routing_llm: ChatAdapter | DecisionModel
    ) -> dict[str, object]:
        """Build per-category destination map for the impact router.

        Returns a dict mapping category names to either a NoOpPassThrough
        (if the tag list is empty) or a DistillerAdapter wrapping TradingTopicsDistiller
        (which ensures ALL target_tags are present in output).
        """
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
                    llm=routing_llm,
                    target_tags=tags,
                    target_tag_examples=self.target_tag_examples,
                    evaluators=self.evaluators,
                )
                destination_map[category] = DistillerAdapter(
                    distiller, self.target_tags
                )
        return destination_map

    def _build_chain(self) -> BaseChain:
        """Construct the full pipeline as a composed chain graph."""
        impact_categories = self.impact_categories or [
            "SpecificSecurityImpact",
            "SegmentAssetLevelImpact",
            "MarketWideImpact",
        ]
        prompt_samples = self.prompt_samples

        destination_map = self._build_category_destinations(self.llm)

        impact_router = CategoricalRouterChain(
            condition=BucketingCondition(self.llm, impact_categories, prompt_samples),
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

    def _invoke(self, docs: list[Any]) -> Any:  # returns ChainResult
        chain = self._build_chain()
        return chain.invoke(docs)
