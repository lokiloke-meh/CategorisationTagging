from __future__ import annotations

from typing import Any

from opentelemetry import trace

from evals.base_evals import BaseEvaluator
from prompt_chains.base_chains import BaseChain, ChainResult, Document
from tagging.category_tagging_one_at_a_time import OneAtATimeCategorisation
from tagging.category_tagging_summariser import TradingTopicsSummarisation
from tagging.utils.data_models import TagResult
from utils.llm_models import ChatAdapter, DecisionModel

tracer = trace.get_tracer(__name__)


class DistillerAdapter:
    """Wrap a distiller, fill any missing target_tags as 'None'.

    Only fills tags NOT in the wrapped distiller's subset — those are evaluated normally.
    Ensures every category destination returns ALL target_tags so cache validation passes.
    """

    name = "DistillerAdapter"

    def __init__(self, distiller: Any, target_tags: list[str]) -> None:
        self.distiller = distiller
        self.target_tags = target_tags

    def invoke(self, docs: list[Any]) -> ChainResult:
        result = self.distiller.invoke(docs)
        if not result.success or not isinstance(result.output_data, TagResult):
            return result
        tags = {**result.output_data.tags}
        for tag in self.target_tags:
            tags.setdefault(tag, "None")
        result.output_data.tags = tags
        return result


class TradingTopicsDistiller(BaseChain):
    """
    Orchestrates OneAtATimeCategorisation and TradingTopicsSummarisation in parallel,
    merges their outputs, and returns a unified TagResult.
    """

    def __init__(
        self,
        llm: ChatAdapter,
        target_tags: list[str],
        target_tag_examples: dict[str, str] | None = None,
        decision_model: DecisionModel | None = None,
        evaluators: list[BaseEvaluator] | None = None,
        eval_max_retries: int = 0,
        invoke_max_retries: int = 0,
    ):
        super().__init__(
            evaluators=evaluators,
            eval_max_retries=eval_max_retries,
            invoke_max_retries=invoke_max_retries,
        )
        self.llm = llm
        self.decision_model = decision_model
        self.target_tags = target_tags
        categorisation_adapter = decision_model or llm
        self.categorisation_chain = OneAtATimeCategorisation(
            llm=categorisation_adapter,
            target_tags=target_tags,
            target_tag_examples=target_tag_examples,
            eval_max_retries=3,
        )
        self.summarisation_chain = TradingTopicsSummarisation(
            llm=llm, target_tags=target_tags
        )

    @staticmethod
    def _merge_branch_outputs(
        categorisation_result: ChainResult,
        summarisation_result: ChainResult,
    ) -> TagResult:
        """Merge a TagResult from categorisation and a dict from summarisation into one TagResult."""
        if not isinstance(categorisation_result.output_data, TagResult):
            raise TypeError(
                f"Categorisation returned {type(categorisation_result.output_data).__name__}, expected TagResult"
            )

        merged_tags = {**categorisation_result.output_data.tags}
        merged_summary = categorisation_result.output_data.summary or ""
        merged_effectivedate = categorisation_result.output_data.effectivedate or None

        if isinstance(summarisation_result.output_data, dict):
            for key, value in summarisation_result.output_data.items():
                if key == "summary" and value:
                    merged_summary = value  # summariser overrides categoriser's summary
                elif key == "effectivedate" and value and value != "NONE":
                    merged_effectivedate = value
                else:
                    merged_tags[key] = value

        return TagResult(
            tags=merged_tags,
            summary=merged_summary or None,
            effectivedate=merged_effectivedate,
        )

    def _invoke(self, docs: list[Document]) -> ChainResult:
        step_history = []
        branch_metadata = {}

        print(f"\n[Starting {self.name} with 2 branches]")

        # Execute categorisation branch
        categorisation_result = self.categorisation_chain.invoke(docs)
        step_history.append(categorisation_result)
        branch_metadata[self.categorisation_chain.name] = categorisation_result.metadata
        if not categorisation_result.success:
            print(
                f"[Aborted. '{self.categorisation_chain.name}' failed: {categorisation_result.error_message}]"
            )
            return ChainResult(
                chain_name=self.name,
                input_data=f"{len(docs)} docs",
                output_data=None,
                success=False,
                error_message=f"Branch '{self.categorisation_chain.name}' failed.",
                history=step_history,
            )

        # Execute summarisation branch
        summarisation_result = self.summarisation_chain.invoke(docs)
        step_history.append(summarisation_result)
        branch_metadata[self.summarisation_chain.name] = summarisation_result.metadata
        if not summarisation_result.success:
            print(
                f"[Aborted. '{self.summarisation_chain.name}' failed: {summarisation_result.error_message}]"
            )
            return ChainResult(
                chain_name=self.name,
                input_data=f"{len(docs)} docs",
                output_data=None,
                success=False,
                error_message=f"Branch '{self.summarisation_chain.name}' failed.",
                history=step_history,
            )

        tag_result = self._merge_branch_outputs(
            categorisation_result, summarisation_result
        )
        print(f"[{self.name} completed successfully]\n")

        return ChainResult(
            chain_name=self.name,
            input_data=f"{len(docs)} documents",
            output_data=tag_result,
            success=True,
            metadata=branch_metadata,
            history=step_history,
        )
