from typing import Literal

import pydantic
from opentelemetry import trace

from evals.base_evals import BaseEvaluator
from prompt_chains.base_chains import BaseChain, ChainResult, Document
from tagging.evals.categories_model_thought_consistency_evals import (
    CategoriesModelThoughtConsistencyEvals,
)
from tagging.evals.repeated_categories_consistency_evals import (
    RepeatedCategoriesConsistencyEvals,
)
from tagging.utils.data_models import EvaluationContext, TagResult
from utils.category_mode import resolve_category_mode
from utils.llm_models import ChatAdapter, DecisionModel

tracer = trace.get_tracer(__name__)


class SingleTradingTopicCategorisation(BaseChain):
    """A BaseChain focused on categorising a single topic."""

    def __init__(
        self,
        llm: ChatAdapter | DecisionModel,
        target_tag: str,
        target_tag_examples: str | None = None,
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
        self.target_tag = target_tag
        self.target_tag_examples = target_tag_examples

    @property
    def name(self):
        return f"{self.__class__.__name__}[{self.target_tag}]"

    def _invoke(self, docs: list[Document]) -> ChainResult:
        input_data = "\n".join(doc.page_content for doc in docs)

        if not isinstance(self.llm, (ChatAdapter, DecisionModel)):
            error_msg = f"llm must be ChatAdapter or DecisionModel, got {type(self.llm).__name__}"
            return ChainResult(
                chain_name=self.name,
                input_data=input_data,
                output_data=None,
                success=False,
                error_message=error_msg,
            )

        if isinstance(self.llm, ChatAdapter):
            return self._invoke_chat_adapter(input_data)
        return self._invoke_decision_model(input_data)

    def _invoke_chat_adapter(self, input_data: str) -> ChainResult:
        system_prompt = f"""
        You are in charge of a trading system and are only interested in the area of {self.target_tag}.

        Read the data provided by the user and determine if it requires you to take any significant action in and only in the area of {self.target_tag}. Categorize the relevance of the area into one of these levels: "High", "Medium", or "Low".
        Rely only on the data provided by the user. 

        For example, for a topic IPAddressChangesbytheExchange, examine if the exchange is changing any network IP addresses that would require adjustments to your own trading system's IP addresses setup, assign "High" if the answer is yes.
        
        Think before answering, explain your reasoning step by step. Ensure the final subcategory in JSON is consistent with the reasoning.

        In your reasoning:
        - Explain only the evidence supporting the selected category.
        - Do not discuss categories that were not selected.
        - Use affirmative language.
        """

        if self.target_tag_examples:
            system_prompt += f"\n\nHere are some examples of what kind of events would fall under the category of {self.target_tag}:\n{self.target_tag_examples}\n\n"

        tags_fields = {
            "reasoning": (str, ...),
            "Relevance": (Literal["Low", "Medium", "High"], ...),
            "Relevance1": (Literal["Low", "Medium", "High"], ...),
            "Relevance2": (Literal["Low", "Medium", "High"], ...),
            "Relevance3": (Literal["Low", "Medium", "High"], ...),
            "Relevance4": (Literal["Low", "Medium", "High"], ...),
        }
        ResponseModel = pydantic.create_model("ResponseModel", **tags_fields)

        payload = {
            "think": True,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_data},
            ],
            "format": ResponseModel.model_json_schema(),
            "options": {"temperature": 0.5},
        }

        llm_result = self.llm.chat_with_expected_type_result(
            chat_payload=payload, result_type="dict"
        )

        if not llm_result.success:
            return ChainResult(
                chain_name=self.name,
                input_data=input_data,
                output_data=None,
                success=False,
                error_message=f"ChatAdapter failed: {llm_result.error_message}",
            )

        vote_values = [
            value
            for key, value in llm_result.model_result.items()
            if key != "reasoning"
        ]

        category_mode = {self.target_tag: resolve_category_mode(vote_values)}

        if not category_mode.get(self.target_tag) in ["Low", "Medium", "High"]:
            return ChainResult(
                chain_name=self.name,
                input_data=input_data,
                output_data=None,
                success=False,
                error_message="Output result is not one of ['Low', 'Medium', 'High']",
            )

        return ChainResult(
            chain_name=self.name,
            input_data=input_data,
            output_data=TagResult(tags=category_mode),
            success=True,
            eval_context=EvaluationContext(
                categories={self.target_tag: category_mode.get(self.target_tag)},
                model_thought=llm_result.model_thought,
                vote_results={self.target_tag: vote_values},
            ),
        )

    def _invoke_decision_model(self, input_data: str) -> ChainResult:

        instructions = f"""
        You are in charge of a trading system and are only interested in the area of {self.target_tag}.

        Read the data provided by the user and determine if it requires you to take any significant action in and only in the area of {self.target_tag}. Categorize the relevance of the area into one of these levels: "High", "Medium", or "Low".
        Rely only on the data provided by the user.
        """

        if self.target_tag_examples:
            instructions += f"\nExamples: {self.target_tag_examples}"

        questions = {
            "relevance": {
                "type": "choice",
                "instructions": instructions,
                "criteria": {
                    "Low": "Low Relevance",
                    "Medium": "Medium Relevance",
                    "High": "High Relevance",
                },
            }
        }
        llm_result = self.llm.make_decision(state=input_data, questions=questions)

        if not llm_result.success:
            return ChainResult(
                chain_name=self.name,
                input_data=input_data,
                output_data=None,
                success=False,
                error_message=f"DecisionModel failed: {llm_result.error_message}",
            )

        answer = llm_result.model_result.get("relevance", {})
        category = answer.get("choice")

        if category not in ("Low", "Medium", "High"):
            return ChainResult(
                chain_name=self.name,
                input_data=input_data,
                output_data=None,
                success=False,
                error_message="Output result is not one of ['Low', 'Medium', 'High']",
            )

        return ChainResult(
            chain_name=self.name,
            input_data=input_data,
            output_data=TagResult(tags={self.target_tag: category}),
            success=True,
            eval_context=EvaluationContext(
                categories={self.target_tag: category},
                answers={"relevance": answer},
            ),
        )


class MultiTradingTopicCategorisation(BaseChain):
    """A BaseChain focused on categorising multiple topics in a single request."""

    def __init__(
        self,
        llm: ChatAdapter | DecisionModel,
        target_tags: list[str],
        target_tag_examples: dict[str, str] | None = None,
    ):
        super().__init__()
        self.llm = llm
        self.target_tags = target_tags
        self.target_tag_examples = target_tag_examples

    @property
    def name(self):
        return f"{self.__class__.__name__}[{self.target_tags}]"

    def _invoke(self, docs: list[Document]) -> ChainResult:
        input_data = "\n".join(doc.page_content for doc in docs)

        if not isinstance(self.llm, (ChatAdapter, DecisionModel)):
            return ChainResult(
                chain_name=self.name,
                input_data=input_data,
                output_data=None,
                success=False,
                error_message=f"llm must be ChatAdapter or DecisionModel, got {type(self.llm).__name__}",
            )

        if isinstance(self.llm, ChatAdapter):
            return self._invoke_chat_adapter(input_data)
        return self._invoke_decision_model(input_data)

    def _invoke_chat_adapter(self, input_data: str) -> ChainResult:
        system_prompt = f"""
        You are in charge of a trading system and are only interested in the area of {self.target_tags}.

        Read the data provided by the user and go through each area {self.target_tags} and determine if it requires you to make changes to your system in and only in that particular area. 
        Categorize the relevance of each area into one of these levels: "High", "Medium", or "Low".
        Rely only on the data provided by the user. 

        """

        tags_fields = {
            tag: (Literal["Low", "Medium", "High"], ...) for tag in self.target_tags
        }
        ResponseModel = pydantic.create_model("ResponseModel", **tags_fields)

        payload = {
            "think": True,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_data},
            ],
            "format": ResponseModel.model_json_schema(),
            "options": {"temperature": 0.5},
        }

        llm_result = self.llm.chat_with_expected_type_result(
            chat_payload=payload, result_type="dict"
        )

        if not llm_result.success:
            return ChainResult(
                chain_name=self.name,
                input_data=input_data,
                output_data=None,
                success=False,
                error_message=f"ChatAdapter failed: {llm_result.error_message}",
            )

        return ChainResult(
            chain_name=self.name,
            input_data=input_data,
            output_data=TagResult(tags=llm_result.model_result),
            success=True,
            eval_context=EvaluationContext(
                categories=llm_result.model_result,
                model_thought=llm_result.model_thought,
            ),
        )

    def _invoke_decision_model(self, input_data: str) -> ChainResult:

        questions = {}
        for tag in self.target_tags:
            instructions = f"Categorize relevance as Low, Medium, or High for {tag}."
            if self.target_tag_examples and tag in self.target_tag_examples:
                instructions += f" Examples: {self.target_tag_examples[tag]}"

            questions[tag] = {
                "type": "choice",
                "instructions": instructions,
                "criteria": {
                    "Low": "Low Relevance",
                    "Medium": "Medium Relevance",
                    "High": "High Relevance",
                },
            }

        llm_result = self.llm.make_decision(state=input_data, questions=questions)

        if not llm_result.success:
            return ChainResult(
                chain_name=self.name,
                input_data=input_data,
                output_data=None,
                success=False,
                error_message=f"DecisionModel failed: {llm_result.error_message}",
            )

        tags = {}
        answers = {}
        for key, answer in llm_result.model_result.items():
            if key not in self.target_tags:
                continue
            if isinstance(answer, dict):
                choice = answer.get("choice")
                if choice in ("Low", "Medium", "High"):
                    tags[key] = choice
                    answers[key] = answer

        return ChainResult(
            chain_name=self.name,
            input_data=input_data,
            output_data=TagResult(tags=tags),
            success=True,
            eval_context=EvaluationContext(categories=tags, answers=answers),
        )


class OneAtATimeCategorisation(BaseChain):
    def __init__(
        self,
        llm: ChatAdapter | DecisionModel,
        target_tags: list[str],
        target_tag_examples: dict[str, str] | None = None,
        evaluators: list[BaseEvaluator] | None = None,
        eval_max_retries: int = 0,
        batch_size: int = 1,
        invoke_max_retries: int = 0,
    ):
        """
        Args:
            batch_size (int): Number of tags to process in a single LLM call. Defaults to 1.
        """
        super().__init__(
            evaluators=evaluators,
            eval_max_retries=eval_max_retries,
            invoke_max_retries=invoke_max_retries,
        )
        self.llm = llm
        self.target_tags = target_tags
        self.target_tag_examples = target_tag_examples
        self.batch_size = max(1, batch_size)

    def _invoke(self, docs: list[Document]) -> ChainResult:
        if not docs:
            return ChainResult(
                chain_name=self.name,
                input_data=None,
                output_data=None,
                success=False,
                error_message="Empty docs passed",
            )

        if not isinstance(self.llm, (ChatAdapter, DecisionModel)):
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join([doc.page_content for doc in docs]),
                output_data=None,
                success=False,
                error_message=f"llm must be ChatAdapter or DecisionModel, got {type(self.llm).__name__}",
            )

        print(
            f"--- OneAtATimeCategorisation: Aggregating {len(self.target_tags)} tags (batch_size={self.batch_size}) ---\n"
        )

        aggregated_tags = {}
        final_summary = None
        final_date = "NONE"
        step_history = []

        for i in range(0, len(self.target_tags), self.batch_size):
            batch_tags = self.target_tags[i : i + self.batch_size]

            if len(batch_tags) == 1:
                # ChatAdapter produces repeated votes + reasoning text → consistency checks make sense
                # DecisionModel returns a single structured result — no repetition to check
                chain_evaluators: list[BaseEvaluator] = []
                if isinstance(self.llm, ChatAdapter):
                    chain_evaluators = [
                        RepeatedCategoriesConsistencyEvals(),
                        CategoriesModelThoughtConsistencyEvals(
                            categories=["Low", "Medium", "High"]
                        ),
                    ]
                tag_chain = SingleTradingTopicCategorisation(
                    self.llm,
                    batch_tags[0],
                    target_tag_examples=(
                        self.target_tag_examples.get(batch_tags[0])
                        if self.target_tag_examples
                        else None
                    ),
                    evaluators=chain_evaluators,
                    eval_max_retries=self.eval_max_retries,
                )
            else:
                tag_chain = MultiTradingTopicCategorisation(
                    self.llm,
                    batch_tags,
                    target_tag_examples=(
                        self.target_tag_examples if self.target_tag_examples else None
                    ),
                )

            result = tag_chain.invoke(docs)
            step_history.append(result)

            if result.success and isinstance(result.output_data, TagResult):
                # Aggregate the tag scores
                aggregated_tags.update(result.output_data.tags)

                # Capture the first available summary and effective date
                if not final_summary:
                    final_summary = result.output_data.summary
                if final_date == "NONE":
                    final_date = result.output_data.effectivedate or "NONE"

        return ChainResult(
            chain_name=self.name,
            input_data="\n".join([doc.page_content for doc in docs]),
            output_data=TagResult(
                tags=aggregated_tags,
                summary=final_summary or None,
                effectivedate=None if final_date == "NONE" else final_date,
            ),
            history=step_history,
            success=(
                all(r.success for r in step_history) if step_history else True
            ),  # False if any individual step failed
        )
