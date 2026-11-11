from typing import Literal

import pydantic
from opentelemetry import trace

from evals.base_evals import BaseEvaluator
from prompt_chains.base_chains import BaseChain, ChainResult, Document
from tagging.utils.data_models import EvaluationContext
from utils.category_mode import resolve_category_mode
from utils.llm_models import ChatAdapter, DecisionModel

tracer = trace.get_tracer(__name__)


class TopicBucketing(BaseChain):
    """Route input text to one of the known subcategories.

    Parameters:
        llm: LLM adapter (ChatAdapter or DecisionModel).
        target_buckets: List of valid category choices.
        prompt_samples: Optional dict mapping bucket names to example texts.
            Only buckets with entries receive examples in their instructions.
        evaluators, eval_max_retries, invoke_max_retries: Inherited from BaseChain.
    """

    def __init__(
        self,
        llm: ChatAdapter | DecisionModel,
        target_buckets: list[str],
        evaluators: list[BaseEvaluator] | None = None,
        eval_max_retries: int = 0,
        invoke_max_retries: int = 0,
        prompt_samples: dict[str, str] | None = None,
    ):
        super().__init__(
            evaluators=evaluators,
            eval_max_retries=eval_max_retries,
            invoke_max_retries=invoke_max_retries,
        )
        self.llm = llm
        self.target_buckets = target_buckets
        self.prompt_samples = prompt_samples

    def _invoke(self, docs: list[Document]) -> ChainResult:
        if not docs:
            return ChainResult(
                chain_name=self.name,
                input_data="",
                output_data="",
                success=False,
                error_message="Empty docs passed",
            )

        if not isinstance(self.llm, (ChatAdapter, DecisionModel)):
            return ChainResult(
                chain_name=self.name,
                input_data="",
                output_data=None,
                success=False,
                error_message=f"llm must be ChatAdapter or DecisionModel, got {type(self.llm).__name__}",
            )

        print(f"--- TopicBucketing CHAIN: Processing {len(docs)} chunks ---\n")

        if isinstance(self.llm, ChatAdapter):
            return self._invoke_chat_adapter(docs)
        else:
            return self._invoke_decision_model(docs)

    def _invoke_chat_adapter(self, docs: list[Document]) -> ChainResult:
        system_prompt = f"""
        You are in charge of a trading system. Your task is to analyze the provided user input and select the one MOST relevant subcategory from below

        Subcategories = {self.target_buckets}

        You can only choose one of the subcategories. Rely only on the data provided by the user.

        Think before answering, explain your reasoning step by step. Conclude on the final subcategory in JSON that is consistent with the reasoning.

        In your reasoning:
        - Write reasoning that supports only the selected category.
        - Do not discuss categories that were not selected.
        - Use affirmative language.
        - Do not use elimination reasoning
        """

        if self.prompt_samples:
            samples_text = "\n\nExamples:\n" + "\n\n".join(
                f"{bucket}: {sample}" for bucket, sample in self.prompt_samples.items()
            )
            system_prompt += samples_text

        all_fields = {
            "reasoning": (str, ...),
            "category": (Literal[*self.target_buckets], ...),
            "category1": (Literal[*self.target_buckets], ...),
            "category2": (Literal[*self.target_buckets], ...),
            "category3": (Literal[*self.target_buckets], ...),
            "category4": (Literal[*self.target_buckets], ...),
        }

        ResponseModel = pydantic.create_model("ResponseModel", **all_fields)

        payload = {
            "think": True,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "\n".join(doc.page_content for doc in docs),
                },
            ],
            "format": ResponseModel.model_json_schema(),
            "options": {"temperature": 0.2},
        }

        result = self.llm.chat_with_expected_type_result(
            chat_payload=payload, result_type="dict"
        )

        if not result.success:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join(doc.page_content for doc in docs),
                output_data=None,
                success=False,
                error_message=f"ChatAdapter failed: {result.error_message}",
            )

        vote_values = [v for k, v in result.model_result.items() if k != "reasoning"]
        category_mode = resolve_category_mode(vote_values)

        if category_mode not in self.target_buckets:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join(doc.page_content for doc in docs),
                output_data=None,
                success=False,
                error_message=f"Output result is not one of {self.target_buckets}",
            )

        return ChainResult(
            chain_name=self.name,
            input_data="\n".join(doc.page_content for doc in docs),
            output_data={"category": category_mode},
            success=True,
            eval_context=EvaluationContext(
                categories={"category": category_mode},
                model_thought=result.model_thought,
                vote_results={"category": vote_values},
            ),
        )

    def _invoke_decision_model(self, docs: list[Document]) -> ChainResult:
        instructions = f"""
        You are in charge of a trading system. Your task is to analyze the provided user input and select the one MOST relevant subcategory from below.
        Select the single most relevant subcategory from: {self.target_buckets}
        """

        _samples = self.prompt_samples or {}
        questions = {
            "category": {
                "type": "choice",
                "instructions": instructions,
                "criteria": {
                    bucket: (_samples.get(bucket) if _samples.get(bucket) else bucket)
                    for bucket in self.target_buckets
                },
            }
        }

        result = self.llm.make_decision(
            state="\n".join(doc.page_content for doc in docs),
            questions=questions,
        )

        if not result.success:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join(doc.page_content for doc in docs),
                output_data=None,
                success=False,
                error_message=f"DecisionModel failed: {result.error_message}",
            )

        route = result.model_result.get("category", {})
        category_choice = route.get("choice")

        if not category_choice or category_choice not in self.target_buckets:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join(doc.page_content for doc in docs),
                output_data=None,
                success=False,
                error_message=f"Output result is not one of {self.target_buckets}",
            )

        return ChainResult(
            chain_name=self.name,
            input_data="\n".join(doc.page_content for doc in docs),
            output_data={"category": category_choice},
            success=True,
            eval_context=EvaluationContext(
                categories={"category": category_choice},
                answers=result.model_result,
            ),
        )
