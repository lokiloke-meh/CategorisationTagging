from typing import Literal

import pydantic
from opentelemetry import trace

from evals.base_evals import BaseEvaluator
from prompt_chains.base_chains import BaseChain, ChainResult, Document
from tagging.utils.data_models import EvaluationContext, TagResult
from utils.llm_models import ChatAdapter, DecisionModel

tracer = trace.get_tracer(__name__)


class AllAtOnceCategorisation(BaseChain):
    def __init__(
        self,
        llm: ChatAdapter | DecisionModel,
        target_tags: list[str],
        tag_examples: dict[str, str] | None = None,
        evaluators: list[BaseEvaluator] | None = None,
    ):
        super().__init__(evaluators=evaluators)
        self.llm = llm
        self.target_tags = target_tags
        self.tag_examples = tag_examples or {}

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

        print(f"--- AllAtOnceCategorisation CHAIN: Processing {len(docs)} chunks ---\n")

        if isinstance(self.llm, ChatAdapter):
            return self._invoke_chat_adapter(docs)
        else:
            return self._invoke_decision_model(docs)

    def _invoke_chat_adapter(self, docs: list[Document]) -> ChainResult:
        system_prompt = f"""
        You are in charge of a trading system and are only interested in information that require you to make changes to your system.

        Given each of these topics {self.target_tags}, read the data provided by the user and categorize the relevance of the topic into one of these levels: "High", "Medium", or "Low".  
        Rely only on the data provided by the user. 

        For example, if the exchange is changing any network IP addresses that would require adjustments to your own trading system, assign "High" to the topic IPAddressChangesbytheExchange, and "Low" to the topic MockDRDisasterRecoverySession.

        Think before answering and reply only in the json format.
        """

        tags_fields = {
            each_tag: (Literal["Low", "Medium", "High"], ...)
            for each_tag in self.target_tags
        }
        TagsModel = pydantic.create_model("TagsModel", **tags_fields)
        ResponseModel = pydantic.create_model("ResponseModel", tags=(TagsModel, ...))

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
            "options": {"temperature": 0.5},
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

        tags = result.model_result.get("tags", {})
        return ChainResult(
            chain_name=self.name,
            input_data="\n".join(doc.page_content for doc in docs),
            output_data=TagResult(tags=tags),
            success=True,
            eval_context=EvaluationContext(
                categories=tags, model_thought=result.model_thought
            ),
        )

    def _invoke_decision_model(self, docs: list[Document]) -> ChainResult:
        questions = {
            tag: {
                "type": "choice",
                "instructions": f"""
                You are in charge of a trading system and are only interested in the area of {tag}.

                Read the data provided by the user and determine if it requires you to take any significant action in and only in the area of {tag}. Categorize the relevance of the area into one of these levels: "High", "Medium", or "Low".
                Rely only on the data provided by the user. 

                """
                + (
                    f"\nExamples: {self.tag_examples.get(tag, '')}"
                    if self.tag_examples.get(tag)
                    else ""
                ),
                "criteria": {
                    "Low": "Low Relevance",
                    "Medium": "Medium Relevance",
                    "High": "High Relevance",
                },
            }
            for tag in self.target_tags
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

        categories = {
            tag: answer["choice"] for tag, answer in result.model_result.items()
        }
        return ChainResult(
            chain_name=self.name,
            input_data="\n".join(doc.page_content for doc in docs),
            output_data=TagResult(tags=categories),
            success=True,
            eval_context=EvaluationContext(
                categories=categories, answers=result.model_result
            ),
        )
