from datetime import date

import pydantic
from opentelemetry import trace

from evals.base_evals import BaseEvaluator
from prompt_chains.base_chains import BaseChain, ChainResult, Document
from utils.llm_models import ChatAdapter

tracer = trace.get_tracer(__name__)


class TradingTopicsSummarisation(BaseChain):
    def __init__(
        self,
        llm: ChatAdapter,
        target_tags: list[str],
        evaluators: list[BaseEvaluator] | None = None,
    ):
        super().__init__(evaluators=evaluators)
        self.llm = llm
        self.target_tags = target_tags

    def _invoke(self, docs: list[Document]) -> ChainResult:
        if not docs:
            return ChainResult(
                chain_name=self.name,
                input_data="",
                output_data="",
                success=False,
                error_message="Empty docs passed",
            )

        print(
            f"--- TradingTopicsSummarisation CHAIN: Processing {len(docs)} chunks ---\n"
        )

        system_prompt = """
        You are in charge of a trading system and are only interested in information that require you to make changes to your system.

        Reply with the effective date of the change if you are very confident of the effective date of the change in the YYYY-MM-DD format. If the data is unclear, leave the effective date as "NONE".

        Add a concise and comprehensive summary of the data that should cover all key points providing relevant details with regards to trading, while avoiding any unnecessary information or repetition.

        Think before answering and reply only in the json format.
        """

        all_fields = {"effectivedate": (date, None), "summary": (str, ...)}

        ResponseModel = pydantic.create_model("ResponseModel", **all_fields)

        payload = {
            "think": False,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "\n".join([doc.page_content for doc in docs]),
                },
            ],
            "format": ResponseModel.model_json_schema(),
            "options": {"temperature": 0.2},
        }

        categorisation_result = self.llm.chat_with_expected_type_result(
            chat_payload=payload, result_type="dict"
        )
        if not categorisation_result.success:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join([doc.page_content for doc in docs]),
                output_data="",
                success=False,
                error_message=categorisation_result.error_message,
            )

        return ChainResult(
            chain_name=self.name,
            input_data="\n".join([doc.page_content for doc in docs]),
            output_data=categorisation_result.model_result,
            success=True,
        )
