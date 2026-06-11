import pydantic
from datetime import date
from typing import Literal
from prompt_chains.base_chains import Document, BaseChain, ChainResult
from utils.myOllama import AIChat
from evals.base_evals import BaseEvaluator
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)


class TradingTopicsCategorisation(BaseChain):
    def __init__(self, llm: AIChat, 
                 target_tags: list[str],
                 evaluators: list[BaseEvaluator] = None):
        super().__init__(evaluators=evaluators)
        self.llm = llm
        self.target_tags = target_tags

    def _invoke(self, docs: list[Document]) -> ChainResult:
        if not docs: 
            return ChainResult(chain_name=self.name,
                               input_data="",
                               output_data="",
                               success=False, 
                               error_message="Empty docs passed")
        
        print(f"--- TradingTopicsCategorisation CHAIN: Processing {len(docs)} chunks ---\n")

        system_prompt = f"""
        You are in charge of a trading system and are only interested in information that require you to make changes to your system.

        Given each of these topics {self.target_tags}, read the data provided by the user and categorize the relevance of the topic into one of these levels: "High", "Medium", or "Low".  
        Rely only on the data provided by the user. 

        For example, if the exchange is changing any network IP addresses that would require adjustments to your own trading system, assign "High" to the topic IPAddressChangesbytheExchange, and "Low" to the topic MockDRDisasterRecoverySession.

        Think before answering and reply only in the json format.
        """

        tags_fields = {each_tag: (Literal["Low", "Medium", "High"], ...) for each_tag in self.target_tags}

        TagsModel = pydantic.create_model("TagsModel", **tags_fields)

        all_fields = {**{"tags": TagsModel}}
        
        ResponseModel = pydantic.create_model("ResponseModel", **all_fields)

        payload = {
            "think": True, 
            "stream": False,
            "messages": [
                {
                "role": "system",
                "content": system_prompt
                },
                {
                "role": "user",
                "content": "\n".join([doc.page_content for doc in docs])
                }
            ],
            "format": ResponseModel.model_json_schema(),
            "options": {"temperature": 0.5}    
        }

        categorisation_result = self.llm.chat_with_expected_type_result(chat_payload=payload, result_type="dict")
        if not categorisation_result.success:
            return ChainResult(chain_name=self.name,
                               input_data="\n".join([doc.page_content for doc in docs]),
                               output_data="",
                               success=False,
                               error_message=categorisation_result.error_message
                               )

        return ChainResult(chain_name=self.name,
                           input_data="\n".join([doc.page_content for doc in docs]),
                           output_data=categorisation_result.model_result,
                           success=True
                           )
    