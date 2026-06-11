import pydantic
from datetime import date
from typing import Literal
from prompt_chains.base_chains import Document, BaseChain, ChainResult
from utils.myOllama import AIChat
from evals.base_evals import BaseEvaluator
from opentelemetry import trace
import statistics

tracer = trace.get_tracer(__name__)


class TopicBucketing(BaseChain):
    def __init__(self, llm: AIChat, 
                 target_buckets: list[str],
                 evaluators: list[BaseEvaluator] = None,
                 eval_max_retries: int = 0,
                 invoke_max_retries: int = 0,
                 prompt_samples: str = None):
        super().__init__(evaluators=evaluators,
                         eval_max_retries=eval_max_retries,
                         invoke_max_retries=invoke_max_retries)
        self.llm = llm
        self.target_buckets = target_buckets
        self.prompt_samples = prompt_samples

    def _invoke(self, docs: list[Document]) -> ChainResult:
        if not docs: 
            return ChainResult(chain_name=self.name,
                               input_data="",
                               output_data="",
                               success=False, 
                               error_message="Empty docs passed")
        
        print(f"--- TopicBucketing CHAIN: Processing {len(docs)} chunks ---\n")

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
            system_prompt += self.prompt_samples

        all_fields = {"reasoning":(str, ...),
                      "category": (Literal[*self.target_buckets], ...),
                      "category1": (Literal[*self.target_buckets], ...),
                      "category2": (Literal[*self.target_buckets], ...),
                      "category3": (Literal[*self.target_buckets], ...),
                      "category4": (Literal[*self.target_buckets], ...)}
        
        
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
            "options": {"temperature": 0.2} 
        }

        categorisation_result = self.llm.chat_with_expected_type_result(chat_payload=payload, result_type="dict")


        if not categorisation_result.success:
            return ChainResult(chain_name=self.name,
                               input_data="\n".join([doc.page_content for doc in docs]),
                               output_data=None,
                               success=False,
                               error_message=categorisation_result.error_message
                               )
        
        # take mode of category results
        repeated_categories = {k:v for k,v in categorisation_result.model_result.items() if k not in ('reasoning')}
        category_mode = {"category": statistics.mode(repeated_categories.values())}
        
        if not category_mode.get("category") in self.target_buckets:
            return ChainResult(chain_name=self.name,
                               input_data="\n".join([doc.page_content for doc in docs]),
                               output_data=None,
                               success=False,
                               error_message=f"Output result is not one of {self.target_buckets}"
                               )

        return ChainResult(chain_name=self.name,
                           input_data="\n".join([doc.page_content for doc in docs]),
                           output_data=category_mode,
                           success=True,
                           eval_context={
                               "repeated_categories": repeated_categories,
                               "model_thought": categorisation_result.model_thought,
                               "single_result_category": category_mode.get("category")}
                           )
