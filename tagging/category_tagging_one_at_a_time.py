import pydantic
from datetime import date
from typing import Literal
from prompt_chains.base_chains import Document, BaseChain, ChainResult
from utils.myOllama import AIChat
from evals.base_evals import BaseEvaluator
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
import statistics
from tagging.evals.repeated_categories_consistency_evals import RepeatedCategoriesConsistencyEvals
from tagging.evals.categories_model_thought_consistency_evals import CategoriesModelThoughtConsistencyEvals

tracer = trace.get_tracer(__name__)


class SingleTradingTopicCategorisation(BaseChain):
    """
    A BaseChain focused on categorising a single topic.
    """
    def __init__(self, llm: AIChat, 
                 target_tag: str,
                 target_tag_examples: str = None,
                 evaluators: list[BaseEvaluator] = None, 
                 eval_max_retries: int = 0,
                 invoke_max_retries: int = 0):
        super().__init__(evaluators=evaluators,
                         eval_max_retries=eval_max_retries,
                         invoke_max_retries=invoke_max_retries)
        self.llm = llm
        self.target_tag = target_tag
        self.target_tag_examples = target_tag_examples

    @property
    def name(self):
        return f"{self.__class__.__name__}[{self.target_tag}]"

    def _invoke(self, docs: list[Document]) -> ChainResult:
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

        tags_fields = {"reasoning":(str, ...),
                       "Relevance": (Literal["Low", "Medium", "High"], ...),
                       "Relevance1": (Literal["Low", "Medium", "High"], ...),
                       "Relevance2": (Literal["Low", "Medium", "High"], ...),
                       "Relevance3": (Literal["Low", "Medium", "High"], ...),
                       "Relevance4": (Literal["Low", "Medium", "High"], ...),}

        ResponseModel = pydantic.create_model("ResponseModel", **tags_fields)

        payload = {
            "think": True, 
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join([doc.page_content for doc in docs])}
            ],
            "format": ResponseModel.model_json_schema(),
            "options": {"temperature": 0.5}    
        }

        categorisation_result = self.llm.chat_with_expected_type_result(chat_payload=payload, result_type="dict")
        
        if not categorisation_result.success:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join([doc.page_content for doc in docs]),
                output_data=None,
                success=False,
                error_message=categorisation_result.error_message
            )

        # take mode of category results
        repeated_categories = {k:v for k,v in categorisation_result.model_result.items() if k not in ("reasoning")}
        category_mode = {self.target_tag: statistics.mode(repeated_categories.values())}

        if not category_mode.get(self.target_tag) in ["Low", "Medium", "High"]:
            return ChainResult(chain_name=self.name,
                               input_data="\n".join([doc.page_content for doc in docs]),
                               output_data=None,
                               success=False,
                               error_message=f"Output result is not one of ['Low', 'Medium', 'High']"
                               )

        return ChainResult(
            chain_name=self.name,
            input_data="\n".join([doc.page_content for doc in docs]),
            output_data={"tags": category_mode},
            success=True,
            eval_context={"repeated_categories": repeated_categories,
                          "model_thought": categorisation_result.model_thought,
                          "single_result_category": category_mode.get(self.target_tag)}
        )


class MultiTradingTopicCategorisation(BaseChain):
    """
    A BaseChain focused on categorising multiple topics in a single request.
    """
    def __init__(self, llm: AIChat, target_tags: list[str]):
        super().__init__()
        self.llm = llm
        self.target_tags = target_tags

    @property
    def name(self):
        return f"{self.__class__.__name__}[{self.target_tags}]"

    def _invoke(self, docs: list[Document]) -> ChainResult:
        system_prompt = f"""
        You are in charge of a trading system and are only interested in the area of {self.target_tags}.

        Read the data provided by the user and go through each area {self.target_tags} and determine if it requires you to make changes to your system in and only in that particular area. Categorize the relevance of each area into one of these levels: "High", "Medium", or "Low".
        Rely only on the data provided by the user. 

        """

        tags_fields = {tag: (Literal["Low", "Medium", "High"], ...) for tag in self.target_tags}
        ResponseModel = pydantic.create_model("ResponseModel", **tags_fields)

        payload = {
            "think": True, 
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join([doc.page_content for doc in docs])}
            ],
            "format": ResponseModel.model_json_schema(),
            "options": {"temperature": 0.5}    
        }

        categorisation_result = self.llm.chat_with_expected_type_result(chat_payload=payload, result_type="dict")
        
        if not categorisation_result.success:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join([doc.page_content for doc in docs]),
                output_data=None,
                success=False,
                error_message=categorisation_result.error_message
            )

        return ChainResult(
            chain_name=self.name,
            input_data="\n".join([doc.page_content for doc in docs]),
            output_data={"tags": categorisation_result.model_result},
            success=True
        )


class TradingTopicsCategorisation(BaseChain):
    def __init__(self, llm: AIChat, 
                 target_tags: list[str],
                 target_tag_examples: dict[str, str] = None,
                 evaluators: list[BaseEvaluator] = None,
                 eval_max_retries: int = 0,
                 batch_size: int = 1,
                 invoke_max_retries: int = 0):
        """
        Args:
            batch_size (int): Number of tags to process in a single LLM call. Defaults to 1.
        """
        super().__init__(evaluators=evaluators,
                         eval_max_retries=eval_max_retries,
                         invoke_max_retries=invoke_max_retries)
        self.llm = llm
        self.target_tags = target_tags
        self.target_tag_examples = target_tag_examples
        self.batch_size = max(1, batch_size)

    def _invoke(self, docs: list[Document]) -> ChainResult:
        if not docs: 
            return ChainResult(chain_name=self.name,
                                 input_data=None,
                                 output_data=None,
                                 success=False, 
                                 error_message="Empty docs passed")
        
        print(f"--- TradingTopicsCategorisation: Aggregating {len(self.target_tags)} tags (batch_size={self.batch_size}) ---\n")

        aggregated_tags = {}
        final_summary = None
        final_date = "NONE"
        step_history = []

        for i in range(0, len(self.target_tags), self.batch_size):
            batch_tags = self.target_tags[i:i + self.batch_size]
            
            if len(batch_tags) == 1:
                tag_chain = SingleTradingTopicCategorisation(self.llm, 
                                                             batch_tags[0],
                                                             target_tag_examples=self.target_tag_examples.get(batch_tags[0]) if self.target_tag_examples else None,
                                                             evaluators=[RepeatedCategoriesConsistencyEvals(),
                                                                         CategoriesModelThoughtConsistencyEvals(categories=["Low", "Medium", "High"])],
                                                             eval_max_retries=self.eval_max_retries)
            else:
                tag_chain = MultiTradingTopicCategorisation(self.llm, batch_tags)
            
            result = tag_chain.invoke(docs)
            step_history.append(result)

            if result.success and result.output_data:
                # Aggregate the tag scores
                aggregated_tags.update(result.output_data.get("tags", {}))
                
                # Capture the first available summary and effective date
                if not final_summary:
                    final_summary = result.output_data.get("summary")
                if final_date == "NONE":
                    final_date = result.output_data.get("effectivedate", "NONE")

        output_data = {
            "tags": aggregated_tags,
            "summary": final_summary or "No summary produced.",
            "effectivedate": final_date
        }

        return ChainResult(chain_name=self.name,
                           input_data="\n".join([doc.page_content for doc in docs]),
                           output_data=output_data,
                           history=step_history,
                           success=all(r.success for r in step_history) if step_history else True # False if any individual step failed
                           )
    