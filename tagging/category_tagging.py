from utils.myOllama import AIChat
from evals.base_evals import BaseEvaluator
from opentelemetry import trace
from tagging.category_tagging_one_at_a_time import TradingTopicsCategorisation
from tagging.category_tagging_summariser import TradingTopicsSummarisation
from prompt_chains.parallel_aggregation_chains import ParallelDictAggregationChain

tracer = trace.get_tracer(__name__)


class TradingTopicsDistiller(ParallelDictAggregationChain):
    def __init__(self, llm: AIChat, 
                 target_tags: list[str],
                 target_tag_examples: dict[str, str] = None,
                 evaluators: list[BaseEvaluator] = None,
                 eval_max_retries: int = 0,
                 invoke_max_retries: int = 0):
        super().__init__(chains=[TradingTopicsCategorisation(llm=llm,
                                                             target_tags=target_tags,
                                                             target_tag_examples=target_tag_examples,
                                                             eval_max_retries=3),
                                TradingTopicsSummarisation(llm=llm, 
                                                           target_tags=target_tags)],
                         evaluators=evaluators,
                         eval_max_retries=eval_max_retries,
                         invoke_max_retries=invoke_max_retries)
        self.llm = llm
        self.target_tags = target_tags
