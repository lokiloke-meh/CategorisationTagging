import os
from tagging.process_markdowns import ProcessingCache, CacheQuery
from utils.otel_setup import setup_otel_tracing
from prompt_chains.base_chains import Document
from prompt_chains.sequential_chains import SequentialChain
from prompt_chains.routing_chains_bool import BoolRouterChain, BoolRoute, TextLengthCondition
from prompt_chains.routing_chains_categorical import CategoricalCondition, CategoricalRouterChain
from prompt_chains.long_text_summariser import MapReduceSummarizeChain, SentenceTextSplitter, RefineSummarizeChain
from ingestion.content_tracker import ContentTracker, ContentReader
from tagging.category_tagging import TradingTopicsDistiller
from tagging.independent_category_tagging import TopicBucketing
from tagging.evals.result_data_evals import ResultDataValidationEvals
from tagging.evals.embeddings_evals import CategoryVSSummaryEmbedEvals
from tagging.evals.repeated_categories_consistency_evals import RepeatedCategoriesConsistencyEvals
from tagging.evals.categories_model_thought_consistency_evals import CategoriesModelThoughtConsistencyEvals
from utils.myOllama import ollamaChat, ollamaEmbeddings


class ImpactCondition(CategoricalCondition):
    def __init__(self, impact_categories: list[str] = ["SpecificSecurityImpact", "SegmentAssetLevelImpact", "MarketWideImpact"]):
        self.impact_categories = impact_categories

    def evaluate(self, docs: list[Document]) -> str | None:

        prompt_samples = """
        SpecificSecurityImpact
        - Use only when the event concerns one identified security (stock, bond, ETF, ticker, issuer) or a very small named set of securities.
        - Examples:
        Listing of stock 123.
        Suspension of ticker XYZ.
        New securities launch of existing products (e.g. new options, new ETF launch).

        SegmentAssetLevelImpact
        - Use when the event concerns a market segment, asset class, trading channel, participant type, or operational subsystem affecting many securities.
        - DO NOT use if only a subset of securities in a segment are affected
        - DO NOT use if the notice impacts systems used by all standard securities
        - Examples:
        Derivatives platform maintenance.
        Futures trading hours

        MarketWideImpact:
        - Use only when the event affects the entire exchange or market infrastructure across segments.
        - Examples:
        Exchange-wide system outage.
        Market-wide trading schedule change.
        """
        
        bucketing_result = TopicBucketing(llm=ollamaChat(model="gemma4:e4b-it-qat"), 
                                          target_buckets=self.impact_categories,
                                          evaluators=[RepeatedCategoriesConsistencyEvals(),
                                                      CategoriesModelThoughtConsistencyEvals(categories=self.impact_categories)],
                                          eval_max_retries=3,
                                          prompt_samples=prompt_samples).invoke(docs)

        if not bucketing_result.success or bucketing_result.output_data is None:
            return None

        result_category = bucketing_result.output_data.get("category")

        return result_category



def process_all_markdowns():

    base_dir = os.path.join(os.getcwd(), "hkex_circulars")
    md_dir = os.path.join(base_dir, "ingest")
    
    # Configuration toggle: Set to True to re-run items that failed in previous attempts
    reprocess_failed_entries = True
    reprocess_older_than = None # Example: "2024-05-10T00:00:00"

    # Initialize both trackers
    tracker = ContentTracker(
        hash_map_filepath=os.path.join(base_dir, "file_hashes.json"), 
        raw_files_directory=md_dir
    )
    reader = ContentReader(tracker)

    processor = ProcessingCache(
        cache_filepath=os.path.join(base_dir, f"processing_results.json"),
        min_timestamp=reprocess_older_than
    )

    processor_cache_query = CacheQuery(processor=processor)
    
    # Define the parameter for this run. 
    # If you change this tomorrow, the cache will automatically invalidate and re-run!
    target_tags = ["MarketWideBoardLotSizeFrameworkChanges", 
                   "MockDRDisasterRecoverySession"]


    target_tag_examples = {
        "MarketWideBoardLotSizeFrameworkChanges": "Changes to the board lot size framework that apply to a large set of securities on the exchange. Introduction of new board lot sizes or tiering.",
        "MockDRDisasterRecoverySession": "Related to mock disaster recovery sessions conducted by the exchange. Any special sessions that are out of the ordinary schedule, or changes to the usual DR drill procedures."
    }

    print(f"Starting batch processing with target_tags: {target_tags}")
    

    trading_topic_categorisation_chain = TradingTopicsDistiller(llm=ollamaChat(model="gemma4:e4b-it-qat"), 
                                                                target_tags=target_tags,
                                                                target_tag_examples=target_tag_examples,
                                                                evaluators=[ResultDataValidationEvals()])

    impact_router = CategoricalRouterChain(condition=ImpactCondition(),
                                            destination_map={"MarketWideImpact": trading_topic_categorisation_chain,
                                                             "SegmentAssetLevelImpact": trading_topic_categorisation_chain},
                                            custom_name="ImpactRouter",
                                            invoke_max_retries=1)


    SummariseThenCategorise = SequentialChain(chains=[RefineSummarizeChain(llm=ollamaChat(model="gemma4:e4b-it-qat"), 
                                                                           text_splitter=SentenceTextSplitter(chunk_size=800)),
                                                      impact_router],
                                                invoke_max_retries=1)
    
    long_text_route = BoolRoute(name="LongText", 
                                condition=TextLengthCondition(min_chars=4000), 
                                destination_chain=SummariseThenCategorise)
    
    process_chain = BoolRouterChain(routes=[long_text_route], 
                                    default_chain=impact_router,
                                    custom_name="Tagging")


    # Loop through all files that the scraper has ever downloaded
    for file_hash, relative_path in tracker.hashes.items():

        # Handle the new dictionary structure in ContentTracker
        entry = tracker.hashes[file_hash]
        entry_title = entry.get("title")
        relative_path = entry["path"] if isinstance(entry, dict) else entry

        print(f"\nEvaluating: {relative_path}")
        
        # Check the Cache First
        cached_result = processor.get_cached_result(
            file_hash, 
            target_tags, 
            ignore_failures=reprocess_failed_entries,
            min_timestamp=reprocess_older_than
        )
        
        if cached_result is not None:
            print(f"[CACHED] Found existing result! Skipping execution.")
            final_result = cached_result
        else:
            # Fetch content and Execute if Cache Misses (or parameter changed)
            markdown_content = reader.fetch_markdown_content(file_hash)
            
            if markdown_content is None:
                print(f"  ->[SKIP] Not a valid markdown or file missing.")
                continue

            markdown_document = Document(page_content=markdown_content, title=entry_title)

            final_result =  process_chain.invoke(docs=[markdown_document])

            print(f"final_result = {final_result}")

            print(final_result.to_dict())
            
            if "output_data" in final_result.to_dict():
                # Save it so we don't have to run it next time)
                result_to_save = {k:v for k,v in final_result.to_dict().items() if k in ["metadata", "output_data", "success", "error_message"]}
                processor.save_result(file_hash, target_tags, result_to_save)
                print(f"    [SAVED] New result cached.")






if __name__ == "__main__":
    # Initialize tracing before starting the main process
    # Traces will be saved to logs/otel_traces.log
    setup_otel_tracing(log_dir=os.path.join("hkex_circulars", 'traces'))
    
    process_all_markdowns()
