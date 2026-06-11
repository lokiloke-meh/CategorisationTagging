from evals.base_evals import EvalResult
from evals.embeddings_evals import EmbeddingsEvaluator
from prompt_chains.base_chains import ChainResult, Document
from tagging.process_markdowns import CacheQuery
from utils.myOllama import AIEmbeddings, LLMResult


class CategorySpecificEmbedEvaluator(EmbeddingsEvaluator):
    """
    Internal evaluator created for each valid category to provide 
    granular traceability in the trace logs.
    """
    def __init__(self, category: str, 
                 current_emb_result: LLMResult, 
                 llm: AIEmbeddings, 
                 cache_query: CacheQuery, 
                 peers_to_compare: int,
                 target_levels: list[str] = None):
        super().__init__()
        self.category = category
        self.current_emb_result = current_emb_result
        self.llm = llm
        self.cache_query = cache_query
        self.peers_to_compare = peers_to_compare
        self.target_levels = target_levels or ["High"]

    @property
    def name(self) -> str:
        """Custom name ensures each category has a unique identifiable span in traces."""
        return f"CategoryConsistency[{self.category}]"

    def _evaluate(self, inputs: list[Document], output: ChainResult) -> EvalResult:
        # Find historical peers for this specific category
        other_hashes = self.cache_query.find_by_tags_score(tag_name=self.category, target_levels=self.target_levels)
        
        valid_peers = []
        for h in other_hashes:
            entry = self.cache_query.get_full_cached_entry(h)
            # Safely navigate nested dicts that might contain None values
            result_dict = entry.get("result") or {}
            output_data = result_dict.get("output_data") or {}
            peer_summary = output_data.get("summary")

            if peer_summary:
                valid_peers.append((h, peer_summary))
        
        sampled_peers = valid_peers[:self.peers_to_compare]
        if not sampled_peers:
            return EvalResult(eval_name=self.name,
                              eval_success=True,
                              passed=True,
                              feedback="No historical peers found for comparison.")

        peer_similarities = {}
        for peer_hash, peer_text in sampled_peers:
            try:
                peer_res = self.llm.get_embeddings(body_payload={"input": peer_text})
                if peer_res.success and peer_res.model_result:
                    # Pass the LLMResult objects directly as calc_cosine_similarity 
                    # accesses the .model_result attribute.
                    cosine_sim = self.__class__.calc_cosine_similarity(self.current_emb_result.model_result, peer_res.model_result)
                    peer_similarities[peer_hash] = {
                        "cosine_similarity": cosine_sim,
                        "similarity_scoring": self.__class__.score_cosine_similarity_result(cosine_sim)
                    }
                else:
                    # Store the error in the results instead of crashing the loop
                    peer_similarities[peer_hash] = {"error": peer_res.error_message or "Failed to fetch peer embedding"}
            except Exception as e:
                peer_similarities[peer_hash] = {"error": str(e)}


        valid_peer_similarities = {each_peer: peer_values["similarity_scoring"] for each_peer, peer_values in peer_similarities.items() if peer_values.get("similarity_scoring")}
        if not valid_peer_similarities:
            return EvalResult(
                eval_name=self.name,
                eval_success=False,
                error_message="No successful cosine similarity calculations."
            )

        avg_score = sum(valid_peer_similarities.values()) / len(valid_peer_similarities)
        
        return EvalResult(
            eval_name=self.name,
            eval_success=True,
            passed=avg_score >= 0.75,
            score=avg_score,
            eval_intermediate_results=peer_similarities
        )


class CategoryVSSummaryEmbedEvals(EmbeddingsEvaluator):
    def __init__(self, llm: AIEmbeddings,
                 cache_query: CacheQuery,
                 peers_to_compare: int = 5,
                 target_levels: list[str] = None):
        super().__init__()
        self.llm = llm
        self.cache_query = cache_query
        self.peers_to_compare = peers_to_compare
        self.target_levels = target_levels or ["High"]
        self.include_parent_eval = False


    def _evaluate(self, inputs: list[Document], output: ChainResult) -> EvalResult:
        if not output.success:
            return EvalResult(
                eval_name=self.name,
                eval_success=False,
                error_message="ChainResult is not successful, nothing to evaluate."
            )
        
        output_data = output.output_data or {}
        summary_text = output_data.get("summary")
        if not summary_text:
            return EvalResult(
                eval_name=self.name,
                eval_success=False,
                error_message="ChainResult has no summary to be used for evaluation."
            )

        try:
            deemed_valid_categories = []

            tags_data = output_data.get("tags", {})
            for k, v in tags_data.items():
                if v in self.target_levels:
                    deemed_valid_categories.append(k)
            
            if not deemed_valid_categories:
                return EvalResult(
                    eval_name=self.name,
                    eval_success=True,
                    passed=True,
                    feedback="No categories reached the threshold for embedding comparison."
                )

            # Fetch embedding for the current summary
            current_res = self.llm.get_embeddings(body_payload={"input": summary_text})
            if not current_res.success or not current_res.model_result:
                raise ValueError(f"Failed to fetch embedding for current summary: {current_res.error_message}")
            
            category_evaluation_details = {}

            # Evaluate similarity for each deemed valid category
            for category in deemed_valid_categories:
                sub_evaluator = CategorySpecificEmbedEvaluator(
                    category=category,
                    current_emb_result=current_res,
                    llm=self.llm,
                    cache_query=self.cache_query,
                    peers_to_compare=self.peers_to_compare,
                    target_levels=self.target_levels
                )
                category_evaluation_details[category] = sub_evaluator.evaluate(inputs, output)

            # Process lists once to optimize performance
            all_sub_results = list(category_evaluation_details.values())
            successful_evals = [res for res in all_sub_results if res.eval_success]
            
            if not successful_evals:
                return EvalResult(
                        eval_name=self.name,
                        eval_success=False,
                        error_message="All CategorySpecificEmbedEvaluator failed.",
                        eval_children=all_sub_results
                    )

            scored_evals = [res for res in successful_evals if res.score is not None]
            if not scored_evals:
                return EvalResult(
                        eval_name=self.name,
                        eval_success=True,
                        passed=True,
                        feedback="All CategorySpecificEmbedEvaluator passed as there were no peers.",
                        eval_children=all_sub_results
                    )

            final_score = sum(res.score for res in scored_evals) / len(scored_evals)
            
            return EvalResult(
                eval_name=self.name,
                eval_success=True,
                passed=final_score >= 0.6,
                score=final_score,
                eval_children=all_sub_results,
                eval_intermediate_results=category_evaluation_details
            )

        except Exception as e:
            return EvalResult(eval_name=self.name, 
                              eval_success=False, 
                              error_message=str(e))
