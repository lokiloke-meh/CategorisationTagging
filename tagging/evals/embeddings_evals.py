from config import cfg
from evals.base_evals import EvalResult
from evals.embeddings_evals import EmbeddingsEvaluator
from ingestion.content_store import ContentStore
from prompt_chains.base_chains import ChainResult, Document
from utils.llm_models import EmbeddingAdapter, LLMResult


class CategorySpecificEmbedEvaluator(EmbeddingsEvaluator):
    """
    Internal evaluator created for each valid category to provide
    granular traceability in the trace logs.
    """

    def __init__(
        self,
        category: str,
        current_emb_result: LLMResult,
        llm: EmbeddingAdapter,
        content_store: ContentStore,
        peers_to_compare: int,
        target_levels: list[str] | None = None,
    ):
        self.category = category
        self.current_emb_result = current_emb_result
        self.llm = llm
        self.content_store = content_store
        self.peers_to_compare = peers_to_compare
        self.target_levels = target_levels or cfg.default_target_levels

    @property
    def name(self) -> str:
        """Custom name ensures each category has a unique identifiable span in traces."""
        return f"CategoryConsistency[{self.category}]"

    def _evaluate(self, inputs: list[Document], output: ChainResult) -> EvalResult:
        # Find historical peers for this specific category
        other_hashes = self.content_store.find_by_tags_score(
            tag_name=self.category, target_levels=self.target_levels
        )

        valid_peers = []
        for h in other_hashes:
            entry = self.content_store.get_latest_result(h)
            # Read summary from file-level (shared across all tags in this pipeline run)
            peer_summary = entry.get("summary") if isinstance(entry, dict) else None

            if peer_summary:
                valid_peers.append((h, peer_summary))

        sampled_peers = valid_peers[: self.peers_to_compare]
        if not sampled_peers:
            return EvalResult(
                eval_name=self.name,
                eval_success=True,
                passed=True,
                feedback="No historical peers found for comparison.",
            )

        peer_similarities = {}
        for peer_hash, peer_text in sampled_peers:
            try:
                peer_res = self.llm.get_embeddings(text=peer_text)
                if peer_res.success and peer_res.model_result:
                    # Pass the LLMResult objects directly as calc_cosine_similarity
                    # accesses the .model_result attribute.
                    cosine_sim = self.__class__.calc_cosine_similarity(
                        self.current_emb_result.model_result, peer_res.model_result
                    )
                    peer_similarities[peer_hash] = {
                        "cosine_similarity": cosine_sim,
                        "similarity_scoring": self.__class__.score_cosine_similarity_result(
                            cosine_sim
                        ),
                    }
                else:
                    # Store the error in the results instead of crashing the loop
                    peer_similarities[peer_hash] = {
                        "error": peer_res.error_message
                        or "Failed to fetch peer embedding"
                    }
            except (ValueError, TypeError) as e:
                peer_similarities[peer_hash] = {"error": str(e)}

        valid_peer_similarities = {
            each_peer: peer_values["similarity_scoring"]
            for each_peer, peer_values in peer_similarities.items()
            if peer_values.get("similarity_scoring")
        }
        if not valid_peer_similarities:
            return EvalResult(
                eval_name=self.name,
                eval_success=False,
                error_message="No successful cosine similarity calculations.",
            )

        avg_score = sum(valid_peer_similarities.values()) / len(valid_peer_similarities)

        return EvalResult(
            eval_name=self.name,
            eval_success=True,
            passed=avg_score >= 0.75,
            score=avg_score,
            eval_intermediate_results=peer_similarities,
        )


class CategoryVSSummaryEmbedEvals(EmbeddingsEvaluator):
    def __init__(
        self,
        llm: EmbeddingAdapter,
        content_store: ContentStore,
        peers_to_compare: int = 5,
        target_levels: list[str] | None = None,
    ):
        self.llm = llm
        self.content_store = content_store
        self.peers_to_compare = peers_to_compare
        self.target_levels = target_levels or cfg.default_target_levels
        self.include_parent_eval = False

    def _evaluate(self, inputs: list[Document], output: ChainResult) -> EvalResult:
        if not output.success:
            return EvalResult(
                eval_name=self.name,
                eval_success=False,
                error_message="ChainResult is not successful, nothing to evaluate.",
            )

        result_output_data = output.output_data
        summary_text = getattr(result_output_data, "summary", None) if result_output_data else None
        if not summary_text:
            return EvalResult(
                eval_name=self.name,
                eval_success=False,
                error_message="ChainResult has no summary to be used for evaluation.",
            )

        try:
            deemed_valid_categories = []

            tag_results_dict = getattr(result_output_data, "tags", {}) or {}
            for tag_name, tag_value in tag_results_dict.items():
                if tag_value in self.target_levels:
                    deemed_valid_categories.append(tag_name)

            if not deemed_valid_categories:
                return EvalResult(
                    eval_name=self.name,
                    eval_success=True,
                    passed=True,
                    feedback="No categories reached the threshold for embedding comparison.",
                )

            # Fetch embedding for the current summary
            current_res = self.llm.get_embeddings(text=summary_text)
            if not current_res.success or not current_res.model_result:
                raise ValueError(
                    f"Failed to fetch embedding for current summary: {current_res.error_message}"
                )

            category_evaluation_details = {}

            # Evaluate similarity for each deemed valid category
            for category in deemed_valid_categories:
                sub_evaluator = CategorySpecificEmbedEvaluator(
                    category=category,
                    current_emb_result=current_res,
                    llm=self.llm,
                    content_store=self.content_store,
                    peers_to_compare=self.peers_to_compare,
                    target_levels=self.target_levels,
                )
                category_evaluation_details[category] = sub_evaluator.evaluate(
                    inputs, output
                )

            # Process lists once to optimize performance
            all_sub_results = list(category_evaluation_details.values())
            successful_evals = [res for res in all_sub_results if res.eval_success]

            if not successful_evals:
                return EvalResult(
                    eval_name=self.name,
                    eval_success=False,
                    error_message="All CategorySpecificEmbedEvaluator failed.",
                    eval_children=all_sub_results,
                )

            scored_evals = [res for res in successful_evals if res.score is not None]
            if not scored_evals:
                return EvalResult(
                    eval_name=self.name,
                    eval_success=True,
                    passed=True,
                    feedback="All CategorySpecificEmbedEvaluator passed as there were no peers.",
                    eval_children=all_sub_results,
                )

            final_score = sum(res.score for res in scored_evals) / len(scored_evals)

            return EvalResult(
                eval_name=self.name,
                eval_success=True,
                passed=final_score >= 0.6,
                score=final_score,
                eval_children=all_sub_results,
                eval_intermediate_results=category_evaluation_details,
            )

        except Exception as e:  # noqa: BLE001
            return EvalResult(
                eval_name=self.name, eval_success=False, error_message=str(e)
            )
