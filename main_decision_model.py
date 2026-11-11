import argparse
import os

from config import cfg
from ingestion.content_store import ContentStore
from prompt_chains.base_chains import Document
from tagging.evals.result_data_evals import ResultDataValidationEvals
from tagging.utils.tag_examples import load_tag_examples
from utils.myllamacpp import llamaCppChat, llamacppDecisionModel
from utils.otel_setup import setup_otel_tracing


def process_all_markdowns(base_dir: str, num_docs: int | None = None) -> None:

    # Unified data store for ingestion, caching, and reporting
    content_store = ContentStore(base_dir=base_dir)

    # Define the parameter for this run.
    # If you change this tomorrow, the cache will automatically invalidate and re-run!
    target_tags = cfg.target_tags

    target_tag_examples = {
        tag: load_tag_examples(tag, spec) for tag, spec in cfg.TAG_EXAMPLES.items()
    }

    print(f"Starting batch processing with target_tags: {target_tags}")

    from tagging.pipelines.tag_pipeline_decision_model import (
        TagPipelineDecisionModel,
    )

    pipeline = TagPipelineDecisionModel(
        llm=llamaCppChat(model=cfg.model_chat),
        decision_model=llamacppDecisionModel(model=cfg.model_decision),
        target_tags=target_tags,
        target_tag_examples=target_tag_examples,
        evaluators=[ResultDataValidationEvals()],
        text_chunk_size=cfg.text_chunk_size,
        min_chars_for_summarization=cfg.min_chars_for_summarization,
        category_tag_config=cfg.category_tag_config,
    )

    pipeline_id, graph_dict = pipeline.pipeline_identity()
    print(f"Pipeline ID: {pipeline_id}")

    file_hashes = (
        content_store.file_hashes[:num_docs] if num_docs else content_store.file_hashes
    )

    for file_hash in file_hashes:
        hash_entry = content_store._hashes[file_hash]
        entry_title = hash_entry.get("title") if isinstance(hash_entry, dict) else None
        rel_path = (
            hash_entry.get("path", hash_entry)
            if isinstance(hash_entry, dict)
            else str(hash_entry)
        )

        print(f"\nEvaluating: {rel_path}")

        # Fetch content
        content = content_store.read_content(file_hash)
        if content is None:
            print("  ->[SKIP] Not a valid markdown or file missing.")
            continue

        markdown_document = Document(page_content=content, title=entry_title)

        # Check existing cache for this pipeline's tags
        if content_store.has_valid_cache(file_hash, pipeline_id, target_tags):
            print("  ->[CACHED] All tags present.")
            continue

        print("  ->[MISS] Missing some or all tags")

        final_result = pipeline.invoke(docs=[markdown_document])

        if not final_result.success:
            print(f"  ->[FAIL] Pipeline failed: {final_result.error_message}")
            continue

        content_store.save_tag_results(
            file_hash,
            pipeline_id,
            final_result,
            graph_dict=graph_dict,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process HKEX circulars with tagging pipeline."
    )
    parser.add_argument(
        "--base-dir",
        default=os.path.join(os.getcwd(), "hkex_circulars"),
        help="Directory containing processing_results.json and file_hashes.json (default: ./hkex_circulars)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help='Config profile name (without .py), e.g. "full_tags".',
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=None,
        help="Number of documents to process (default: all).",
    )
    args = parser.parse_args()

    if args.config:
        cfg.use_profile(args.config)

    setup_otel_tracing(log_dir=os.path.join(args.base_dir, "traces"))
    process_all_markdowns(args.base_dir, num_docs=args.num_docs)
