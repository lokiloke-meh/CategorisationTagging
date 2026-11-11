from __future__ import annotations

import tempfile
from typing import Any

from prompt_chains.base_chains import ChainResult
from utils.llm_models import LLMResult


class _DeterministicMockLLM:
    """Returns fixed outputs for all chain types."""

    def __init__(self, result: dict[str, Any] | str | None = None) -> None:
        self.result = result if result is not None else {}

    def chat_with_expected_type_result(
        self, chat_payload: dict, result_type: str
    ) -> LLMResult:
        return LLMResult(
            model_name="mock",
            user_input="",
            system_prompt="",
            model_result=self.result,
            success=True,
        )


# ── Chains: empty-doc guard ────────────────────────────────────────────


def test_categorisation_chains_fail_on_empty_docs() -> None:
    """All categorisation chains return success=False when invoked with no documents."""
    from tagging.category_tagging_all_at_once import AllAtOnceCategorisation
    from tagging.category_tagging_one_at_a_time import OneAtATimeCategorisation

    for chain in (
        AllAtOnceCategorisation(llm=_DeterministicMockLLM(), target_tags=["tag"]),
        OneAtATimeCategorisation(llm=_DeterministicMockLLM(), target_tags=["tag"]),
    ):
        result: ChainResult = chain.invoke([])
        assert result.success is False


# ── ContentStore: tag scoring lookup (nested dict traversal) ───────────


def test_find_by_tags_score_returns_correct_hashes() -> None:
    """Only hashes matching target levels are returned through per-tag lookups."""
    from ingestion.content_store import ContentStore
    from tagging.utils.data_models import TagResult

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ContentStore(base_dir=tmpdir)

        # High score → should be found
        high_result = ChainResult(
            input_data="test",
            output_data=TagResult(tags={"MarketWideImpact": "High"}),
            chain_name="test_chain",
            success=True,
        )
        store.save_tag_results("hash1", "pid", high_result, graph_dict={})

        # Low score → should NOT be found
        low_result = ChainResult(
            input_data="test",
            output_data=TagResult(tags={"MarketWideImpact": "Low"}),
            chain_name="test_chain",
            success=True,
        )
        store.save_tag_results("hash2", "pid", low_result, graph_dict={})

        assert set(store.find_by_tags_score("MarketWideImpact", ["High"])) == {"hash1"}


# ── ContentStore: report deduplication ────────────────────────────────


def test_report_skips_already_reported_items() -> None:
    """Second report with same pipeline returns None — items already in tracking are skipped.

    Third report with a different pipeline returns content again — no cross-pipeline dedup.
    """
    from ingestion.content_store import ContentStore
    from tagging.utils.data_models import TagResult

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ContentStore(base_dir=tmpdir)

        store._hashes["h1"] = {
            "path": "test.md",
            "source_url": "https://example.com/1",
            "title": "Test Doc",
        }
        result = ChainResult(
            input_data="test",
            output_data=TagResult(tags={"MarketWideImpact": "High"}, summary="S"),
            chain_name="test_chain",
            success=True,
        )
        store.save_tag_results("h1", "pid_a", result, graph_dict={})
        store.save_tag_results("h1", "pid_b", result, graph_dict={})

        # First report with pid_a — should produce output
        report1 = store.generate_report(
            target_levels=["High"], output_file=f"{tmpdir}/r.md",
            pipeline_id="pid_a",
        )
        assert "# SUMMARY REPORT" in report1 and "Test Doc" in report1

        # Second report with same pid_a — dedup kicks in, returns None
        assert (
            store.generate_report(target_levels=["High"], output_file=f"{tmpdir}/r2.md", pipeline_id="pid_a")
            is None
        )

        # Third report with different pipeline — no cross-pipeline contamination
        report3 = store.generate_report(
            target_levels=["High"], output_file=f"{tmpdir}/r3.md",
            pipeline_id="pid_b",
        )
        assert "# SUMMARY REPORT" in report3 and "Test Doc" in report3


# ── ContentStore: TagResult validation & merge behavior ───────────────


def test_save_tag_results_validates_tagresult_output() -> None:
    """save_tag_results raises TypeError when output_data is not a TagResult."""
    from ingestion.content_store import ContentStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ContentStore(base_dir=tmpdir)

        # Non-TagResult output should raise TypeError with chain name + type info
        bad_result = ChainResult(
            input_data="test",
            output_data={"tags": {"MarketWideImpact": "High"}},
            chain_name="bad_chain",
            success=True,
        )
        import pytest

        with pytest.raises(TypeError, match="bad_chain.*dict"):
            store.save_tag_results("hash1", "pid", bad_result, graph_dict={})


def test_save_tag_results_merges_tags_across_runs() -> None:
    """Saving different tags for the same file accumulates them in tag_results."""
    from ingestion.content_store import ContentStore
    from tagging.utils.data_models import TagResult

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ContentStore(base_dir=tmpdir)

        # First run: save MarketWideImpact tag
        result_a = ChainResult(
            input_data="test",
            output_data=TagResult(tags={"MarketWideImpact": "High"}),
            chain_name="test_chain",
            success=True,
        )
        store.save_tag_results("hash1", "pid", result_a, graph_dict={})

        # Second run: save Relevance tag (same file, same pipeline)
        result_b = ChainResult(
            input_data="test",
            output_data=TagResult(tags={"Relevance": "High"}),
            chain_name="test_chain",
            success=True,
        )
        store.save_tag_results("hash1", "pid", result_b, graph_dict={})

        # Both tags should be present under the pipeline key
        pipeline_entry = store._results["hash1"]["pid"]
        assert "MarketWideImpact" in pipeline_entry["tag_results"]
        assert "Relevance" in pipeline_entry["tag_results"]
