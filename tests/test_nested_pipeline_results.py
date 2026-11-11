from __future__ import annotations

import tempfile


def test_save_isolates_pipelines() -> None:
    """Same hash, two pipelines → each has its own tag set via has_valid_cache."""
    from ingestion.content_store import ContentStore
    from prompt_chains.base_chains import ChainResult
    from tagging.utils.data_models import TagResult

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ContentStore(base_dir=tmpdir)

        result_a = ChainResult(
            input_data="test",
            output_data=TagResult(tags={"MarketWideImpact": "High"}),
            chain_name="c1",
            success=True,
        )
        result_b = ChainResult(
            input_data="test",
            output_data=TagResult(tags={"Relevance": "Low"}),
            chain_name="c2",
            success=True,
        )

        store.save_tag_results("hashA", "pid_a", result_a, graph_dict={})
        store.save_tag_results("hashA", "pid_b", result_b, graph_dict={})

        assert store.has_valid_cache("hashA", "pid_a", ["MarketWideImpact"]) is True
        assert store.has_valid_cache("hashA", "pid_b", ["Relevance"]) is True
        assert store.has_valid_cache("hashA", "pid_b", ["MarketWideImpact"]) is False


def test_find_by_tags_score_uses_latest_pipeline() -> None:
    """Same hash, pid_a=High then pid_b=Low → find returns Low's result only."""
    from ingestion.content_store import ContentStore
    from prompt_chains.base_chains import ChainResult
    from tagging.utils.data_models import TagResult

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ContentStore(base_dir=tmpdir)

        result_a = ChainResult(
            input_data="test",
            output_data=TagResult(tags={"MarketWideImpact": "High"}),
            chain_name="c1",
            success=True,
        )
        import time
        time.sleep(0.05)

        result_b = ChainResult(
            input_data="test",
            output_data=TagResult(tags={"MarketWideImpact": "Low"}),
            chain_name="c2",
            success=True,
        )

        store.save_tag_results("hashA", "pid_a", result_a, graph_dict={})
        store.save_tag_results("hashA", "pid_b", result_b, graph_dict={})

        assert set(store.find_by_tags_score("MarketWideImpact", ["High"])) == set()
        assert set(store.find_by_tags_score("MarketWideImpact", ["Low"])) == {"hashA"}
