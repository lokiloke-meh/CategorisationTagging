from __future__ import annotations

# ── Fixtures ────────────────────────────────────────────────────────────────


class _MockChatAdapter:
    """Minimal ChatAdapter-like object for pipeline_identity testing."""

    def __init__(
        self, model_name: str = "test-model", url: str = "http://localhost:8080"
    ) -> None:
        self.model_name = model_name
        self.url = url


class _MockDecisionModel:
    """Minimal DecisionModel-like object for pipeline_identity testing."""

    def __init__(
        self, model_name: str = "dm-v1", url: str = "http://localhost:8081"
    ) -> None:
        self.model_name = model_name
        self.url = url

    def make_decision(self, state: dict, questions: list) -> object:
        raise NotImplementedError


# ── TagPipeline tests ───────────────────────────────────────────────────────


def _make_tag_pipeline(**overrides):
    from tagging.pipelines.tag_pipeline import TagPipeline

    base = {
        "llm": _MockChatAdapter(model_name="test-model"),
        "target_tags": ["TagA", "TagB"],
        "text_chunk_size": 800,
    }
    base.update(overrides)
    return TagPipeline(**base)


class TestPipelineIdentityTagPipeline:

    def test_hash_changes_when_llm_model_name_changes(self):
        p1 = _make_tag_pipeline()
        p2 = _make_tag_pipeline(llm=_MockChatAdapter(model_name="different-model"))
        h1, g1 = p1.pipeline_identity()
        h2, g2 = p2.pipeline_identity()
        assert h1 != h2, "Different LLM model names should produce different hashes"
        assert g1["pipeline"]["llm"]["model_name"] == "test-model"
        assert g2["pipeline"]["llm"]["model_name"] == "different-model"

    def test_hash_changes_when_target_tags_change(self):
        p1 = _make_tag_pipeline()
        p2 = _make_tag_pipeline(target_tags=["TagA", "TagC"])
        h1, g1 = p1.pipeline_identity()
        h2, g2 = p2.pipeline_identity()
        assert h1 != h2, "Different target tags should produce different hashes"
        assert g1["pipeline"]["target_tags"] == ["TagA", "TagB"]
        assert g2["pipeline"]["target_tags"] == ["TagA", "TagC"]

    def test_hash_changes_when_text_chunk_size_changes(self):
        p1 = _make_tag_pipeline()
        p2 = _make_tag_pipeline(text_chunk_size=400)
        h1, g1 = p1.pipeline_identity()
        h2, g2 = p2.pipeline_identity()
        assert h1 != h2, "Different text_chunk_size should produce different hashes"
        assert g1["pipeline"]["text_chunk_size"] == 800
        assert g2["pipeline"]["text_chunk_size"] == 400

    def test_same_config_produces_same_hash(self):
        p1 = _make_tag_pipeline()
        p2 = _make_tag_pipeline()
        h1, g1 = p1.pipeline_identity()
        h2, g2 = p2.pipeline_identity()
        assert h1 == h2, "Identical config should produce the same hash"
        assert g1 == g2


# ── TagPipelineDecisionModel tests ──────────────────────────────────────────


def _make_tag_pipeline_dm(**overrides):
    from tagging.pipelines.tag_pipeline_decision_model import (
        TagPipelineDecisionModel,
    )

    base = {
        "llm": _MockChatAdapter(model_name="test-model"),
        "decision_model": _MockDecisionModel(model_name="dm-v1"),
        "target_tags": ["TagA", "TagB"],
        "text_chunk_size": 800,
    }
    base.update(overrides)
    return TagPipelineDecisionModel(**base)


class TestPipelineIdentityTagPipelineDecisionModel:

    def test_hash_changes_when_decision_model_changes(self):
        p1 = _make_tag_pipeline_dm()
        p2 = _make_tag_pipeline_dm(
            decision_model=_MockDecisionModel(model_name="dm-v2")
        )
        h1, g1 = p1.pipeline_identity()
        h2, g2 = p2.pipeline_identity()
        assert h1 != h2, "Different decision models should produce different hashes"
        assert g1["pipeline"]["decision_model"]["model_name"] == "dm-v1"
        assert g2["pipeline"]["decision_model"]["model_name"] == "dm-v2"

    def test_hash_changes_when_llm_model_name_changes(self):
        p1 = _make_tag_pipeline_dm()
        p2 = _make_tag_pipeline_dm(llm=_MockChatAdapter(model_name="different-model"))
        h1, g1 = p1.pipeline_identity()
        h2, g2 = p2.pipeline_identity()
        assert h1 != h2, "Different LLM model names should produce different hashes"
        assert g1["pipeline"]["llm"]["model_name"] == "test-model"
        assert g2["pipeline"]["llm"]["model_name"] == "different-model"

    def test_hash_changes_when_target_tags_change(self):
        p1 = _make_tag_pipeline_dm()
        p2 = _make_tag_pipeline_dm(target_tags=["TagA", "TagC"])
        h1, g1 = p1.pipeline_identity()
        h2, g2 = p2.pipeline_identity()
        assert h1 != h2, "Different target tags should produce different hashes"
        assert g1["pipeline"]["target_tags"] == ["TagA", "TagB"]
        assert g2["pipeline"]["target_tags"] == ["TagA", "TagC"]

    def test_same_config_produces_same_hash(self):
        p1 = _make_tag_pipeline_dm()
        p2 = _make_tag_pipeline_dm()
        h1, g1 = p1.pipeline_identity()
        h2, g2 = p2.pipeline_identity()
        assert h1 == h2, "Identical config should produce the same hash"
        assert g1 == g2
