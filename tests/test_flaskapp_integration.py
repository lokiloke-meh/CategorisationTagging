from __future__ import annotations

from pathlib import Path

import pytest

from tagging.flaskapp.app import create_app

DATA_STORE = Path(__file__).parent / "flaskapp_integration_samples"
TOPIC_NAME = "test_topic"
ARTICLE_HASH = "5307f68d4ddd582102780a78ad1e4cd44003dd89e401ad6600726fe268b6e991"


@pytest.fixture()
def client():
    app = create_app(str(DATA_STORE))
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


# ── Index & API ─────────────────────────────────────────────────────────

@pytest.mark.integration
def test_index_shows_topic(client):
    """Verifies the index page loads and contains the app title."""
    res = client.get("/")
    assert res.status_code == 200
    assert b"Notice & Tagging Visualizer" in res.data


@pytest.mark.integration
def test_index_query_returns_results(client):
    """Verifies querying by topic returns a populated report results section."""
    res = client.get(f"/?topics={TOPIC_NAME}")
    assert res.status_code == 200
    assert b"Report Results" in res.data


@pytest.mark.integration
def test_api_tags(client):
    """Verifies the /api/tags endpoint returns known tag names for real article data."""
    res = client.get(f"/api/tags?topics={TOPIC_NAME}")
    assert res.status_code == 200
    data = res.get_json()
    assert "tags" in data
    tag_names = set(data["tags"])
    assert "OrderInfrastructureChanges" in tag_names
    assert "AllCorporateActions" in tag_names


# ── Article detail page — three tabs render properly ───────────────────

@pytest.mark.integration
def test_article_detail_page(client):
    """Verifies the article detail page loads with all three tab sections and their iframes."""
    res = client.get(f"/article/{TOPIC_NAME}/{ARTICLE_HASH}")
    assert res.status_code == 200
    body = res.data.decode()

    # Tab 1: Trace Gantt Timeline — iframe loads the trace chart
    assert "Trace Gantt Timeline" in body or "gantt" in body.lower()
    assert f'/article/{TOPIC_NAME}/{ARTICLE_HASH}/trace' in body

    # Tab 2: Pipeline Architecture — iframe loads pipeline DAG
    assert "Pipeline Architecture" in body or "pipeline" in body.lower()
    assert f'/article/{TOPIC_NAME}/{ARTICLE_HASH}/pipeline' in body

    # Tab 3: Raw Document Content — inline, not an iframe
    assert "Raw Document Content" in body
    assert "BounceBit" in body  # title/content snippet


@pytest.mark.integration
def test_article_trace_embed(client):
    """Verifies the trace Gantt tab renders Plotly content from real parquet span data."""
    res = client.get(f"/article/{TOPIC_NAME}/{ARTICLE_HASH}/trace")
    assert res.status_code == 200
    body = res.data.decode()

    # Should have the Plotly chart container (not an empty/no-trace fallback)
    assert "plotly-gantt" in body
    # If trace data exists, should contain span task names, not "No Trace Data"
    assert "No Trace Data" not in body


@pytest.mark.integration
def test_article_pipeline_embed(client):
    """Verifies the pipeline tab renders a DAG from the registry without fallback messages."""
    res = client.get(f"/article/{TOPIC_NAME}/{ARTICLE_HASH}/pipeline")
    assert res.status_code == 200
    body = res.data.decode()

    # Should contain a rendered pipeline visualization, not registry-not-found fallback
    assert "Pipeline Registry Not Found" not in body
    assert "Empty Pipeline Registry" not in body


# ── Error cases ────────────────────────────────────────────────────────

@pytest.mark.integration
def test_article_not_found(client):
    """Verifies the article route returns 404 for a nonexistent hash."""
    res = client.get(f"/article/{TOPIC_NAME}/nonexistenthash123")
    assert res.status_code == 404
