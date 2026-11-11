from __future__ import annotations

import json
from pathlib import Path

import pytest

from tagging.flaskapp import services


def _make_topic_dir(
    tmp_path: Path,
    topic_name: str,
    articles: list[dict],
) -> Path:
    """Create a minimal topic directory with processing_results.json and file_hashes.json."""
    topic_dir = tmp_path / topic_name
    topic_dir.mkdir()

    hashes: dict[str, object] = {}
    results: dict[str, dict] = {}

    for art in articles:
        fh = art["file_hash"]
        pid = art["pipeline_id"]

        rel_path = f"{topic_name}/{art.get('filename', 'article.md')}"
        content_file = topic_dir / rel_path
        content_file.parent.mkdir(parents=True, exist_ok=True)
        content_file.write_text(art.get("content", "Dummy article"), encoding="utf-8")

        hashes[fh] = {
            "path": rel_path,
            "processed_timestamp": art.get("processed_ts", "2024-06-15T09:00:00+00:00"),
            "title": art.get("title", f"Article {fh[:8]}"),
            "source_url": art.get("source_url", ""),
        }

        tag_results = {}
        for tag_name, score in art["tags"].items():
            tag_results[tag_name] = {
                "output_data": {tag_name: score},
                "metadata": {"trace_id": art.get("trace_id", f"trace_{fh[:8]}")},
                "success": True,
            }

        results.setdefault(fh, {})[pid] = {
            "pipeline_id": pid,
            "processed_timestamp": art.get("result_ts", "2024-06-15T10:00:00+00:00"),
            "tag_results": tag_results,
        }

    (topic_dir / "file_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (topic_dir / "processing_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return topic_dir


@pytest.fixture()
def data_store(tmp_path: Path) -> Path:
    """Creates a minimal data store with two topics for service-level tests."""
    _make_topic_dir(
        tmp_path,
        "hkex_circulars",
        [
            {
                "file_hash": "a" * 64,
                "pipeline_id": "tagger_v1",
                "tags": {"TradingHoursAmendment": "High", "MarketImpact": "Medium"},
                "title": "HKEX Circular 001/2024",
                "filename": "hkex_circulars/circular_001.md",
                "content": "This circular announces changes to trading hours effective 1 July 2024.",
                "trace_id": "trace_aaaaaa",
            },
        ],
    )

    _make_topic_dir(
        tmp_path,
        "bybit_announcements",
        [
            {
                "file_hash": "b" * 64,
                "pipeline_id": "tagger_v1",
                "tags": {"TradingHoursAmendment": "High"},
                "title": "Bybit Maintenance Notice",
                "filename": "bybit_announcements/maint_001.md",
                "content": "System maintenance on 1 August 2024.",
                "trace_id": "trace_bbbbbb",
            },
        ],
    )

    return tmp_path


# ── Service tests (unit, no external files) ────────────────────────────

def test_list_available_topics(data_store: Path):
    """Verifies available topics are discovered from the data store directory."""
    topics = services.list_available_topics(str(data_store))
    topic_names = {t["name"] for t in topics}
    assert "hkex_circulars" in topic_names


def test_get_tags_for_topics(data_store: Path):
    """Verifies tag names are extracted from processing results for a given topic."""
    tags = services.get_tags_for_topics(["hkex_circulars"], str(data_store))
    assert tags == ["MarketImpact", "TradingHoursAmendment"]


def test_run_report_filters_by_tag_and_score(data_store: Path):
    """Verifies run_report filters articles by specific tag name and score level."""
    articles = services.run_report(
        topics=["hkex_circulars"],
        filter_tags=["TradingHoursAmendment"],
        target_levels=["High"],
        custom_path=str(data_store),
    )
    assert len(articles) == 1
    matched = articles[0]["matched_tags"]
    assert matched[0]["tag"] == "TradingHoursAmendment"
    assert matched[0]["score"] == "High"


def test_run_report_includes_all_matching_scores(data_store: Path):
    """Verifies run_report collects all tags matching any of the given target scores."""
    articles = services.run_report(
        topics=["hkex_circulars"],
        target_levels=["High", "Medium"],
        custom_path=str(data_store),
    )
    tags = {m["tag"] for m in articles[0]["matched_tags"]}
    assert tags == {"MarketImpact", "TradingHoursAmendment"}
