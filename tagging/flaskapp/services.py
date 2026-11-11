from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ingestion.content_store import ContentStore  # noqa: E402


def resolve_data_store_dir(custom_path: str | None = None) -> Path:
    """Resolves the root data directory.

    Priority:
    1. custom_path argument
    2. DATA_STORE_DIR environment variable
    3. Default: <project_root>/data_store
    """
    if custom_path:
        p = Path(custom_path)
    elif os.environ.get("DATA_STORE_DIR"):
        p = Path(os.environ["DATA_STORE_DIR"])
    else:
        p = _PROJECT_ROOT / "data_store"

    return p.resolve()


def list_available_topics(custom_path: str | None = None) -> list[dict[str, Any]]:
    """Lists all topic directories under the root data directory.

    Returns a list of dicts: [{'name': topic_name, 'has_results': bool, 'article_count': int}]
    """
    root_dir = resolve_data_store_dir(custom_path)
    if not root_dir.exists() or not root_dir.is_dir():
        return []

    topics: list[dict[str, Any]] = []
    for item in sorted(root_dir.iterdir()):
        if item.is_dir() and not item.name.startswith((".", "_")):
            results_path = item / "processing_results.json"
            has_results = results_path.exists()
            article_count = 0
            if has_results:
                try:
                    with open(results_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    article_count = len(data)
                except Exception:
                    article_count = 0

            topics.append(
                {
                    "name": item.name,
                    "has_results": has_results,
                    "article_count": article_count,
                }
            )

    return topics


def get_tags_for_topics(
    topics: list[str], custom_path: str | None = None
) -> list[str]:
    """Scans processing_results.json for the specified topics and returns all distinct tag names."""
    root_dir = resolve_data_store_dir(custom_path)
    found_tags: set[str] = set()

    for topic in topics:
        results_file = root_dir / topic / "processing_results.json"
        if not results_file.exists():
            continue

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for file_hash, pipeline_dict in data.items():
                if not isinstance(pipeline_dict, dict):
                    continue
                # pipeline_dict can be nested by pipeline_id or flat
                for p_key, p_val in pipeline_dict.items():
                    if isinstance(p_val, dict) and "tag_results" in p_val:
                        found_tags.update(p_val["tag_results"].keys())
                    elif p_key == "tag_results" and isinstance(p_val, dict):
                        found_tags.update(p_val.keys())
        except Exception as e:
            print(f"Error reading tags from {results_file}: {e}", file=sys.stderr)

    return sorted(found_tags)


def run_report(
    topics: list[str],
    filter_tags: list[str] | None = None,
    since_date: str | None = None,
    target_levels: list[str] | None = None,
    custom_path: str | None = None,
) -> list[dict[str, Any]]:
    """Runs report query across multiple topics following ContentStore filtering logic.

    Returns a deduplicated list of matching articles (Option B), each with its matched tags.
    """
    if target_levels is None:
        target_levels = ["High"]

    root_dir = resolve_data_store_dir(custom_path)
    articles: list[dict[str, Any]] = []

    for topic in topics:
        topic_dir = root_dir / topic
        if not topic_dir.exists() or not (topic_dir / "processing_results.json").exists():
            continue

        store = ContentStore(base_dir=str(topic_dir))
        normalized_results = store._results_by_pipeline

        for file_hash, pipeline_entries in normalized_results.items():
            candidate = store._pick_latest_entry_by_timestamp(pipeline_entries)
            if not isinstance(candidate, dict):
                continue

            pipeline_tag_results = candidate.get("tag_results", {})
            if not pipeline_tag_results:
                continue

            hash_entry = store._hashes.get(file_hash, {})
            if not isinstance(hash_entry, dict):
                hash_entry = {}

            processed_timestamp = (
                hash_entry.get("processed_timestamp")
                or candidate.get("processed_timestamp", "")
            )
            if since_date and processed_timestamp[:10] < since_date:
                continue

            # Check tags matching target_levels
            matched_tags: list[dict[str, str]] = []
            has_filter_match = False

            trace_id = None
            for tag_key, current_tag_entry in pipeline_tag_results.items():
                if not isinstance(current_tag_entry, dict):
                    continue

                if not trace_id:
                    trace_id = current_tag_entry.get("metadata", {}).get("trace_id")

                tag_data = current_tag_entry.get("output_data") or {}
                tag_score = tag_data.get(tag_key)

                if tag_score in target_levels:
                    is_in_filter = not filter_tags or (tag_key in filter_tags)
                    if is_in_filter:
                        has_filter_match = True
                    matched_tags.append(
                        {
                            "tag": tag_key,
                            "score": tag_score,
                            "is_filter_match": is_in_filter,
                        }
                    )

            if not has_filter_match or not matched_tags:
                continue

            file_path = hash_entry.get("path", "")
            title = hash_entry.get("title")
            if not title:
                title = os.path.splitext(os.path.basename(file_path))[0] if file_path else file_hash

            articles.append(
                {
                    "topic": topic,
                    "file_hash": file_hash,
                    "title": title,
                    "summary": candidate.get("summary") or "N/A",
                    "effective_date": candidate.get("effectivedate") or "NONE",
                    "source_url": hash_entry.get("source_url") or "",
                    "file_path": file_path,
                    "processed_timestamp": processed_timestamp[:10] if processed_timestamp else "",
                    "pipeline_id": candidate.get("pipeline_id") or "",
                    "trace_id": trace_id or "",
                    "matched_tags": sorted(matched_tags, key=lambda x: x["tag"]),
                }
            )

    # Sort articles: newest processed first, then title
    articles.sort(
        key=lambda x: (x["processed_timestamp"], x["title"]),
        reverse=True,
    )
    return articles


def get_article_details(
    topic: str, file_hash: str, custom_path: str | None = None
) -> dict[str, Any] | None:
    """Retrieves full article details for the article inspection page."""
    root_dir = resolve_data_store_dir(custom_path)
    topic_dir = root_dir / topic
    if not topic_dir.exists():
        return None

    store = ContentStore(base_dir=str(topic_dir))
    hash_entry = store._hashes.get(file_hash)
    pipeline_entries = store._results_by_pipeline.get(file_hash)

    # Return None if the file_hash is unknown to this topic
    if hash_entry is None and pipeline_entries is None:
        return None

    if not isinstance(hash_entry, dict):
        hash_entry = {}
    if not isinstance(pipeline_entries, dict):
        pipeline_entries = {}

    candidate = store._pick_latest_entry_by_timestamp(pipeline_entries)
    if not isinstance(candidate, dict):
        candidate = {}

    file_path = hash_entry.get("path", "")
    title = hash_entry.get("title")
    if not title:
        title = os.path.splitext(os.path.basename(file_path))[0] if file_path else file_hash

    # Find trace_id
    trace_id = None
    tag_results = candidate.get("tag_results", {})
    for tag_name, tag_val in tag_results.items():
        if isinstance(tag_val, dict) and not trace_id:
            trace_id = tag_val.get("metadata", {}).get("trace_id")

    # Read raw content
    raw_content = store.read_content(file_hash)
    if raw_content is None and file_path:
        # Check if file physically exists even if not .md/.txt/.csv
        physical_path = topic_dir / file_path
        if physical_path.exists():
            if physical_path.suffix.lower() in (".md", ".txt", ".json", ".csv"):
                try:
                    with open(physical_path, "r", encoding="utf-8") as f:
                        raw_content = f.read()
                except Exception as e:
                    raw_content = f"Error reading file content: {e}"
            else:
                raw_content = f"[Binary file: {physical_path.name} ({physical_path.suffix})]"

    return {
        "topic": topic,
        "file_hash": file_hash,
        "title": title,
        "source_url": hash_entry.get("source_url") or "",
        "file_path": file_path,
        "processed_timestamp": hash_entry.get("processed_timestamp") or candidate.get("processed_timestamp", ""),
        "pipeline_id": candidate.get("pipeline_id") or "",
        "trace_id": trace_id or "",
        "summary": candidate.get("summary") or "N/A",
        "effective_date": candidate.get("effectivedate") or "NONE",
        "tag_results": tag_results,
        "raw_content": raw_content,
        "topic_dir": str(topic_dir),
    }
