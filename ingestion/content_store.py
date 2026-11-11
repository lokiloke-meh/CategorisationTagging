"""Notice data store that manages the full lifecycle of a notice:
  1. Ingestion — hash file content, detect duplicates, read saved files
  2. Processing — cache results keyed by file hash + pipeline config
  3. Reporting — track which hashes/tags have already been included in reports

Usage::

    store = ContentStore(base_dir="/path/to/data")
    # Ingestion
    file_hash = store.hash_text(text)
    if store.check_duplicate(file_hash):
        return  # already processed

    # Processing (per-tag caching, additive across runs)
    pipeline_id, graph_dict = pipeline.pipeline_identity()
    chain_result = pipeline.invoke(docs=[document])
    store.save_tag_results(file_hash, pipeline_id, chain_result, graph_dict)

    # Reporting
    for hash_id in store.find_by_tags_score("MarketWideImpact", ["High"]):
        ...  # generate report entries

"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import UTC, datetime

from config import cfg
from prompt_chains.base_chains import ChainResult
from tagging.utils.data_models import TagResult


class ContentStore:
    """Notice data store for ingestion, processing cache, and reporting.

    Args:
        base_dir: Directory containing all data files (hashes.json, results.json, etc.)
            Also used as the root folder where original scraped files are stored.

    Attributes:
        hashes_filepath: Path to the file_hashes.json mapping hash → file metadata.
        results_filepath: Path to the processing_results.json cache of chain outputs.
        tracking_filepath: Path to the reported_items.json deduplication tracker.
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.hashes_filepath = os.path.join(base_dir, "file_hashes.json")
        self.results_filepath = os.path.join(base_dir, "processing_results.json")
        self.pipeline_registry_filepath = os.path.join(
            base_dir, "pipeline_registry.json"
        )
        self._hashes: dict[str, object] = self._load_json(self.hashes_filepath)
        self._results: dict[str, object] = self._load_json(self.results_filepath)
        self._registries: dict[str, object] = self._load_json(
            self.pipeline_registry_filepath
        )

    # ── File hashing & ingestion ────────────────────────────────────────

    def hash_text(self, text: str, ignore_pattern: str | None = None) -> str:
        """Hash text content. Optionally strips dynamic content before hashing."""
        if ignore_pattern:
            text = re.sub(ignore_pattern, "", text)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def hash_bytes(self, data: bytes) -> str:
        """Hash raw binary data (e.g., PDF file bytes)."""
        return hashlib.sha256(data).hexdigest()

    def check_duplicate(
        self, file_hash: str, min_timestamp: str | None = None
    ) -> str | None:
        """Check if the file hash is known AND the file physically exists.

        Returns the relative filepath if valid duplicate, else None.
        If min_timestamp provided, records older than that are ignored.
        """
        entry = self._hashes.get(file_hash)
        if not isinstance(entry, dict):
            return None

        rel_path = entry.get("path")
        ts = entry.get("processed_timestamp")

        if min_timestamp and (not ts or ts < min_timestamp):
            return None

        full_path = os.path.join(self.base_dir, rel_path)
        if os.path.exists(full_path):
            return rel_path
        return None

    def mark_saved(
        self,
        file_hash: str,
        relative_path: str,
        source_url: str | None = None,
        title: str | None = None,
    ) -> None:
        """Register a newly saved file's hash, path, and metadata."""
        self._hashes[file_hash] = {
            "path": relative_path,
            "processed_timestamp": datetime.now(UTC).isoformat(),
            "source_url": source_url,
            "title": title,
        }
        self._save_json(self.hashes_filepath)

    def read_content(self, file_hash: str) -> str | None:
        """Return text content for the given hash. Returns None if unknown or unreadable."""
        relative_path = self.check_duplicate(file_hash)
        if not relative_path or not any(
            relative_path.lower().endswith(ext) for ext in (".md", ".txt", ".csv")
        ):
            return None
        full_path = os.path.join(self.base_dir, relative_path)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    # ── Processing cache ────────────────────────────────────────────────

    def save_tag_results(
        self,
        file_hash: str,
        pipeline_id: str,
        chain_result: ChainResult,
        graph_dict: dict,
    ) -> None:
        """Save processing results keyed by file hash and pipeline.

        Validates output_data is TagResult, merges per-tag results with any previous
        run of the same pipeline for this hash (preserving old tags), generates timestamp,
        persists to disk, updates registry.

        Args:
            file_hash: SHA-256 hash of the input document.
            pipeline_id: Opaque identifier for the pipeline config used.
            chain_result: ChainResult from a tagging chain (output_data must be TagResult).
            graph_dict: Serialized execution graph from pipeline_identity() for registry entry.

        Raises:
            TypeError: If output_data is not a TagResult instance.
        """
        if not isinstance(chain_result.output_data, TagResult):
            actual = type(chain_result.output_data).__name__
            raise TypeError(
                f"Chain '{chain_result.chain_name}' must produce TagResult "
                f"in output_data, got {actual}"
            )

        chain_dict = chain_result.to_dict()
        result_output = chain_dict.get("output_data", {})

        # Extract per-tag results from the chain output (no filtering — preserve all)
        tag_scores = result_output.get("tags") or {}
        new_tag_results: dict[str, dict] = {
            tag_name: {
                "output_data": {tag_name: score},
                "metadata": chain_dict.get("metadata", {}),
                "success": bool(chain_dict.get("success", True)),
                "error_message": chain_dict.get("error_message"),
            }
            for tag_name, score in tag_scores.items()
        }

        # Merge new tags into previous run of the same pipeline for this hash
        existing_pipeline_entry = (
            self._results.setdefault(file_hash, {}).get(pipeline_id) or {}
        )
        merged_tag_results = {
            **(existing_pipeline_entry.get("tag_results") or {}),
            **new_tag_results,
        }

        result_entry: dict[str, object] = {
            "pipeline_id": pipeline_id,
            "processed_timestamp": datetime.now(UTC).isoformat(),
            "tag_results": merged_tag_results,
        }
        for config_key in ("summary", "effectivedate"):
            if value := result_output.get(config_key):
                result_entry[config_key] = value
        self._results[file_hash][pipeline_id] = result_entry
        self._save_json(self.results_filepath)

        # Update pipeline registry with graph metadata
        now_iso = datetime.now(UTC).isoformat()
        reg_entry: dict[str, object] = self._registries.get(pipeline_id) or {}
        if "file_count" in reg_entry:
            reg_entry["file_count"] += 1
        else:
            reg_entry["first_used"] = now_iso
            reg_entry["class_name"] = chain_result.output_data.__class__.__name__
            reg_entry["file_count"] = 1
        reg_entry["last_used"] = now_iso
        reg_entry["chain_graph"] = graph_dict
        self._registries[pipeline_id] = reg_entry
        self._save_json(self.pipeline_registry_filepath)

    @property
    def _results_by_pipeline(self) -> dict[str, dict]:
        """Adapter: present results uniformly regardless of on-disk format.

        Two possible structures may exist under self._results:

        **Legacy flat** (single run per hash):
            {hash → {pipeline_id: "abc", processed_timestamp: "...", tag_results: {...}}}
            Detection: "pipeline_id" in raw_value is True — it's a metadata field.

        **New nested** (multiple runs per hash):
            {hash → {"abc": {processed_timestamp: "...", tag_results: {...}, pipeline_id: "abc"}}}
            Detection: the keys are pipeline IDs, not metadata fields.

        Detects legacy flat entries (where `pipeline_id` is a metadata key instead
        of a dictionary key), converts them in-place to nested format `{hash → {pid → entry}}`,
        and persists the cleaned state to disk. Subsequent reads see only clean data.
        """
        needs_save = False

        for file_hash, hash_entries in self._results.items():
            if not isinstance(hash_entries, dict):
                continue

            # Legacy flat entry: pipeline_id is a metadata field (string value at top level)
            legacy_pid = (
                hash_entries.get("pipeline_id")
                if isinstance(hash_entries.get("pipeline_id"), str)
                else None
            )
            if legacy_pid is not None:
                # Build new dict: legacy entry under its own key, plus any already-nested entries
                cleaned: dict[str, object] = {
                    legacy_pid: {k: v for k, v in hash_entries.items()}
                }
                for k, v in list(hash_entries.items()):
                    if isinstance(v, dict) and "pipeline_id" in v:
                        cleaned[k] = v
                self._results[file_hash] = cleaned
                needs_save = True

        if needs_save:
            self._save_json(self.results_filepath)

        return self._results

    def find_by_tags_score(
        self,
        tag_name: str,
        target_levels: list[str] | None = None,
        pipeline_id: str | None = None,
    ) -> list[str]:
        """Return file hashes where the latest entry has a specific tag level.

        If ``pipeline_id`` is given, only that pipeline's entries are considered.
        Otherwise picks the most recent entry per hash by ``processed_timestamp``.
        Default target_levels comes from config.cfg.default_target_levels.
        """
        if target_levels is None:
            target_levels = cfg.default_target_levels

        uniform_results = self._results_by_pipeline
        matching_hashes: list[str] = []

        for file_hash, pipeline_entries in uniform_results.items():
            # Select which entries to scan
            if pipeline_id is not None:
                entry_candidates = (
                    {pipeline_id: pipeline_entries[pipeline_id]}
                    if isinstance(pipeline_entries, dict)
                    and pipeline_id in pipeline_entries
                    else {}
                )
            else:
                # Pick latest by timestamp
                latest_candidate_for_hash = self._pick_latest_entry_by_timestamp(
                    pipeline_entries
                )
                entry_candidates = (
                    {"_latest": latest_candidate_for_hash}
                    if isinstance(latest_candidate_for_hash, dict)
                    else {}
                )

            for candidate_entry in entry_candidates.values():
                candidate_tag_results = (
                    candidate_entry.get("tag_results", {})
                    if isinstance(candidate_entry, dict)
                    else {}
                )
                candidate_tag_result = candidate_tag_results.get(tag_name) or {}
                tag_output_data = candidate_tag_result.get("output_data") or {}
                score_value = tag_output_data.get(tag_name)

                if score_value is not None and score_value in target_levels:
                    matching_hashes.append(file_hash)
                    break

        return matching_hashes

    @staticmethod
    def _pick_latest_entry_by_timestamp(entries: dict[str, object]) -> dict | None:
        """Pick the entry with the most recent ``processed_timestamp``."""
        if not entries:
            return None
        latest_candidate: dict | None = None
        for candidate in entries.values():
            ts = (
                candidate.get("processed_timestamp", "")
                if isinstance(candidate, dict)
                else ""
            )
            if ts > (
                latest_candidate.get("processed_timestamp", "")
                if isinstance(latest_candidate, dict)
                else ""
            ):
                latest_candidate = candidate
        return latest_candidate

    def get_latest_result(self, file_hash: str) -> dict | None:
        """Return the latest pipeline entry for a single hash."""
        entries_by_pipeline_id = self._results_by_pipeline.get(file_hash)
        if not isinstance(entries_by_pipeline_id, dict):
            return entries_by_pipeline_id
        return self._pick_latest_entry_by_timestamp(entries_by_pipeline_id)

    def has_valid_cache(
        self,
        file_hash: str,
        pipeline_id: str,
        target_tags: list[str],
    ) -> bool:
        """Return True if a cached result exists for this specific pipeline and covers all target tags."""
        normalized_entries = self._results_by_pipeline
        entries_for_hash = normalized_entries.get(file_hash)
        if not isinstance(entries_for_hash, dict):
            return False
        pipeline_entry = entries_for_hash.get(pipeline_id)
        if not isinstance(pipeline_entry, dict):
            return False
        cached_tag_names = set((pipeline_entry.get("tag_results") or {}).keys())
        return all(tag_name in cached_tag_names for tag_name in target_tags)

    # ── Reporting ───────────────────────────────
    def latest_processed_date(self) -> str:
        """Return the latest processed_timestamp date string from results."""
        return max(
            (
                h.get("processed_timestamp", "")[:10]
                for entries in self._results.values()
                if isinstance(entries, dict)
                for h in entries.values()
                if isinstance(h, dict)
            ),
            default=datetime.now(UTC).strftime("%Y-%m-%d"),
        )

    def get_latest_pipeline_id(self) -> str | None:
        """Return the pipeline_id of the most recently processed result across all hashes."""
        best_pid: str | None = None
        best_ts = ""
        for entries in self._results.values():
            if not isinstance(entries, dict):
                continue
            for pid, entry in entries.items():
                ts = (
                    entry.get("processed_timestamp", "")
                    if isinstance(entry, dict)
                    else ""
                )
                if ts > best_ts:
                    best_ts = ts
                    best_pid = pid
        return best_pid

    def generate_report(
        self,
        target_levels: list[str] | None = None,
        filter_tags: list[str] | None = None,
        output_file: str | None = None,
        since_date: str | None = None,
        pipeline_id: str | None = None,
    ) -> str | None:
        """Generate a markdown report of notices grouped by tags exceeding threshold.

        Args:
            target_levels: Categorical levels to include (default: config.cfg.default_target_levels).
            filter_tags: Tags to filter by (None = all tags).
            output_file: Optional path to write the report. If provided, updates tracking.
            since_date: Only include entries processed on or after this date (YYYY-MM-DD).
            pipeline_id: Limit report to a specific pipeline. Default: latest entry per hash.

        Returns:
            Report content as string, or None if no results found.
        """
        if target_levels is None:
            target_levels = cfg.default_target_levels

        report_by_tag: dict[str, list[dict]] = defaultdict(list)
        normalized_results = self._results_by_pipeline

        for file_hash, pipeline_entries in normalized_results.items():
            # Select which entries to scan
            if pipeline_id is not None:
                candidate_entry_for_hash = (
                    pipeline_entries.get(pipeline_id)
                    if isinstance(pipeline_entries, dict)
                    else None
                )
            else:
                # Pick latest by timestamp
                candidate_entry_for_hash = self._pick_latest_entry_by_timestamp(
                    pipeline_entries
                )
            if not isinstance(candidate_entry_for_hash, dict):
                continue

            pipeline_tag_results = candidate_entry_for_hash.get("tag_results", {})
            if not pipeline_tag_results:
                continue

            # Summary/effectivedate stored at pipeline-entry level
            summary_text = candidate_entry_for_hash.get("summary") or "N/A"
            effective_date_value = (
                candidate_entry_for_hash.get("effectivedate") or "NONE"
            )

            hash_entry = self._hashes.get(file_hash, {})
            if not isinstance(hash_entry, dict):
                continue

            source_url = hash_entry.get("source_url", "No URL available")
            file_path = hash_entry.get("path", "")
            title = hash_entry.get("title")
            if not title:
                title = os.path.splitext(os.path.basename(file_path))[0]

            for tag_key, current_tag_entry in pipeline_tag_results.items():
                if filter_tags and tag_key not in filter_tags:
                    continue

                tag_data = current_tag_entry.get("output_data") or {}
                tag_score_value = tag_data.get(tag_key)

                if tag_score_value is None or tag_score_value not in target_levels:
                    continue

                # Filter by processed timestamp
                if since_date:
                    entry_timestamp = hash_entry.get("processed_timestamp", "")[:10]
                    if entry_timestamp < since_date:
                        continue

                report_by_tag[tag_key].append(
                    {
                        "title": title,
                        "summary": summary_text,
                        "effective_date": effective_date_value,
                        "source_url": source_url,
                        "file_path": file_path,
                        "file_hash": file_hash,
                        "score": tag_score_value,
                    }
                )

        if not report_by_tag:
            print(
                f"No new results found with levels {target_levels} in {self.base_dir}."
            )
            return None

        lines = ["# SUMMARY REPORT", ""]
        for tag in sorted(report_by_tag.keys()):
            items = report_by_tag[tag]
            lines.append(f"## {tag} ({len(items)} items)")
            for item in sorted(items, key=lambda x: x["score"], reverse=True):
                lines.append(f"  * **Title**: {item['title']}")
                lines.append(f"  * **Summary**: {item['summary']}")
                lines.append(f"  * **Source URL**: {item['source_url']}")
                lines.append(f"  * **File Hash**: {item['file_hash']}")
                lines.append("  ---")
            lines.append("")

        report_content = "\n".join(lines)
        print(report_content)

        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(report_content)
                print(f"\n[INFO] Report exported to: {os.path.abspath(output_file)}")
            except OSError as e:
                print(f"Error exporting report: {e}")

        return report_content

    # ── Iteration support ───────────────────────────────────────────────

    @property
    def file_hashes(self) -> list[str]:
        """Return all known file hashes."""
        return list(self._hashes.keys())

    # ── Internal persistence helpers ────────────────────────────────────

    def _load_json(self, filepath: str) -> dict:
        """Load JSON file, returning empty dict if missing or corrupt."""
        if not os.path.exists(filepath):
            return {}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except OSError, json.JSONDecodeError:
            return {}

    def _save_json(self, filepath: str) -> None:
        """Persist dict to JSON file."""
        if "hashes" in filepath:
            data = self._hashes
        elif "results" in filepath:
            data = self._results
        elif "pipeline_registry" in filepath:
            data = self._registries
        else:
            return

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
