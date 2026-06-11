import os
import json
import argparse
from collections import defaultdict
from datetime import datetime

class ReportTracker:
    """
    Tracks which file hashes and tags have already been included in reports.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.reported_items = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def is_reported(self, file_hash: str, tag: str) -> bool:
        """Checks if a specific tag for a file hash has been reported."""
        return file_hash in self.reported_items and tag in self.reported_items[file_hash]

    def add_batch(self, new_items: dict):
        """
        Updates the tracking data with a dictionary of {hash: [tags]}.
        """
        for fh, tlist in new_items.items():
            if fh not in self.reported_items:
                self.reported_items[fh] = []
            # Only add tags not already present
            for tag in tlist:
                if tag not in self.reported_items[fh]:
                    self.reported_items[fh].append(tag)

    def save(self):
        """Persists the tracking data to disk."""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.reported_items, f, indent=4)

def generate_report(base_dir: str, target_levels: list[str] = None, output_file: str = None, filter_tags: list = None):
    """
    Reads processing_results.json and file_hashes.json to produce a report
    of notices grouped by tags that exceed a specific score threshold.
    """
    if target_levels is None:
        target_levels = ["High"]

    if output_file is None:
        output_file = os.path.join(base_dir, f"report_{datetime.now().strftime('%Y-%m-%d')}.md")

    results_path = os.path.join(base_dir, "processing_results.json")
    hashes_path = os.path.join(base_dir, "file_hashes.json")
    tracking_path = os.path.join(base_dir, "reported_items.json")

    if not os.path.exists(results_path):
        print(f"Error: Result file not found at {results_path}")
        return
    if not os.path.exists(hashes_path):
        print(f"Error: Hash file not found at {hashes_path}")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    with open(hashes_path, 'r', encoding='utf-8') as f:
        hashes = json.load(f)

    tracker = ReportTracker(tracking_path)

    # Dictionary to group by tag: tag_name -> list of notice details
    report_by_tag = defaultdict(list)
    newly_reported = defaultdict(list)

    for file_hash, entry in results.items():
        # 'result' contains the serialized ChainResult (output_data and metadata)
        result_wrapper = entry.get("result", {})
        output_data = result_wrapper.get("output_data")
        
        if not output_data:
            continue
        
        tags = output_data.get("tags", {})
        summary = output_data.get("summary", "N/A")
        effective_date = output_data.get("effectivedate", "NONE")
        
        # Retrieve ingestion metadata
        hash_entry = hashes.get(file_hash, {})
        # Handle case where hash_entry might be a dict or a simple path string
        source_url = hash_entry.get("source_url", "No URL available") if isinstance(hash_entry, dict) else "No URL available"
        file_path = hash_entry.get("path", hash_entry) if isinstance(hash_entry, dict) else hash_entry

        # Determine a title from metadata or fallback to the filename
        title = hash_entry.get("title") if isinstance(hash_entry, dict) else None
        if not title:
            title = os.path.splitext(os.path.basename(file_path))[0]

        for tag, score in tags.items():
            # Apply tag filter if provided
            if filter_tags and tag not in filter_tags:
                continue
                
            if score in target_levels:
                # Skip if already reported for this tag
                if tracker.is_reported(file_hash, tag):
                    continue

                report_by_tag[tag].append({
                    "title": title,
                    "summary": summary,
                    "effective_date": effective_date,
                    "source_url": source_url,
                    "file_path": file_path,
                    "score": score
                })
                newly_reported[file_hash].append(tag)

    if not report_by_tag:
        print(f"No new results found with levels {target_levels} in {base_dir}.")
        return

    # Prepare the report content
    report_lines = []
    report_lines.append("# SUMMARY REPORT")
    report_lines.append("")

    # Sort tags alphabetically
    for tag in sorted(report_by_tag.keys()):
        items = report_by_tag[tag]
        report_lines.append(f"## {tag} ({len(items)} items)")
        
        # Sort items within a tag by score (highest first)
        for item in sorted(items, key=lambda x: x['score'], reverse=True):
            report_lines.append(f"  * **Title**: {item['title']}")
            report_lines.append(f"  * **Summary**: {item['summary']}")
            report_lines.append(f"  * **Source URL**: {item['source_url']}")
            report_lines.append("  ---")
        report_lines.append("")

    report_content = "\n".join(report_lines)

    # Output to console
    print(report_content)

    # Export to file if requested
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"\n[INFO] Report successfully exported to: {os.path.abspath(output_file)}")
            
            # Update tracking only upon successful file export
            tracker.add_batch(newly_reported)
            tracker.save()
        except Exception as e:
            print(f"Error exporting report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a report from processed tagging results.")
    parser.add_argument("base_dir", help="Directory containing processing_results.json and file_hashes.json")
    parser.add_argument("--levels", default="High", help="Comma-separated list of categorical levels for inclusion (default: High)")
    parser.add_argument("--output", help="Optional path to export the report as a Markdown file")
    parser.add_argument("--tags", help="Optional comma-separated list of tags to filter by")
    
    args = parser.parse_args()

    # Parse comma-separated tags into a list
    filter_tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
    # Parse comma-separated levels into a list
    target_levels = [lv.strip() for lv in args.levels.split(",")] if args.levels else ["High"]

    generate_report(args.base_dir, target_levels, args.output, filter_tags)