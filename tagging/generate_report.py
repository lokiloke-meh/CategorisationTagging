import argparse
import os

from config import cfg
from ingestion.content_store import ContentStore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a report from processed tagging results."
    )
    parser.add_argument(
        "base_dir",
        help="Directory containing processing_results.json and file_hashes.json",
    )
    parser.add_argument(
        "--levels",
        default=cfg.default_target_levels[0],
        help=f"Comma-separated list of categorical levels for inclusion (default: {','.join(cfg.default_target_levels)})",
    )
    parser.add_argument(
        "--output", help="Optional path to export the report as a Markdown file"
    )
    parser.add_argument(
        "--tags", help="Optional comma-separated list of tags to filter by"
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only include entries processed on or after this date (YYYY-MM-DD). Default: all entries.",
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        help="Limit report to a specific pipeline ID. Default: latest entry per hash.",
    )

    args = parser.parse_args()

    # Parse comma-separated tags into a list
    filter_tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
    # Parse comma-separated levels into a list
    target_levels = (
        [lv.strip() for lv in args.levels.split(",")]
        if args.levels
        else cfg.default_target_levels
    )

    content_store = ContentStore(base_dir=args.base_dir)

    # Compute output file path (only used if there are results)
    if args.output:
        output_file = args.output
    else:
        latest = content_store.latest_processed_date()
        effective_pipeline_id = args.pipeline or content_store.get_latest_pipeline_id() or ""
        pipeline_segment = f"_{effective_pipeline_id}"
        tag_segment = f"_{'-'.join(filter_tags)}" if filter_tags else "_all"
        date_segment = (
            f"_{args.since.replace('-', '')}_to_{latest}"
            if args.since
            else f"_to_{latest}"
        )
        output_file = os.path.join(
            args.base_dir, f"report{pipeline_segment}{tag_segment}{date_segment}.md"
        )
        report_pipeline_id: str | None = effective_pipeline_id if effective_pipeline_id else None

    report_content = content_store.generate_report(
        target_levels=target_levels,
        filter_tags=filter_tags,
        output_file=output_file,
        since_date=args.since,
        pipeline_id=report_pipeline_id,
    )
