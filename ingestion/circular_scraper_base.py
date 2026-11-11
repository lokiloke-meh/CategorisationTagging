from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader
from pypdf.errors import PdfReadError as PdfReaderError

from ingestion.content_store import ContentStore


def _ensure_logging() -> None:
    """Configure root logging if no handlers are present (CLI usage)."""
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


logger = logging.getLogger(__name__)


class CircularScraper(ABC):
    """Abstract base for circular scrapers.

    Interface (the seam callers cross):
        days_to_scrape: How many past days to fetch data for.
        output_dir: Where markdown/PDFs/hashes are stored (default ``data_store/<name>/``).
        run(): Execute the full scrape pipeline.

    Each subclass overrides ``_fetch_circular_records()`` and
    ``_generate_markdown()`` to implement its own data source while keeping
    download, save, dedup, and error handling shared.
    """

    storage_name: str = ""
    timestamp_ignore_pattern: str = r"(?i)scraped_at\s*:\s*.+\n"

    def __init__(self, days_to_scrape: int = 30, output_dir: str | None = None) -> None:
        self.days_to_scrape = days_to_scrape

        if output_dir is None:
            base_directory_path = Path("data_store") / self.storage_name
        else:
            base_directory_path = Path(output_dir)

        self.base_directory = base_directory_path
        self.ingest_directory = self.base_directory / "ingest"
        self.pdf_storage_directory = self.base_directory / "pdf"

        for directory in (
            self.base_directory,
            self.ingest_directory,
            self.pdf_storage_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.content_store = ContentStore(base_dir=str(self.base_directory))
        self.collected_errors: list[str] = []

    # ── Static utilities (shared across scrapers) ───────────────────────

    @staticmethod
    def is_pdf_url(url: str) -> bool:
        """Check if a URL points to a PDF file (ignoring query parameters)."""
        if not url:
            return False
        path_part = url.split("?")[0]
        return Path(path_part).suffix.lower() == ".pdf"

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Replace all non-alphanumeric chars with underscores, collapse runs, cap at 120."""
        name_part, ext = os.path.splitext(filename)
        # Treat bare ".", empty, or non-extension-like as no extension
        # Valid extensions: only letters after dot (e.g., .pdf, .md), nothing else
        ext_body = ext.lstrip(".")
        if not ext or ext == "." or not ext_body.isalpha():
            ext = ""
            name_part = filename
        cleaned = re.sub(r"[^\w]", "_", name_part)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        base = cleaned[:120] or "untitled"
        return f"{base}{ext}" if ext else base

    @staticmethod
    def _days_start_date(today: datetime, days_to_scrape: int) -> datetime:
        """Return start date for an N-day scrape where N=1 means today only."""
        return today - timedelta(days=max(0, days_to_scrape - 1))

    @staticmethod
    def get_unique_filename(directory_path: Path, base_filename: str) -> str:
        """Auto-increment filename to avoid collisions."""
        name_part, extension = os.path.splitext(base_filename)
        counter_value = 1
        while (directory_path / f"{name_part}_{counter_value}{extension}").exists():
            counter_value += 1
        return (
            f"{name_part}_{counter_value}{extension}"
            if counter_value > 1
            else base_filename
        )

    @staticmethod
    def extract_pdf_text(filepath: Path) -> str:
        """Extract text from a PDF file. Returns placeholder if no text found."""
        try:
            reader = PdfReader(str(filepath))
            pages = (
                page.extract_text() for page in reader.pages if page.extract_text()
            )
            full_text_content = "\n\n".join(pages).strip()
            return (
                full_text_content
                if full_text_content
                else "*[No extractable text found — may be a scanned image]*"
            )
        except (PdfReaderError, OSError) as exc:
            return f"*[Error reading PDF file: {exc}]*"

    @staticmethod
    def extract_zip_contents(zip_path: Path) -> str | None:
        """Extract text from all files inside a ZIP archive.

        Returns aggregated markdown with per-file section headers, or None if no content extracted.
        Supported formats: .pdf (via pypdf), .txt, .csv (UTF-8).
        Unsupported formats are logged at INFO level and skipped.
        """
        import zipfile

        sections: list[str] = []
        temp_files: list[Path] = []

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            file_list = [
                entry for entry in zip_ref.namelist() if not entry.endswith("/")
            ]

            try:
                for archive_entry in file_list:
                    temp_filepath = (
                        zip_path.parent / f"_temp_{Path(archive_entry).name}"
                    )
                    temp_files.append(temp_filepath)

                    with (
                        zip_ref.open(archive_entry) as source_file,
                        open(temp_filepath, "wb") as dest_file,
                    ):
                        dest_file.write(source_file.read())

                    extracted_text = CircularScraper._extract_single_format(
                        temp_filepath
                    )
                    if extracted_text is not None:
                        sections.append(f"---\n## {archive_entry}\n\n{extracted_text}")
            except (OSError, zipfile.BadZipFile) as exc:
                logger.warning("Failed to extract from ZIP %s: %s", zip_path.name, exc)

        for temp_filepath in temp_files:
            try:
                temp_filepath.unlink(missing_ok=True)
            except OSError:
                pass

        return "\n\n".join(sections) if sections else None

    @staticmethod
    def _extract_single_format(filepath: Path) -> str | None:
        """Extract text from a single file based on its extension."""
        file_ext = filepath.suffix.lower()

        match file_ext:
            case ".pdf":
                return CircularScraper.extract_pdf_text(filepath)
            case ".txt" | ".csv":
                try:
                    text_content = filepath.read_text(encoding="utf-8")
                    return text_content.strip() if text_content.strip() else None
                except UnicodeDecodeError:
                    logger.info(
                        "[SKIPPED] %s (%s) — file format not supported yet",
                        filepath.name,
                        file_ext,
                    )
                    return None
            case _:
                logger.info(
                    "[SKIPPED] %s (%s) — file format not supported yet",
                    filepath.name,
                    file_ext,
                )
                return None

    # ── Abstract methods (each scraper implements its own data source) ──

    @abstractmethod
    def _fetch_circular_records(self) -> list[dict[str, str]]:
        """Fetch all circular records for the configured date range.

        Returns a deduplicated list of dicts with at least these keys:
            - title (str): Display title of the circular
            - pdf_url (str): Direct URL to the PDF document

        Other keys are optional and will be passed through to ``_generate_markdown()``.
        """

    def _core_frontmatter(
        self, title: str, pdf_storage_path: str, download_timestamp: str
    ) -> dict[str, Any]:
        """Return the shared frontmatter keys every markdown file needs."""
        return {
            "Title": title,
            "Scraped_At": download_timestamp,
        }

    def _extra_frontmatter_fields(self, **extra_fields: Any) -> dict[str, Any]:
        """Return subclass-specific frontmatter keys. Override in subclasses."""
        return {}

    def assemble_markdown(
        self,
        title: str,
        pdf_storage_path: str,
        download_timestamp: str,
        body_text: str,
        **extra_fields: Any,
    ) -> str:
        """Assemble complete markdown with YAML frontmatter and body text.

        The base class owns frontmatter construction (Title, Scraped_At, etc.).
        Subclasses provide extra fields via ``_extra_frontmatter_fields()``.
        """
        merged: dict[str, Any] = {
            **self._core_frontmatter(title, pdf_storage_path, download_timestamp),
            **self._extra_frontmatter_fields(**extra_fields),
        }
        # Omit falsy values (empty strings, empty lists) from frontmatter
        clean = {k: v for k, v in merged.items() if v}
        frontmatter = yaml.dump(clean, sort_keys=False)
        return f"---\n{frontmatter}---\n\n{body_text}"

    @abstractmethod
    def _generate_body(
        self,
        pdf_storage_path: str,
        pdf_text: str,
        **extra_fields: Any,
    ) -> str:
        """Generate the body text (after YAML frontmatter) for a circular record.

        Subclasses override this to format PDF content according to their source's conventions.
        The base class handles frontmatter, dedup hashing, and saving.
        """

    def _process_record(self, record: dict[str, Any]) -> tuple[str | None, str, dict[str, Any]]:
        """Process a single circular record into body text for markdown generation.

        Returns:
            (body_text_or_None, pdf_storage_path, extra_fields_dict)
            If body_text is None, the record is skipped (e.g., download failed).
        """
        relative_pdf_storage_path = self._download_pdf(
            pdf_document_url=record["pdf_url"],
            referrer_page_url="",
        )

        if not relative_pdf_storage_path:
            return None, "", {}

        local_pdf_filepath = (
            self.pdf_storage_directory / Path(relative_pdf_storage_path).name
        )
        pdf_text = self.extract_pdf_text(local_pdf_filepath)

        extra_fields = {k: v for k, v in record.items() if k != "title"}
        body_text = self._generate_body(
            pdf_storage_path=relative_pdf_storage_path,
            pdf_text=pdf_text,
            **extra_fields,
        )
        return body_text, relative_pdf_storage_path, extra_fields

    def _report_errors(self) -> None:
        """Log collected errors or success message."""
        if self.collected_errors:
            logger.info(
                "\n%s\nWORKFLOW COMPLETED WITH %d ERROR(S)\n%s",
                "=" * 50,
                len(self.collected_errors),
                "=" * 50,
            )
            for i, error_message in enumerate(self.collected_errors, 1):
                logger.info("  %d. %s", i, error_message)
        else:
            logger.info("\nWorkflow completed successfully with 0 errors.")

    # ── Shared pipeline logic (save, dedup) ─────────────────────────────

    def _save_markdown(
        self,
        title: str,
        source_url: str,
        content_hash: str,
        markdown_content: str,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """Save markdown if unique (deduplication via ContentStore)."""
        already_saved_path = self.content_store.check_duplicate(content_hash)
        if already_saved_path:
            logger.info(
                "[DEDUPLICATED] Identical to '%s'. Skipping MD save.",
                already_saved_path,
            )
            return

        sanitized_filename = self.sanitize_filename(title) + ".md"
        unique_markdown_filename = self.get_unique_filename(
            self.ingest_directory, sanitized_filename
        )
        markdown_filepath = self.ingest_directory / unique_markdown_filename
        markdown_filepath.write_text(markdown_content, encoding="utf-8")

        relative_storage_path = f"ingest/{unique_markdown_filename}"
        self.content_store.mark_saved(
            file_hash=content_hash,
            relative_path=relative_storage_path,
            source_url=source_url,
            title=title,
        )
        logger.info("Saved markdown to %s", unique_markdown_filename)

    # ── Public entry point (the shared interface) ───────────────────────

    def run(self) -> None:
        """Execute the full scrape pipeline."""
        _ensure_logging()
        logger.info(
            "Starting %s scraper for the past %d days...",
            type(self).__name__,
            self.days_to_scrape,
        )

        records = self._fetch_circular_records()
        logger.info("\nTotal unique records gathered: %d\n", len(records))

        if not records:
            logger.info("No records found for the specified date range.")
            return

        for idx, record in enumerate(records):
            title = record.get("title", "Untitled")
            logger.info("[%d/%d] Processing: %s", idx + 1, len(records), title)

            body_text, pdf_storage_path, extra_fields = self._process_record(record)
            if body_text is None:
                continue

            generated_markdown = self.assemble_markdown(
                title=title,
                pdf_storage_path=pdf_storage_path or "",
                download_timestamp=datetime.now(UTC).isoformat(),
                body_text=body_text,
                **extra_fields,
            )

            content_hash = self.content_store.hash_text(
                generated_markdown, ignore_pattern=self.timestamp_ignore_pattern
            )
            source_url = extra_fields.get("source_url", "") or extra_fields.get("pdf_url", "")
            self._save_markdown(
                title=title,
                source_url=source_url,
                content_hash=content_hash,
                markdown_content=generated_markdown,
                extra_fields=extra_fields,
            )

        self._report_errors()

    @staticmethod
    def build_parser(description: str) -> ArgumentParser:
        """Build a shared CLI parser for all scrapers."""
        parser = ArgumentParser(description=description)
        parser.add_argument(
            "--days",
            "-d",
            type=int,
            default=1,
            help="Number of past days to scrape (default: 1)",
        )
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            default=None,
            help="Output directory (default: data_store/<scraper_name>/)",
        )
        return parser


def main_from_parser(scraper_class: type[CircularScraper], description: str) -> None:
    """Run a scraper from CLI args built by ``build_parser()``."""
    parser = CircularScraper.build_parser(description)

    # Allow subclass to add its own arguments before parsing
    extend_fn = getattr(scraper_class, "extend_cli_parser", None)
    if callable(extend_fn):
        parser = extend_fn(parser)

    args = parser.parse_args()
    kwargs: dict[str, object] = {"days_to_scrape": args.days}
    if hasattr(args, "output") and args.output is not None:
        kwargs["output_dir"] = args.output
    if hasattr(args, "from_date") and args.from_date is not None:
        kwargs["from_date"] = args.from_date
    if hasattr(args, "to_date") and args.to_date is not None:
        kwargs["to_date"] = args.to_date

    scraper_instance = scraper_class(**kwargs)  # type: ignore[arg-type]
    scraper_instance.run()
