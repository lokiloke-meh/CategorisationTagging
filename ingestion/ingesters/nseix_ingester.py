"""NSE India circular scraper.

Both importable class and runnable CLI:

    from ingestion.ingesters.nseix_ingester import NSEIXCircularScraper
    scraper = NSEIXCircularScraper(from_date="20261001", to_date="20261031")
    scraper.run()

CLI:

    python -m ingestion.ingesters.nseix_ingester --from-date 20261004 --to-date 20261005

"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

from ingestion.circular_scraper_base import CircularScraper


class NSEIXCircularScraper(CircularScraper, ABC):
    """Scrape NSE India circulars via JSON API and save as markdown.

    Additional features (NSEIX-specific):
        from_date / to_date: Explicit date range as YYYYMMDD.
    """

    storage_name = "nseix_circulars"

    def __init__(
        self,
        days_to_scrape: int = 30,
        output_dir: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> None:
        self.from_date = from_date
        self.to_date = to_date
        super().__init__(days_to_scrape=days_to_scrape, output_dir=output_dir)
        # NSEIX handles ZIP archives — create zip/ directory on demand
        self.zip_storage_directory = self.base_directory / "zip"
        self.zip_storage_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_parser() -> ArgumentParser:
        """Build the shared CLI parser with NSEIX-specific extensions."""
        parser = CircularScraper.build_parser("NSE India Circular Scraper")
        return NSEIXCircularScraper.extend_cli_parser(parser)

    @staticmethod
    def extend_cli_parser(parser: ArgumentParser) -> ArgumentParser:
        """Add NSEIX-specific CLI arguments on top of the shared parser."""
        parser.add_argument(
            "--from-date",
            "-f",
            type=str,
            default=None,
            help="Start date as YYYYMMDD (overrides --days if provided)",
        )
        parser.add_argument(
            "--to-date",
            "-t",
            type=str,
            default=None,
            help="End date as YYYYMMDD (required with --from-date)",
        )
        return parser

    def _resolve_date_range(self) -> tuple[str, str] | None:
        """Return (from_str, to_str) in DD-MM-YYYY for API, or None if invalid."""
        if self.from_date and self.to_date:
            # Parse YYYYMMDD → format as DD-MM-YYYY for NSEIX API
            try:
                start = datetime.strptime(self.from_date.strip(), "%Y%m%d").replace(
                    tzinfo=UTC
                )
                end = datetime.strptime(self.to_date.strip(), "%Y%m%d").replace(
                    tzinfo=UTC
                )
                return (
                    start.strftime("%d-%m-%Y"),
                    end.strftime("%d-%m-%Y"),
                )
            except ValueError:
                logger.error(
                    "Date format must be YYYYMMDD (got: %s, %s)",
                    self.from_date,
                    self.to_date,
                )
                return None

        today = datetime.now(UTC)
        start_date = CircularScraper._days_start_date(today, self.days_to_scrape)
        formatted_start = start_date.strftime("%d-%m-%Y")
        formatted_end = today.strftime("%d-%m-%Y")
        return (formatted_start, formatted_end)

    def _fetch_circular_records(self) -> list[dict[str, str]]:
        """Fetch all circulars for the configured date range via NSEIX API."""
        date_range = self._resolve_date_range()
        if date_range is None:
            return []

        from_str, to_str = date_range
        api_url = (
            f"https://www.nseix.com/api/circulars?fromdate={from_str}&todate={to_str}"
        )

        # Browser-like headers required — NSEIX blocks plain requests
        fetch_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.nseix.com/circulars/all-circulars",
        }

        logger.info(
            "Fetching circulars for date range %s to %s...",
            self.from_date or "auto",
            self.to_date or "auto",
        )

        # Retry with exponential backoff: 15s → 30s → 60s between attempts
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                logger.info("API request attempt %d/%d...", attempt + 1, max_retries)
                response = requests.get(api_url, headers=fetch_headers, timeout=60)
                response.raise_for_status()
                raw_records = response.json()
                break
            except requests.RequestException as exc:
                wait_time = (attempt + 1) * 15
                logger.warning(
                    "API request failed (attempt %d/%d): %s. Waiting %ds before retry...",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_time,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
            except ValueError as exc:
                logger.error("Invalid JSON response from API: %s", exc)
                return []

        if response is None or not response.ok:
            logger.error("Failed to fetch circulars after %d attempts.", max_retries)
            return []

        if not isinstance(raw_records, list):
            logger.error(
                "Expected array of records, got %s", type(raw_records).__name__
            )
            return []

        if not isinstance(raw_records, list):
            logger.error(
                "Expected array of records, got %s", type(raw_records).__name__
            )
            return []

        # Parse and classify each record
        collected_circular_records: list[dict[str, str]] = []
        seen_attachments: set[str] = set()

        for raw_record in raw_records:
            circular_no_raw = raw_record.get("CIRCULAR_NO", "")
            circular_no_cleaned = re.sub(r"\r", "", str(circular_no_raw)).strip()
            if not circular_no_cleaned:
                continue

            circ_date_display = raw_record.get("CIRCDATE", "")
            try:
                parsed_date = datetime.strptime(
                    str(circ_date_display), "%B %d, %Y"
                ).replace(tzinfo=UTC)
                circ_date_iso = parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                logger.warning(
                    "Could not parse date '%s' for circular: %s",
                    circ_date_display,
                    raw_record.get("CIRC_SUBJECT", ""),
                )
                circ_date_iso = str(circ_date_display)

            # Determine attachment type and URL
            file_link = raw_record.get("CIRC_FILELINK", "")
            attachment_url = raw_record.get("CIRC_ATTACHMENT", "")
            primary_url = file_link or attachment_url

            if not primary_url:
                logger.warning(
                    "No attachment found for circular: %s",
                    raw_record.get("CIRC_SUBJECT", ""),
                )
                continue

            # Dedup on CIRC_FILELINK (the canonical URL)
            dedup_key = file_link or urlparse(primary_url).geturl()
            if dedup_key in seen_attachments:
                logger.info(
                    "[DEDUPLICATED] Attachment %s already processed.", dedup_key
                )
                continue
            seen_attachments.add(dedup_key)

            # Classify: ZIP vs PDF
            lower_url = primary_url.lower()
            is_zip = (
                ".zip" in urlparse(primary_url).path or "attachmentType=1" in lower_url
            )  # NSEIX uses query param for type

            record_data: dict[str, str] = {
                "title": raw_record.get("CIRC_SUBJECT", "Untitled"),
                "circular_no": circular_no_cleaned,
                "date": circ_date_iso,
                "category": raw_record.get("CC_COMPANY_NAME", "NSEIX"),
                "source_url": primary_url,
            }

            if is_zip:
                record_data["is_zip"] = "true"
                record_data["zip_url"] = primary_url
            else:
                record_data["pdf_url"] = primary_url

            collected_circular_records.append(record_data)

        logger.info(
            "Collected %d unique circular records.", len(collected_circular_records)
        )
        return collected_circular_records

    def _download_zip_archive(self, zip_url: str, title: str = "") -> Path | None:
        """Download a ZIP file with ContentStore byte-level dedup.

        Returns local Path or None on failure. Applies 3s cooldown between downloads.
        Uses longer timeout (120s) and retry logic for large files.
        """
        time.sleep(3)

        # Browser-like headers required — NSEIX blocks plain requests
        fetch_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.nseix.com/circulars/all-circulars",
        }

        # Retry with exponential backoff: 15s → 30s → 60s between attempts
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                logger.info("ZIP download attempt %d/%d...", attempt + 1, max_retries)
                response = requests.get(zip_url, headers=fetch_headers, timeout=120)
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                wait_time = (attempt + 1) * 15
                logger.warning(
                    "ZIP download failed (attempt %d/%d): %s. Waiting %ds before retry...",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_time,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)

        if response is None or not response.ok:
            logger.warning(
                "Failed to fetch ZIP from %s after %d attempts.", zip_url, max_retries
            )
            return None

        content_hash = self.content_store.hash_bytes(response.content)
        already_saved_path = self.content_store.check_duplicate(content_hash)
        if already_saved_path:
            logger.info(
                "[DEDUPLICATED] ZIP identical to '%s'. Skipping download.",
                already_saved_path,
            )
            return self.base_directory / already_saved_path

        # Determine filename from Content-Disposition header, then URL path, then fallback
        content_disposition = response.headers.get("Content-Disposition", "")
        raw_filename = None

        if "filename=" in content_disposition:
            match_result = re.search(
                r"""filename\*?\s*=\s*["']?([^"';\s]+)["']?""", content_disposition
            )
            raw_filename = match_result.group(1) if match_result else None

        if not raw_filename or not raw_filename.strip():
            path_name = Path(urlparse(zip_url).path).name
            if path_name and "." in path_name:
                raw_filename = path_name
            elif response.headers.get("Content-Type") == "application/zip":
                raw_filename = "archive.zip"
            else:
                # Use circular number from record title as fallback (set by caller)
                raw_filename = (
                    f"{self.sanitize_filename(title)}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.zip"
                    if title
                    else f"archive_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.zip"
                )

        sanitized_name = self.sanitize_filename(raw_filename)
        unique_zip_name = self.get_unique_filename(
            self.zip_storage_directory, sanitized_name
        )
        zip_filepath = self.zip_storage_directory / unique_zip_name
        zip_filepath.write_bytes(response.content)

        relative_path = f"zip/{unique_zip_name}"
        self.content_store.mark_saved(
            file_hash=content_hash,
            relative_path=relative_path,
            source_url=zip_url,
            title=sanitized_name,
        )
        logger.info("Saved ZIP archive to %s", unique_zip_name)

        time.sleep(3)
        return zip_filepath

    def _download_pdf(
        self, pdf_document_url: str, referrer_page_url: str = "", title: str = ""
    ) -> str | None:
        """Download a PDF with ContentStore byte-level dedup.

        Returns relative storage path ``pdf/<filename>`` or None on failure.
        Uses browser-like headers — NSEIX blocks plain requests.
        """
        fetch_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": referrer_page_url
            or "https://www.nseix.com/circulars/all-circulars",
        }

        try:
            response = requests.get(pdf_document_url, headers=fetch_headers, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch PDF from %s: %s", pdf_document_url, exc)
            return None

        content_hash = self.content_store.hash_bytes(response.content)
        already_saved_path = self.content_store.check_duplicate(content_hash)
        if already_saved_path:
            logger.info("[DEDUPLICATED] Reusing %s", already_saved_path)
            return already_saved_path

        # Determine filename from URL path or Content-Disposition header
        content_disposition = response.headers.get("Content-Disposition", "")
        raw_filename = None

        if "filename=" in content_disposition:
            match_result = re.search(
                r"""filename\*?\s*=\s*["']?([^"';\s]+)["']?""", content_disposition
            )
            raw_filename = match_result.group(1) if match_result else None

        if not raw_filename or not raw_filename.strip():
            path_name = Path(urlparse(pdf_document_url).path).name
            if path_name and "." in path_name:
                raw_filename = path_name
            else:
                # Use circular number/title for meaningful filename
                raw_filename = (
                    f"{self.sanitize_filename(title)}.pdf" if title else "circular.pdf"
                )

        sanitized_name = self.sanitize_filename(raw_filename)
        unique_pdf_name = self.get_unique_filename(
            self.pdf_storage_directory, sanitized_name
        )
        pdf_filepath = self.pdf_storage_directory / unique_pdf_name
        pdf_filepath.write_bytes(response.content)

        relative_path = f"pdf/{unique_pdf_name}"
        self.content_store.mark_saved(
            file_hash=content_hash,
            relative_path=relative_path,
            source_url=pdf_document_url,
            title=sanitized_name,
        )
        logger.info("Saved PDF to %s", unique_pdf_name)

        time.sleep(3)
        return relative_path

    def _process_record(
        self, record: dict[str, Any]
    ) -> tuple[str | None, str, dict[str, Any]]:
        """Process NSEIX record — handles both ZIP and PDF archives."""
        is_zip = record.get("is_zip") == "true"

        try:
            if is_zip:
                zip_path = self._download_zip_archive(
                    record["zip_url"], title=record.get("title", "")
                )
                if not zip_path:
                    return None, "", {}

                aggregated_text = CircularScraper.extract_zip_contents(zip_path)
                pdf_text = (
                    aggregated_text
                    if aggregated_text
                    else "*[No extractable text found in ZIP]*"
                )
            else:
                path = self._download_pdf(
                    pdf_document_url=record["pdf_url"],
                    referrer_page_url="",
                    title=record.get("title", ""),
                )
                if not path:
                    return None, "", {}

                local_pdf_filepath = self.pdf_storage_directory / Path(path).name
                pdf_text = self.extract_pdf_text(local_pdf_filepath)

            extra_fields = {k: v for k, v in record.items() if k != "title"}
            body_text = self._generate_body(
                pdf_storage_path=path if not is_zip else "",
                pdf_text=pdf_text,
                **extra_fields,
            )
            return body_text, path if not is_zip else "", extra_fields
        except (requests.RequestException, OSError) as exc:
            error_msg = f"Error processing '{record.get('title')}': {exc}"
            logger.error(error_msg)
            self.collected_errors.append(error_msg)
            return None, "", {}

    def _extra_frontmatter_fields(self, **extra_fields) -> dict:  # type: ignore[override]
        """Return NSEIX-specific frontmatter keys."""
        is_zip = extra_fields.get("is_zip") == "true"
        result: dict[str, object] = {
            "Circular_Number": extra_fields.get("circular_no", ""),
            "Date": extra_fields.get("date", ""),
            "Category": extra_fields.get("category", "NSEIX"),
            "Source_URL": extra_fields.get("source_url", ""),
            "Document_Type": "ZIP Archive" if is_zip else "PDF",
        }
        if not is_zip and (path := extra_fields.get("pdf_storage_path")):
            result["Downloaded_PDF_Paths"] = [path]
        return result

    def _generate_body(
        self,
        pdf_storage_path: str,
        pdf_text: str,
        **extra_fields,
    ) -> str:
        """Generate NSEIX body content (raw PDF/ZIP extracted text)."""
        return pdf_text


if __name__ == "__main__":
    from ingestion.circular_scraper_base import main_from_parser

    main_from_parser(NSEIXCircularScraper, "NSE India Circular Scraper")
