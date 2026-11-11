"""MCX (Multi Commodity Exchange) circular scraper.

Uses the MCX JSON API to fetch circulars for a date range, paginating through all
pages until every record is collected. Downloads PDFs and generates markdown files.

Both importable class and runnable CLI:

    from ingestion.ingesters.mcx_ingester import MCXCircularScraper
    scraper = MCXCircularScraper(days_to_scrape=7)
    scraper.run()

    # or with explicit date range (additional feature):
    scraper = MCXCircularScraper(from_date="20261001", to_date="20261031")
    scraper.run()

"""

from __future__ import annotations

import logging
import time
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml
from curl_cffi import requests as cffi_requests
from curl_cffi.requests.exceptions import RequestException

logger = logging.getLogger(__name__)

from ingestion.circular_scraper_base import CircularScraper


class MCXCircularScraper(CircularScraper):
    """Scrape MCX circulars via JSON API and save as markdown + PDF.

    Additional features (MCX-specific):
        from_date / to_date: Optional explicit date range (YYYYMMDD) instead
            of computing from days_to_scrape. Only one mode can be active at a time.
    """

    storage_name = "mcx_circulars"

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

        # curl_cffi with Chrome impersonation to bypass Akamai WAF
        self.http_session = cffi_requests.Session(impersonate="chrome120")

    @staticmethod
    def build_parser() -> ArgumentParser:
        """Build the shared CLI parser with MCX-specific extensions."""
        parser = CircularScraper.build_parser("MCX Circular Scraper")
        return MCXCircularScraper.extend_cli_parser(parser)

    @staticmethod
    def extend_cli_parser(parser: ArgumentParser) -> ArgumentParser:
        """Add MCX-specific CLI arguments on top of the shared parser."""
        parser.add_argument(
            "--from-date", "-f", type=str, default=None,
            help="Start date as YYYYMMDD (overrides --days if provided)",
        )
        parser.add_argument(
            "--to-date", "-t", type=str, default=None,
            help="End date as YYYYMMDD (required with --from-date)",
        )
        return parser

    def _resolve_date_range(self) -> tuple[str | None, str | None]:
        """Return (from_date_str, to_date_str) in DD/MM/YYYY for API."""
        if self.from_date and self.to_date:
            # Parse YYYYMMDD → format as DD/MM/YYYY for MCX API
            try:
                start = datetime.strptime(self.from_date.strip(), "%Y%m%d").replace(
                    tzinfo=UTC
                )
                end = datetime.strptime(self.to_date.strip(), "%Y%m%d").replace(
                    tzinfo=UTC
                )
                return (
                    start.strftime("%d/%m/%Y"),
                    end.strftime("%d/%m/%Y"),
                )
            except ValueError as exc:
                logger.error("Invalid date format: %s. Expected YYYYMMDD.", exc)
                return None, None

        today = datetime.now(UTC)
        start_date = CircularScraper._days_start_date(today, self.days_to_scrape)
        formatted_start = start_date.strftime("%d/%m/%Y")
        formatted_end = today.strftime("%d/%m/%Y")
        return formatted_start, formatted_end

    def _fetch_circular_records(self) -> list[dict[str, str]]:
        """Fetch all circulars for the configured date range via paginated API."""
        from_date_str, to_date_str = self._resolve_date_range()

        if from_date_str is None or to_date_str is None:
            logger.info("Both --from-date and --to-date are required when not using --days.")
            return []

        base_api_endpoint = "https://www.mcxindia.com/circulars/all-circulars/GetFilteredAnnouncements"
        collected_circular_records: list[dict[str, str]] = []
        seen_pdf_urls: set[str] = set()

        page_number = 1
        total_record_count: int | None = None

        if not self._establish_session("https://www.mcxindia.com/circulars/all-circulars"):
            return []

        while True:
            api_request_url = (
                f"{base_api_endpoint}"
                f"?CircularTitle=&CircularsCategory=&CircularNo="
                f"&fromdate={from_date_str}&todate={to_date_str}&page={page_number}"
            )

            logger.info("Fetching page %d for date range %s to %s...", page_number, from_date_str, to_date_str)

            try:
                json_response = self.http_session.get(
                    api_request_url, impersonate="chrome120", timeout=60
                )
                if not json_response.ok:
                    wait_seconds = (page_number % 3 + 1) * 5
                    logger.warning(
                        "API returned HTTP %d for page %d. Retrying in %ds...",
                        json_response.status_code, page_number, wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue

                api_json_data = json_response.json()

                if total_record_count is None:
                    total_record_count = (
                        api_json_data.get("TotalRecordCount", 0) or len(api_json_data.get("AllResult", []))
                    )
                    logger.info("Total records to fetch: %d", total_record_count)

                response_records = api_json_data.get("AllResult", [])
                fetched_this_page = 0

                for record_item in response_records:
                    pdf_document_url = record_item.get("CircularFile", "")
                    if not CircularScraper.is_pdf_url(pdf_document_url):
                        continue
                    if pdf_document_url in seen_pdf_urls:
                        continue

                    seen_pdf_urls.add(pdf_document_url)
                    collected_circular_records.append({
                        "title": record_item.get("DisplayTitle", ""),
                        "circular_number": record_item.get("CircularNo", ""),
                        "display_date": record_item.get("DisplayDate", ""),
                        "category": record_item.get("CircularsCategory", ""),
                        "pdf_url": pdf_document_url,
                    })
                    fetched_this_page += 1

                logger.info(
                    "  -> Fetched %d circular(s) this page (%d total so far).",
                    fetched_this_page, len(collected_circular_records),
                )

                if fetched_this_page == 0 or len(collected_circular_records) >= total_record_count:
                    break

                page_number += 1
                time.sleep(2.0)

            except RequestException as exc:
                self.collected_errors.append(f"API request error on page {page_number}: {exc}")
                break

        return collected_circular_records

    def _establish_session(self, main_page_url: str) -> bool:
        """Fetch the main circulars page to establish cookies/session with Akamai."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                homepage_response = self.http_session.get(
                    main_page_url, impersonate="chrome120", timeout=60
                )
                if homepage_response.ok:
                    logger.info("Session cookies established with MCX website.")
                    return True

                wait_seconds = (attempt + 1) * 5
                logger.warning(
                    "Main page returned HTTP %d. Retrying in %ds (attempt %d/%d)...",
                    homepage_response.status_code, wait_seconds, attempt + 1, max_retries,
                )
                time.sleep(wait_seconds)

            except RequestException as exc:
                wait_seconds = (attempt + 1) * 5
                logger.warning("Network error fetching main page: %s. Retrying in %ds...", exc, wait_seconds)
                time.sleep(wait_seconds)

        self.collected_errors.append("Failed to establish session after all retries.")
        return False

    def _download_pdf(self, pdf_document_url: str, referrer_page_url: str) -> str | None:
        """Download a PDF document using curl_cffi with Chrome impersonation."""
        try:
            pdf_response = self.http_session.get(
                pdf_document_url, impersonate="chrome120",
                headers={"Referer": referrer_page_url}, timeout=60,
            )
            if not pdf_response.ok:
                self.collected_errors.append(f"HTTP {pdf_response.status_code} on PDF: {pdf_document_url}")
                return None

            pdf_file_bytes = pdf_response.content
            pdf_content_hash = self.content_store.hash_bytes(pdf_file_bytes)

            already_stored_path = self.content_store.check_duplicate(pdf_content_hash)
            if already_stored_path:
                logger.info("[DEDUPLICATED] Reusing %s", already_stored_path)
                return already_stored_path

            url_pathname = urlparse(pdf_document_url).path
            sanitized_pdf_filename = self.sanitize_filename(
                Path(url_pathname).name or "circular.pdf"
            )
            unique_pdf_filename = self.get_unique_filename(self.pdf_storage_directory, sanitized_pdf_filename)
            local_pdf_filepath = self.pdf_storage_directory / unique_pdf_filename
            local_pdf_filepath.write_bytes(pdf_file_bytes)

            relative_storage_path = f"pdf/{unique_pdf_filename}"
            self.content_store.mark_saved(
                file_hash=pdf_content_hash,
                relative_path=relative_storage_path,
                source_url=pdf_document_url,
                title=sanitized_pdf_filename,
            )
            return relative_storage_path

        except RequestException as exc:
            self.collected_errors.append(f"Network error downloading PDF {pdf_document_url} - {exc}")
            return None
        except OSError as exc:
            self.collected_errors.append(f"Failed to save PDF file: {exc}")
            return None

    def _extra_frontmatter_fields(self, **extra_fields) -> dict:  # type: ignore[override]
        """Return MCX-specific frontmatter keys."""
        result = {
            "Circular_Number": extra_fields.get("circular_number", ""),
            "Circular_Date": extra_fields.get("display_date", ""),
            "Circulars_Category": extra_fields.get("category", ""),
            "Source_URL": extra_fields.get("pdf_url", ""),
            "Document_Type": "PDF Direct Link",
        }
        if path := extra_fields.get("pdf_storage_path"):
            result["Downloaded_PDF_Paths"] = [path]
        return result

    def _generate_body(
        self,
        pdf_storage_path: str,
        pdf_text: str,
        **extra_fields,
    ) -> str:
        """Generate MCX-specific body content."""
        source_url = extra_fields.get("pdf_url", "")
        return (
            f"# Circular\n\n**Note:** Direct PDF document.\n\n"
            f"- [View Local PDF](../{pdf_storage_path})\n"
            f"- [Original Link]({source_url})\n\n"
            f"---\n## PDF Content\n\n{pdf_text}\n"
        )


if __name__ == "__main__":
    from ingestion.circular_scraper_base import main_from_parser

    main_from_parser(MCXCircularScraper, "MCX Circular Scraper")
