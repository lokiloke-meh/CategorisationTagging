"""HKEX Participant & Members Circular scraper.

Uses HTML scraping to fetch circulars for a date range, paginating through each day's
page until all records are collected. Downloads PDFs and generates markdown files.

Both importable class and runnable CLI:

    from ingestion.ingesters.hkex_ingester import HKEXCircularScraper
    scraper = HKEXCircularScraper(days_to_scrape=7)
    scraper.run()

CLI usage::

    uv run -m ingestion.ingesters.hkex_ingester --days 30

"""

from __future__ import annotations

import logging
import os
import re
import time
from argparse import ArgumentParser
from datetime import UTC, datetime, timedelta
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

from ingestion.circular_scraper_base import CircularScraper


class HKEXCircularScraper(CircularScraper):
    """Scrape HKEX participant circulars via HTML scraping and save as markdown + PDF."""

    storage_name = "hkex_circulars"

    def __init__(
        self,
        days_to_scrape: int = 30,
        base_url: str = "https://www.hkex.com.hk",
        output_dir: str | None = None,
    ) -> None:
        self.base_url = base_url
        super().__init__(days_to_scrape=days_to_scrape, output_dir=output_dir)

        # WAF-evasion HTTP session (browser-like headers for Akamai)
        self.http_session = requests.Session()
        self.http_session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
            }
        )

    def _fetch_circular_records(self) -> list[dict[str, str]]:
        """Fetch all circulars for the configured date range via HTML scraping."""
        collected_circular_records: list[dict[str, str]] = []
        seen_pdf_urls: set[str] = set()
        today = datetime.now(UTC)

        for i in range(self.days_to_scrape):
            target_date = today - timedelta(days=i)
            formatted_date = target_date.strftime("%Y-%m-%d")
            daily_search_url = (
                f"{self.base_url}/Services/Circulars-and-Notices/"
                f"Participant-and-Members-Circulars?sc_lang=en&DateFrom={formatted_date}&DateTo={formatted_date}"
            )

            logger.info("[%s] Fetching data...", formatted_date)

            try:
                html_response = self.http_session.get(daily_search_url, timeout=30)
                html_response.raise_for_status()
            except requests.RequestException as exc:
                self.collected_errors.append(
                    f"Failed to fetch URL for {formatted_date}: {exc}"
                )
                continue

            parsed_html = BeautifulSoup(html_response.text, "html.parser")
            all_anchor_tags = parsed_html.find_all("a", href=True)

            circulars_found_today = 0
            for anchor in all_anchor_tags:
                raw_href = anchor.get("href")

                if (
                    not CircularScraper.is_pdf_url(raw_href)
                    and "/-/media/" not in raw_href
                ):
                    continue

                for label_span in anchor.find_all(
                    "span", class_=re.compile(r"pdf", re.IGNORECASE)
                ):
                    label_span.decompose()

                circular_title = anchor.get_text(separator=" ", strip=True)
                if not circular_title:
                    continue

                resolved_url = urljoin(self.base_url, raw_href)

                if resolved_url in seen_pdf_urls:
                    continue
                seen_pdf_urls.add(resolved_url)

                collected_circular_records.append(
                    {
                        "title": circular_title,
                        "pdf_url": resolved_url,
                        "referer": daily_search_url,
                    }
                )
                circulars_found_today += 1

            if circulars_found_today > 0:
                logger.info("  -> Found %d Circular(s).", circulars_found_today)

            time.sleep(1.5)

        return collected_circular_records

    def _download_pdf(
        self, pdf_document_url: str, referrer_page_url: str
    ) -> str | None:
        """Download a PDF document using requests.Session with WAF-evasion headers."""
        try:
            request_headers = dict(self.http_session.headers)
            request_headers["Referer"] = referrer_page_url

            pdf_response = self.http_session.get(
                pdf_document_url, headers=request_headers, timeout=30
            )
            if not pdf_response.ok:
                self.collected_errors.append(
                    f"HTTP {pdf_response.status_code} on PDF: {pdf_document_url}"
                )
                return None

            pdf_file_bytes = pdf_response.content
            pdf_content_hash = self.content_store.hash_bytes(pdf_file_bytes)

            already_stored_path = self.content_store.check_duplicate(pdf_content_hash)
            if already_stored_path:
                logger.info("[DEDUPLICATED] Reusing %s", already_stored_path)
                return already_stored_path

            url_pathname = urlparse(pdf_document_url).path
            sanitized_pdf_filename = self.sanitize_filename(
                os.path.basename(url_pathname) or "circular.pdf"
            )
            unique_pdf_filename = self.get_unique_filename(
                self.pdf_storage_directory, sanitized_pdf_filename
            )
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

        except requests.RequestException as exc:
            self.collected_errors.append(
                f"Network error downloading PDF {pdf_document_url} - {exc}"
            )
            return None
        except OSError as exc:
            self.collected_errors.append(f"Failed to save PDF file: {exc}")
            return None

    @staticmethod
    def build_parser() -> ArgumentParser:
        """Build the shared CLI parser."""
        return CircularScraper.build_parser("HKEX Circular Scraper")

    def _extra_frontmatter_fields(self, **extra_fields) -> dict:  # type: ignore[override]
        """Return HKEX-specific frontmatter keys."""
        result = {
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
        """Generate HKEX-specific body content."""
        source_url = extra_fields.get("pdf_url", "")
        return (
            f"# Circular\n\n**Note:** This circular is a direct PDF document.\n\n"
            f"- [View Local PDF](../{pdf_storage_path})\n"
            f"- [Original Link]({source_url})\n\n"
            f"---\n## PDF Content\n\n{pdf_text}\n"
        )


if __name__ == "__main__":
    from ingestion.circular_scraper_base import main_from_parser

    main_from_parser(HKEXCircularScraper, "HKEX Circular Scraper")
