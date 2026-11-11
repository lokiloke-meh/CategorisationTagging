"""Bybit Announcement scraper via V5 Public API.

Both importable class and runnable CLI::

    from ingestion.ingesters.bybit_announcement_ingester import BybitAnnouncementScraper
    scraper = BybitAnnouncementScraper(days_to_scrape=7)
    scraper.run()

CLI usage::

    uv run -m ingestion.ingesters.bybit_announcement_ingester --days 7
    uv run -m ingestion.ingesters.bybit_announcement_ingester --days 7 --category-type new_crypto
    uv run -m ingestion.ingesters.bybit_announcement_ingester --days 7 --output /custom/path

"""

from __future__ import annotations

import logging
import re
import time
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from curl_cffi import requests as cffi_requests

from ingestion.circular_scraper_base import (
    CircularScraper,
    main_from_parser,
)

logger = logging.getLogger(__name__)


# ── Category type reference (for user discoverability) ───
_CATEGORY_TYPE_NAMES: dict[str, str] = {
    "new_crypto": "New Cryptocurrency Listings",
    "delistings": "Delistings",
    "trading_rules_changes": "Trading Rules Changes",
    "maintenance": "Maintenance Updates",
    "system_upgrade": "System Upgrades",
}


class BybitAnnouncementScraper(CircularScraper):
    """Scrape Bybit announcements via the V5 Public API and save as markdown."""

    storage_name = "bybit_announcements"

    def __init__(
        self,
        days_to_scrape: int = 30,
        output_dir: str | None = None,
        category_type: str | None = None,
        locale: str = "en-US",
    ) -> None:
        self.category_type = category_type
        self.locale = locale
        super().__init__(days_to_scrape=days_to_scrape, output_dir=output_dir)

        # Bybit V5 Announcements API endpoint
        self.list_api_url = "https://api.bybit.com/v5/announcements/index"
        self.portal_base_url = "https://announcements.bybit.com"

        # curl_cffi session with Chrome impersonation for Cloudflare bypass
        self.http_session = cffi_requests.Session(impersonate="chrome120")
        self.http_session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": f"{self.locale},{self.locale.split('-')[0]};q=0.9",
                "Referer": f"{self.portal_base_url}/{self.locale}/",
                "Origin": self.portal_base_url,
            }
        )

    # ── CLI extension ───────────────────────────────────────────────────

    @staticmethod
    def extend_cli_parser(parser: ArgumentParser) -> ArgumentParser:
        """Add Bybit-specific CLI arguments on top of the shared parser."""
        available = ", ".join(sorted(_CATEGORY_TYPE_NAMES.keys()))
        parser.add_argument(
            "--category-type",
            "-c",
            type=str,
            default=None,
            help=(
                f"Announcement category filter. "
                f"Available: {available}. Leave empty for all categories."
            ),
        )
        parser.add_argument(
            "--locale",
            type=str,
            default="en-US",
            help="Language locale code (default: en-US)",
        )
        return parser

    def _download_pdf(
        self, pdf_document_url: str, referrer_page_url: str
    ) -> str | None:
        """Download a PDF from Bybit article attachments."""
        try:
            request_headers = {
                "Accept": "application/pdf,text/html,*/*",
                "Referer": referrer_page_url,
            }

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

            url_pathname = Path(pdf_document_url).name
            sanitized_pdf_filename = (
                self.sanitize_filename(url_pathname or "attachment.pdf") + ".pdf"
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

        except cffi_requests.RequestException as exc:
            self.collected_errors.append(
                f"Network error downloading PDF {pdf_document_url} - {exc}"
            )
            return None
        except OSError as exc:
            self.collected_errors.append(f"Failed to save PDF file: {exc}")
            return None

    # ── Abstract method implementations ────────────────────────────────

    def _fetch_circular_records(self) -> list[dict[str, str]]:
        """Fetch all announcements from Bybit V5 API within the configured date range.

        Paginates until an announcement falls outside the user's search window
        (announcements returned newest-first).
        """
        collected: list[dict[str, str]] = []
        cutoff_date = CircularScraper._days_start_date(
            datetime.now(UTC), self.days_to_scrape
        )
        seen_urls: set[str] = set()

        page_num = 1
        max_pages = 20  # Safety limit (Bybit has ~9000 total announcements)

        while page_num <= max_pages:
            params: dict[str, str | int] = {
                "locale": self.locale,
                "page": page_num,
                "limit": 20,
            }
            if self.category_type:
                params["type"] = self.category_type

            try:
                api_response = self.http_session.get(
                    self.list_api_url, params=params, timeout=30
                )
                if not api_response.ok:
                    self.collected_errors.append(
                        f"API returned HTTP {api_response.status_code} on page {page_num}"
                    )
                    break

                response_json = api_response.json()
                if response_json.get("retCode") != 0:
                    self.collected_errors.append(
                        f"Bybit API error (code={response_json.get('retCode')}): "
                        f"{response_json.get('retMsg')}"
                    )
                    break

                result_data = response_json.get("result", {})
                items = result_data.get("list", [])
                total_items = result_data.get("total", 0)

            except cffi_requests.RequestException as exc:
                self.collected_errors.append(f"Exception on page {page_num}: {exc}")
                break

            if not items:
                logger.info("No more records returned. Pagination complete.")
                break

            announcements_added_this_page = 0
            for item in items:
                title = item.get("title")
                url = item.get("url")
                date_timestamp_milliseconds = item.get("dateTimestamp")

                if not title or not url:
                    continue

                # Parse category (Bybit returns `type` as either a dict or string)
                type_object = item.get("type", {})
                if isinstance(type_object, dict):
                    category_key = type_object.get("key", "")
                    category_title = type_object.get("title", "")
                else:
                    category_key = str(type_object)
                    category_title = str(type_object)

                # Check chronological cutoff (millisecond epoch timestamp — may be string or int)
                publish_date = datetime.now(UTC)
                if date_timestamp_milliseconds is not None:
                    ts = (
                        int(date_timestamp_milliseconds)
                        if isinstance(date_timestamp_milliseconds, str)
                        else date_timestamp_milliseconds
                    )
                    publish_date = datetime.fromtimestamp(ts / 1000, tz=UTC)
                    if publish_date < cutoff_date:
                        logger.info(
                            "Skipping old item (%s < %s): %s",
                            publish_date.strftime("%Y-%m-%d"),
                            cutoff_date.strftime("%Y-%m-%d"),
                            title,
                        )
                        continue

                if url in seen_urls:
                    continue
                seen_urls.add(url)

                collected.append(
                    {
                        "title": title.strip(),
                        "source_url": url,
                        "category_key": category_key or "",
                        "category_title": category_title or "",
                        "publish_date": publish_date.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                announcements_added_this_page += 1

            logger.info(
                "Added %d announcement(s) from page %d.",
                announcements_added_this_page,
                page_num,
            )

            if total_items and len(collected) >= total_items:
                break

            page_num += 1
            time.sleep(0.3)  # Respect Bybit rate limit

        return collected

    def _process_record(
        self, record: dict[str, Any]
    ) -> tuple[str | None, str, dict[str, Any]]:
        """Process a Bybit announcement — fetch article body via HTML scraping."""
        source_url = record["source_url"]
        downloaded_pdfs: list[str] = []
        markdown_body = ""

        # Fetch article page HTML with retry logic
        for attempt in range(3):
            try:
                article_headers = {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                }
                article_response = self.http_session.get(
                    source_url, headers=article_headers, timeout=30
                )
                if not article_response.ok:
                    raise RuntimeError(f"HTTP {article_response.status_code}")

                from bs4 import BeautifulSoup
                from markdownify import markdownify as md

                detail_page_soup = BeautifulSoup(article_response.text, "html.parser")

                # Target Bybit announcement article body container
                main_content = (
                    detail_page_soup.find("article")
                    or detail_page_soup.find(
                        "div",
                        class_=re.compile(
                            r"article-detail|article-content|content-detail",
                            re.IGNORECASE,
                        ),
                    )
                    or detail_page_soup.find("main")
                    or detail_page_soup.body
                )

                if main_content:
                    article_body_html = str(main_content)

                    # Scan for embedded PDF attachments
                    pdf_links = self._find_pdf_links_in_html(
                        article_body_html, source_url
                    )
                    if pdf_links:
                        for pdf_url, link_title in pdf_links:
                            logger.info("  -> Found embedded PDF: %s", link_title)
                            saved = self._download_pdf(
                                pdf_document_url=pdf_url,
                                referrer_page_url=source_url,
                            )
                            if saved:
                                downloaded_pdfs.append(saved)

                    # Convert HTML to markdown
                    markdown_body = md(
                        str(main_content),
                        heading_style="ATX",
                        strip=["script", "style"],
                    )

                break  # Success — exit retry loop

            except (RuntimeError, OSError) as exc:
                wait_time = 15 * (2**attempt)
                logger.warning(
                    "Article fetch failed for %s (attempt %d/3): %s. Retrying in %ds...",
                    source_url,
                    attempt + 1,
                    exc,
                    wait_time,
                )
                if attempt < 2:
                    time.sleep(wait_time)
                else:
                    self.collected_errors.append(
                        f"Failed to fetch article {source_url}: {exc}"
                    )

        extra_fields = {
            "category_key": record.get("category_key", ""),
            "category_title": record.get("category_title", ""),
            "publish_date": record.get("publish_date", "N/A"),
            "source_url": source_url,
            "Document_Type": (
                "Announcement (Contains PDF)" if downloaded_pdfs else "Announcement"
            ),
        }

        if downloaded_pdfs:
            extra_fields["Downloaded_PDF_Paths"] = downloaded_pdfs

        body_text = self._generate_body(
            pdf_storage_path=downloaded_pdfs[0] if downloaded_pdfs else "",
            pdf_text=markdown_body,
            **extra_fields,
        )
        time.sleep(0.3)  # Cooldown between article fetches
        return (
            body_text,
            downloaded_pdfs[0] if downloaded_pdfs else "",
            extra_fields,
        )

    @staticmethod
    def _find_pdf_links_in_html(
        html_content: str, base_url: str
    ) -> list[tuple[str, str]]:
        """Extract PDF links from HTML content."""
        try:
            from bs4 import BeautifulSoup

            html_soup = BeautifulSoup(html_content, "html.parser")
            pdf_links: list[tuple[str, str]] = []

            for anchor_element in html_soup.find_all("a", href=True):
                href_value = anchor_element.get("href", "")
                if href_value.lower().endswith(".pdf"):
                    full_url = urljoin(base_url, href_value)
                    link_title = (anchor_element.text or "attachment").strip()
                    pdf_links.append((full_url, link_title or "attachment"))

            return pdf_links
        except Exception:  # noqa: BLE001
            return []

    def _extra_frontmatter_fields(self, **extra_fields) -> dict:  # type: ignore[override]
        """Return Bybit-specific frontmatter keys."""
        result: dict[str, object] = {
            "Category_Key": extra_fields.get("category_key", ""),
            "Category_Title": extra_fields.get("category_title", ""),
            "Publish_Date": extra_fields.get("publish_date", "N/A"),
            "Source_URL": extra_fields.get("source_url", ""),
            "Document_Type": extra_fields.get("Document_Type", "Announcement"),
        }
        if paths := extra_fields.get("Downloaded_PDF_Paths"):
            result["Downloaded_PDF_Paths"] = paths
        return result

    def _generate_body(
        self,
        pdf_storage_path: str,
        pdf_text: str,
        **extra_fields: Any,
    ) -> str:
        """Generate Bybit body content with optional PDF attachments."""
        source_url = extra_fields.get("source_url", "")

        # Build body: article content (pdf_text here = converted markdown from HTML)
        body_content = (
            pdf_text.strip() if pdf_text else "*No article body content available.*"
        )

        return f"# Announcement\n\n**Note:** This announcement contains online article content.\n\n{'- [View Local PDF](../' + pdf_storage_path + ')\n' if pdf_storage_path else ''}- [Original Link]({source_url})\n\n---\n## Article Content\n\n{body_content}\n"


if __name__ == "__main__":
    main_from_parser(BybitAnnouncementScraper, "Bybit Announcement Scraper")
