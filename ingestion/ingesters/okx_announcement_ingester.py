"""OKX Announcement scraper via V5 Support API.

Both importable class and runnable CLI::

    from ingestion.ingesters.okx_ingester import OKXAnnouncementScraper
    scraper = OKXAnnouncementScraper(days_to_scrape=7)
    scraper.run()

CLI usage::

    uv run -m ingestion.ingesters.okx_ingester --days 7
    uv run -m ingestion.ingesters.okx_ingester --days 7 --announcement-type api
    uv run -m ingestion.ingesters.okx_ingester --days 7 --output /custom/path

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


# ── Announcement type reference (for user discoverability) ───
_ANNOUNCEMENT_TYPE_NAMES: dict[str, str] = {
    "new_listings": "New Listings",
    "delisting": "Delisting",
    "maintenance": "Maintenance",
    "system_upgrade": "System Upgrade",
    "network_maintenance": "Network Maintenance",
    "withdrawal_suspension": "Withdrawal Suspension",
    "trading_rules": "Trading Rules",
    "api": "API Updates",
}


class OKXAnnouncementScraper(CircularScraper):
    """Scrape OKX announcements via the V5 Support API and save as markdown."""

    storage_name = "okx_announcements"

    def __init__(
        self,
        days_to_scrape: int = 30,
        output_dir: str | None = None,
        announcement_type: str | None = None,
    ) -> None:
        self.announcement_type = announcement_type
        super().__init__(days_to_scrape=days_to_scrape, output_dir=output_dir)

        # OKX V5 Support API endpoint
        self.list_api_url = "https://www.okx.com/api/v5/support/announcements"

        # curl_cffi session with Chrome impersonation for Cloudflare bypass
        self.http_session = cffi_requests.Session(impersonate="chrome120")
        self.http_session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.okx.com/en-sg/help/category/announcements",
                "Origin": "https://www.okx.com",
            }
        )

    # ── CLI extension ───────────────────────────────────────────────────

    @staticmethod
    def extend_cli_parser(parser: ArgumentParser) -> ArgumentParser:
        """Add OKX-specific CLI arguments on top of the shared parser."""
        available = ", ".join(sorted(_ANNOUNCEMENT_TYPE_NAMES.keys()))
        parser.add_argument(
            "--announcement-type",
            type=str,
            help=(
                f"Announcement category filter. "
                f"Available: {available}. Leave empty for all categories."
            ),
        )
        return parser

    # ── Abstract method implementations ────────────────────────────────

    def _fetch_circular_records(self) -> list[dict[str, str]]:
        """Fetch all announcements from OKX V5 API within the configured date range.

        Paginates until an announcement falls outside the user's search window
        (announcements returned newest-first).
        """
        collected: list[dict[str, str]] = []
        cutoff_date = CircularScraper._days_start_date(datetime.now(UTC), self.days_to_scrape)
        seen_urls: set[str] = set()

        page_num = 1
        max_pages = 50  # Safety limit
        reached_cutoff = False

        while not reached_cutoff and page_num <= max_pages:
            params: dict[str, str] = {"page": str(page_num)}
            if self.announcement_type:
                params["annType"] = self.announcement_type

            try:
                resp = self.http_session.get(
                    self.list_api_url, params=params, timeout=30
                )
                if not resp.ok:
                    self.collected_errors.append(
                        f"API returned HTTP {resp.status_code} on page {page_num}"
                    )
                    break

                json_data = resp.json()
                # OKX returns data as [{details: [...]}], flatten to announcements
                raw_items = json_data.get("data", [])
                items = []
                for block in raw_items:
                    items.extend(block.get("details", []))
                total_pages = int(json_data.get("totalPage", 1))

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
                published_timestamp_milliseconds = item.get("pTime")
                category = item.get("annType", "")

                if not title or not url:
                    continue

                # Check chronological cutoff (millisecond epoch timestamp — may be string or int)
                publish_date = datetime.now(UTC)
                if published_timestamp_milliseconds is not None:
                    ts = int(published_timestamp_milliseconds) if isinstance(published_timestamp_milliseconds, str) else published_timestamp_milliseconds
                    publish_date = datetime.fromtimestamp(ts / 1000, tz=UTC)
                    if publish_date < cutoff_date:
                        logger.info(
                            "Cutoff reached (%s < %s). Stopping pagination.",
                            publish_date.strftime("%Y-%m-%d"),
                            cutoff_date.strftime("%Y-%m-%d"),
                        )
                        reached_cutoff = True
                        break

                if url in seen_urls:
                    continue
                seen_urls.add(url)

                collected.append(
                    {
                        "title": title.strip(),
                        "source_url": url,
                        "category": category or "",
                        "publish_date": publish_date.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                announcements_added_this_page += 1

            logger.info("Added %d announcement(s) from page %d.", announcements_added_this_page, page_num)

            if page_num >= total_pages or reached_cutoff:
                break

            page_num += 1
            time.sleep(0.5)  # Respect OKX rate limit

        return collected

    def _process_record(
        self, record: dict[str, Any]
    ) -> tuple[str | None, str, dict[str, Any]]:
        """Process an OKX announcement — fetch article body via HTML scraping."""
        source_url = record["source_url"]
        downloaded_pdfs: list[str] = []
        markdown_body = ""

        # Fetch article page HTML with retry logic
        for attempt in range(3):
            try:
                headers = {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                }
                article_response = self.http_session.get(source_url, headers=headers, timeout=30)
                if not article_response.ok:
                    raise RuntimeError(f"HTTP {article_response.status_code}")

                from bs4 import BeautifulSoup
                from markdownify import markdownify as md

                detail_page_soup = BeautifulSoup(article_response.text, "html.parser")

                # Target OKX article body container
                main_content = (
                    detail_page_soup.find("article")
                    or detail_page_soup.find(
                        "div",
                        class_=re.compile(r"article-content|article-body|help-detail", re.IGNORECASE),
                    )
                    or detail_page_soup.find("main")
                    or detail_page_soup.body
                )

                if main_content:
                    body_html = str(main_content)

                    # Scan for embedded PDF attachments
                    pdf_links = self._find_pdf_links_in_html(body_html, source_url)
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
                    markdown_body = md(str(main_content), heading_style="ATX", strip=["script", "style"])

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
            "category": record.get("category", ""),
            "publish_date": record.get("publish_date", "N/A"),
            "source_url": source_url,
            "Document_Type": "Announcement (Contains PDF)" if downloaded_pdfs else "Announcement",
        }

        if downloaded_pdfs:
            extra_fields["Downloaded_PDF_Paths"] = downloaded_pdfs

        body_text = self._generate_body(
            pdf_storage_path=downloaded_pdfs[0] if downloaded_pdfs else "",
            pdf_text=markdown_body,
            **extra_fields,
        )
        time.sleep(0.3)  # Cooldown between article fetches
        return body_text, downloaded_pdfs[0] if downloaded_pdfs else "", extra_fields

    @staticmethod
    def _find_pdf_links_in_html(html_content: str, base_url: str) -> list[tuple[str, str]]:
        """Extract PDF links from HTML content."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "html.parser")
            pdf_links: list[tuple[str, str]] = []

            for a_tag in soup.find_all("a", href=True):
                href = a_tag.get("href", "")
                if href.lower().endswith(".pdf"):
                    full_url = urljoin(base_url, href)
                    link_title = (a_tag.text or "attachment").strip()
                    pdf_links.append((full_url, link_title or "attachment"))

            return pdf_links
        except Exception:  # noqa: BLE001
            return []

    def _download_pdf(
        self, pdf_document_url: str, referrer_page_url: str
    ) -> str | None:
        """Download a PDF from OKX article attachments."""
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

    def _extra_frontmatter_fields(self, **extra_fields) -> dict:  # type: ignore[override]
        """Return OKX-specific frontmatter keys."""
        result: dict[str, object] = {
            "Category": extra_fields.get("category", ""),
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
        """Generate OKX body content with optional PDF attachments."""
        source_url = extra_fields.get("source_url", "")

        # Build body: article content (pdf_text here = converted markdown from HTML)
        body_content = (
            pdf_text.strip() if pdf_text else "*No article body content available.*"
        )

        return f"# Announcement\n\n**Note:** This announcement contains online article content.\n\n{'- [View Local PDF](../' + pdf_storage_path + ')\n' if pdf_storage_path else ''}- [Original Link]({source_url})\n\n---\n## Article Content\n\n{body_content}\n"


if __name__ == "__main__":
    main_from_parser(OKXAnnouncementScraper, "OKX Announcement Scraper")
