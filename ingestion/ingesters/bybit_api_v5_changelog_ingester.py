"""Bybit V5 Changelog scraper via single-page DOM slicing.

Parses ``https://bybit-exchange.github.io/docs/changelog/v5`` and treats each
date heading as an individual announcement.

CLI usage::

    uv run -m ingestion.ingesters.bybit_api_v5_changelog_ingester --days 30

"""

from __future__ import annotations

import logging
import re
from argparse import ArgumentParser
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests

from ingestion.circular_scraper_base import (
    CircularScraper,
    main_from_parser,
)

logger = logging.getLogger(__name__)


class BybitAPIChangelogScraper(CircularScraper):
    """Parse the single-page Bybit V5 changelog and output each date as markdown."""

    storage_name = "bybit_api_v5_changelogs"
    target_url = "https://bybit-exchange.github.io/docs/changelog/v5"

    def __init__(self, days_to_scrape: int = 30, output_dir: str | None = None) -> None:
        super().__init__(days_to_scrape=days_to_scrape, output_dir=output_dir)

        # curl_cffi session with Chrome impersonation for the GitHub Pages site
        self.http_session = cffi_requests.Session(impersonate="chrome120")  # type: ignore[attr-defined]
        self.http_session.headers.update(
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            }
        )

    # ── CLI extension ───────────────────────────────────────────────────

    @staticmethod
    def extend_cli_parser(parser: ArgumentParser) -> ArgumentParser:
        """Add Changelog-specific CLI arguments on top of the shared parser."""
        return parser

    # ── Abstract method implementations ────────────────────────────────

    def _fetch_circular_records(self) -> list[dict[str, str]]:
        """Fetch and slice the changelog page into date-based records.

        The changelog is a single Docusaurus page with ``YYYY-MM-DD`` headings.
        We find all such headings, collect sibling content between consecutive
        dates, and return each as a record whose ``html_body`` already contains
        the markdown-ready HTML slice.
        """
        records: list[dict[str, str]] = []

        try:
            fetch_response = self.http_session.get(self.target_url, timeout=30)
            if not fetch_response.ok:
                self.collected_errors.append(
                    f"Changelog page returned HTTP {fetch_response.status_code}"
                )
                return records

            fetched_html = fetch_response.text

        except cffi_requests.RequestException as exc:  # type: ignore[name-defined]
            self.collected_errors.append(f"Failed to fetch changelog page: {exc}")
            return records

        parsed_soup = BeautifulSoup(fetched_html, "html.parser")

        main_content_container = (
            parsed_soup.find("div", class_=re.compile(r"theme-doc-markdown|markdown"))
            or parsed_soup.find("article")
            or parsed_soup.find("main")
        )
        if not main_content_container:
            self.collected_errors.append(
                "Could not locate changelog markdown container."
            )
            return records

        date_pattern = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})")
        date_headings: list[tuple[Any, str]] = []

        for heading_element in main_content_container.find_all(
            ["h1", "h2", "h3", "h4"]
        ):
            raw_text = heading_element.get_text(strip=True).replace(
                "\u200b", ""
            )  # Remove zero-width spaces
            date_match = date_pattern.match(raw_text)
            if date_match:
                date_headings.append((heading_element, date_match.group(1)))

        cutoff_date = CircularScraper._days_start_date(
            datetime.now(UTC), self.days_to_scrape
        )

        for slice_index, (current_heading, date_string) in enumerate(date_headings):
            entry_date = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=UTC)
            if entry_date < cutoff_date:
                logger.info(
                    "Reached cutoff (%s < %s). Stopping.",
                    date_string,
                    cutoff_date.strftime("%Y-%m-%d"),
                )
                break

            next_heading = (
                date_headings[slice_index + 1][0]
                if slice_index + 1 < len(date_headings)
                else None
            )

            anchor_id = current_heading.get("id") or date_string
            record_url = f"{self.target_url}#{anchor_id}"

            content_nodes: list[str] = []
            sibling_node = current_heading.next_sibling
            while sibling_node:
                if sibling_node == next_heading:
                    break
                if hasattr(sibling_node, "__class__") and type(
                    sibling_node
                ).__name__ in (
                    "Tag",
                    "NavigableString",
                ):
                    if next_heading is not None and next_heading in getattr(
                        sibling_node, "descendants", []
                    ):
                        break
                    content_nodes.append(str(sibling_node))
                sibling_node = sibling_node.next_sibling

            raw_slice_html = "".join(content_nodes)

            cleaned_soup = BeautifulSoup(raw_slice_html, "html.parser")

            for hash_link in cleaned_soup.find_all(
                "a", class_=re.compile(r"hash-link")
            ):
                hash_link.decompose()

            for anchor_tag in cleaned_soup.find_all("a", href=True):
                anchor_tag["href"] = urljoin(self.target_url, anchor_tag["href"])

            records.append(
                {
                    "title": f"Bybit V5 API Update - {date_string}",
                    "source_url": record_url,
                    "publish_date": date_string,
                    "html_body": str(cleaned_soup),
                }
            )

        logger.info("Sliced %d changelog entry/entries from page.", len(records))
        return records

    def _process_record(
        self, record: dict[str, Any]
    ) -> tuple[str | None, str, dict[str, Any]]:
        """Process a changelog date slice — convert HTML to markdown."""
        html_content = record.get("html_body", "")
        source_url = record["source_url"]

        try:
            from markdownify import markdownify as md

            converted_markdown = md(
                html_content, heading_style="ATX", strip=["script", "style"]
            )
        except Exception:  # noqa: BLE001
            logger.warning("Markdown conversion failed for %s", source_url)
            converted_markdown = ""

        extra_fields: dict[str, Any] = {
            "category_key": "changelog",
            "category_title": "V5 API Changelog",
            "publish_date": record.get("publish_date", ""),
            "source_url": source_url,
            "Document_Type": "Changelog Announcement",
        }

        body_text = self._generate_body(
            pdf_storage_path="",
            pdf_text=converted_markdown,
            **extra_fields,
        )
        return (body_text, "", extra_fields)

    @staticmethod
    def _find_pdf_links_in_html(  # type: ignore[no-untyped-def]
        html_content: str, base_url: str
    ) -> list[tuple[str, str]]:
        """Extract PDF links from HTML content (required by base class)."""
        try:
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

    def _extra_frontmatter_fields(self, **extra_fields: Any) -> dict[str, Any]:
        """Return Changelog-specific frontmatter keys."""
        return {
            "Category_Key": extra_fields.get("category_key", ""),
            "Category_Title": extra_fields.get("category_title", ""),
            "Publish_Date": extra_fields.get("publish_date", ""),
            "Source_URL": extra_fields.get("source_url", ""),
            "Document_Type": extra_fields.get(
                "Document_Type", "Changelog Announcement"
            ),
        }

    def _generate_body(
        self,
        pdf_storage_path: str,
        pdf_text: str,
        **extra_fields: Any,
    ) -> str:
        """Generate changelog body content."""
        source_url = extra_fields.get("source_url", "")
        article_content = (
            pdf_text.strip() if pdf_text else "*No changelog content available.*"
        )

        return (
            f"# Changelog\n\n"
            f"**Note:** This changelog entry is from the Bybit V5 API documentation.\n\n"
            f"- [Original Link]({source_url})\n\n"
            f"---\n## Update Content\n\n{article_content}\n"
        )


if __name__ == "__main__":
    main_from_parser(BybitAPIChangelogScraper, "Bybit V5 Changelog Scraper")
