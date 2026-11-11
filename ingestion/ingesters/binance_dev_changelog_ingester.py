"""Binance Developer Docs multi-product changelog scraper.

Dynamically discovers every product changelog via ``llms.txt`` and the
navigation tree, then slices each date heading into an individual announcement.

CLI usage::

    uv run -m ingestion.ingesters.binance_dev_changelog_ingester --days 30

"""

from __future__ import annotations

import logging
import re
import time
from argparse import ArgumentParser
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag
from curl_cffi import requests as cffi_requests
from markdownify import markdownify as md

from ingestion.circular_scraper_base import (
    CircularScraper,
    main_from_parser,
)

logger = logging.getLogger(__name__)


class BinanceDeveloperChangelogScraper(CircularScraper):
    """Discover and scrape every product changelog across Binance Developer Docs."""

    storage_name = "binance_dev_changelogs"
    base_url = "https://developers.binance.com"

    def __init__(self, days_to_scrape: int = 30, output_dir: str | None = None) -> None:
        super().__init__(days_to_scrape=days_to_scrape, output_dir=output_dir)

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

    # ── Discovery helpers ───────────────────────────────────────────────

    @staticmethod
    def _extract_product_name_from_url(url: str) -> str:
        """Derive a human-readable product name from a changelog URL path.

        Example: ``https://developers.binance.com/en/docs/products/wallet/change-log``
        → ``Wallet``.  Compound slugs like ``derivatives-trading-usds-futures`` are
        split into words and title-cased.
        """
        path_parts = [p for p in urlparse(url).path.split("/") if p]

        changelog_index = -1
        for idx, part in enumerate(path_parts):
            if re.search(r"change[\s_-]*log", part, re.IGNORECASE):
                changelog_index = idx
                break

        if changelog_index <= 0:
            return "General API"

        product_slug = path_parts[changelog_index - 1]
        return re.sub(r"-", " ", product_slug).title()

    def _discover_from_llms_manifest(self) -> dict[str, str]:
        """Read ``/en/docs/llms.txt`` and extract changelog URLs with product names.

        The manifest lists all docs under a single ``## Documentation`` header,
        so the product name is derived from the URL path slug (e.g.
        ``/products/wallet/change-log`` → ``Wallet``).
        """
        manifest_url = f"{self.base_url}/en/docs/llms.txt"
        logger.info("Querying dynamic AI manifest: %s", manifest_url)

        changelog_registry: dict[str, str] = {}

        try:
            manifest_response = self.http_session.get(manifest_url, timeout=20)
            if not manifest_response.ok:
                logger.warning(
                    "Manifest returned HTTP %d — skipping discovery.",
                    manifest_response.status_code,
                )
                return changelog_registry

            for raw_line in manifest_response.text.splitlines():
                line = raw_line.strip()

                # Match markdown links: [text](path)
                link_match = re.search(r"\]\(([^)]+)\)", line)
                if not link_match:
                    continue

                raw_path = link_match.group(1).strip()
                full_url = urljoin(f"{self.base_url}/en/", raw_path).split("#")[0]

                # Only keep changelog URLs
                if not re.search(r"change[\s_-]*log", raw_path, re.IGNORECASE):
                    continue

                product_name = self._extract_product_name_from_url(full_url)
                changelog_registry[full_url] = product_name

            logger.info(
                "Discovered %d changelogs from llms.txt.", len(changelog_registry)
            )

        except cffi_requests.RequestException as exc:  # type: ignore[name-defined]
            self.collected_errors.append(f"Failed to query llms.txt: {exc}")

        return changelog_registry

    def _discover_from_navigation_tree(self) -> dict[str, str]:
        """Spider the root documentation homepage and extract changelog links."""
        root_url = f"{self.base_url}/en/docs/"
        logger.info("Spidering root documentation navigation: %s", root_url)

        changelog_registry: dict[str, str] = {}

        try:
            nav_response = self.http_session.get(root_url, timeout=20)
            if not nav_response.ok:
                return changelog_registry

            parsed_soup = BeautifulSoup(nav_response.text, "html.parser")

            for anchor_tag in parsed_soup.find_all("a", href=True):
                href_value = anchor_tag["href"]
                link_text = anchor_tag.get_text(separator=" ", strip=True)

                if not re.search(
                    r"change[\s_-]*log", link_text, re.IGNORECASE
                ) and not re.search(r"change[\s_-]*log", href_value, re.IGNORECASE):
                    continue

                full_url = urljoin(self.base_url, href_value).split("#")[0]
                product_name = self._extract_product_name_from_url(full_url)
                changelog_registry[full_url] = product_name

            logger.info(
                "Discovered %d changelogs from navigation tree.",
                len(changelog_registry),
            )

        except cffi_requests.RequestException as exc:  # type: ignore[name-defined]
            self.collected_errors.append(f"Failed to spider navigation tree: {exc}")

        return changelog_registry

    def discover_all_changelogs(self) -> dict[str, str]:
        """Merge manifest and navigation discovery into a single URL→product map."""
        registry = self._discover_from_llms_manifest()

        nav_results = self._discover_from_navigation_tree()
        for url, product in nav_results.items():
            if url not in registry:
                registry[url] = product

        return registry

    # ── Date parsing helpers ────────────────────────────────────────────

    @staticmethod
    def _parse_date_string(date_raw: str) -> datetime | None:
        """Parse ``YYYY-MM-DD`` or ``Mon DD, YYYY`` into a timezone-aware datetime."""
        cleaned = re.sub(r"(?:st|nd|rd|th)", "", date_raw).strip().rstrip(",")

        for fmt in ("%Y-%m-%d", "%b %d %Y", "%B %d %Y"):
            try:
                parsed_dt = datetime.strptime(cleaned, fmt)  # noqa: DTZ007
                return parsed_dt.replace(tzinfo=UTC)
            except ValueError:
                continue

        return None

    # ── DOM slicing helper ──────────────────────────────────────────────

    @staticmethod
    def _markdown_to_html(raw_md: str) -> str:
        """Convert raw markdown (``## DATE`` headings) into basic HTML for BeautifulSoup.

        Binance Developer Docs serves some changelog pages as plain ``.md`` files
        (Content-Type: application/octet-stream).  This converter wraps them so
        BeautifulSoup can parse the structure.
        """
        lines = raw_md.splitlines()
        html_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading_match:
                hash_count = len(heading_match.group(1))
                level = min(hash_count, 6)
                content = heading_match.group(2).strip()
                html_lines.append(f"<h{level}>{content}</h{level}>")
            elif not stripped:
                html_lines.append("<br>")
            elif line.startswith(("- ", "* ")):
                html_lines.append(f"<li>{stripped[2:]}</li>")
            else:
                html_lines.append(f"<p>{stripped}</p>")

        return f"<article>\n{chr(10).join(html_lines)}\n</article>"

    def _slice_page_entries(
        self,
        html_content: str,
        page_url: str,
        product_name: str,
    ) -> list[dict[str, Any]]:
        """Segment a changelog page between date headings into distinct entries.

        Handles both HTML pages (Docusaurus-style) and raw markdown files
        served as ``application/octet-stream`` by Binance Developer Docs.
        """
        # Detect raw markdown: no <html> wrapper → convert to basic HTML
        if "<html" not in html_content.lower()[:500]:
            html_content = self._markdown_to_html(html_content)

        parsed_soup = BeautifulSoup(html_content, "html.parser")

        main_container = (
            parsed_soup.find("article")
            or parsed_soup.find("main")
            or parsed_soup.find(
                "div", class_=re.compile(r"content|markdown|theme-doc", re.IGNORECASE)
            )
            or parsed_soup.body
        )
        if not main_container:
            return []

        date_pattern = re.compile(
            r"\b(\d{4}-\d{2}-\d{2})\b|"
            r"\b([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b"
        )
        date_headings: list[tuple[Any, str, datetime]] = []

        for heading_element in main_container.find_all(["h1", "h2", "h3", "h4"]):
            raw_text = heading_element.get_text(separator=" ", strip=True).replace(
                "\u200b", ""
            )
            date_match = date_pattern.search(raw_text)
            if not date_match:
                continue

            date_raw_string = date_match.group(0)
            parsed_date_obj = self._parse_date_string(date_raw_string)
            if parsed_date_obj is None:
                continue

            date_headings.append((heading_element, date_raw_string, parsed_date_obj))

        entries: list[dict[str, Any]] = []
        for slice_idx, (h_tag, date_raw_str, date_obj) in enumerate(date_headings):
            next_heading = (
                date_headings[slice_idx + 1][0]
                if slice_idx + 1 < len(date_headings)
                else None
            )

            anchor_id = h_tag.get("id") or date_raw_str
            entry_url = f"{page_url}#{anchor_id}"

            content_nodes: list[str] = []
            sibling_node = h_tag.next_sibling
            while sibling_node:
                if sibling_node == next_heading:
                    break
                if isinstance(sibling_node, Tag):
                    if next_heading is not None and next_heading in getattr(
                        sibling_node, "descendants", []
                    ):
                        break
                    content_nodes.append(str(sibling_node))
                sibling_node = sibling_node.next_sibling

            slice_html = "".join(content_nodes)
            cleaned_soup = BeautifulSoup(slice_html, "html.parser")

            for anchor_tag in cleaned_soup.find_all("a", href=True):
                anchor_tag["href"] = urljoin(page_url, anchor_tag["href"])

            entries.append(
                {
                    "product": product_name,
                    "date": date_obj.strftime("%Y-%m-%d"),
                    "url": entry_url,
                    "html": str(cleaned_soup),
                }
            )

        return entries

    # ── Abstract method implementations ────────────────────────────────

    def _fetch_circular_records(self) -> list[dict[str, str]]:
        """Discover all product changelogs and slice each into date-based records."""
        collected: list[dict[str, str]] = []
        cutoff_date = CircularScraper._days_start_date(
            datetime.now(UTC), self.days_to_scrape
        )

        changelog_registry = self.discover_all_changelogs()
        if not changelog_registry:
            logger.warning("No changelog URLs discovered — aborting.")
            return collected

        logger.info(
            "Scraping %d product changelog(s) for the past %d days.",
            len(changelog_registry),
            self.days_to_scrape,
        )

        url_index = 0
        total_changelogs = len(changelog_registry)

        for page_url, product_name in changelog_registry.items():
            url_index += 1
            logger.info(
                "[%d/%d] Scraping %s: %s",
                url_index,
                total_changelogs,
                product_name,
                page_url,
            )

            try:
                fetch_response = self.http_session.get(page_url, timeout=25)
                if not fetch_response.ok:
                    self.collected_errors.append(
                        f"Changelog returned HTTP {fetch_response.status_code} for {product_name}"
                    )
                    continue

                canonical_url = str(fetch_response.url)
            except cffi_requests.RequestException as exc:  # type: ignore[name-defined]
                self.collected_errors.append(f"Failed to fetch {page_url}: {exc}")
                continue

            page_entries = self._slice_page_entries(
                html_content=fetch_response.text,
                page_url=canonical_url,
                product_name=product_name,
            )

            in_window_count = 0
            for entry in page_entries:
                if (
                    entry["date"] is None
                    or self._parse_date_string(entry["date"]) is None
                ):
                    continue

                entry_datetime = self._parse_date_string(entry["date"])
                assert entry_datetime is not None  # just typed above

                if entry_datetime < cutoff_date:
                    break

                title = f"Binance {product_name} API Update - {entry['date']}"
                collected.append(
                    {
                        "title": title,
                        "source_url": entry["url"],
                        "publish_date": entry["date"],
                        "html_body": entry["html"],
                        "category_key": product_name.lower().replace(" ", "_"),
                        "category_title": f"{product_name} Changelog",
                    }
                )
                in_window_count += 1

            logger.info(
                "Saved %d announcement(s) for %s.",
                in_window_count,
                product_name,
            )
            time.sleep(0.3)  # Respect Binance rate limit

        return collected

    def _process_record(
        self, record: dict[str, Any]
    ) -> tuple[str | None, str, dict[str, Any]]:
        """Process a changelog date slice — convert HTML to markdown."""
        html_content = record.get("html_body", "")
        source_url = record["source_url"]

        try:
            converted_markdown = md(
                html_content, heading_style="ATX", strip=["script", "style"]
            )
        except Exception:  # noqa: BLE001
            logger.warning("Markdown conversion failed for %s", source_url)
            converted_markdown = ""

        extra_fields: dict[str, Any] = {
            "category_key": record.get("category_key", ""),
            "category_title": record.get("category_title", ""),
            "publish_date": record.get("publish_date", ""),
            "source_url": source_url,
            "Document_Type": "Developer Changelog Announcement",
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
                "Document_Type", "Developer Changelog Announcement"
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
            f"**Note:** This changelog entry is from the Binance Developer Documentation.\n\n"
            f"- [Original Link]({source_url})\n\n"
            f"---\n## Update Content\n\n{article_content}\n"
        )


if __name__ == "__main__":
    main_from_parser(BinanceDeveloperChangelogScraper, "Binance Dev Changelog Scraper")
