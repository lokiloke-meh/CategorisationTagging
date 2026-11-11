"""Binance Announcement scraper via Apex BAPI.

Both importable class and runnable CLI::

    from ingestion.ingesters.binance_ingester import BinanceAnnouncementScraper
    scraper = BinanceAnnouncementScraper(days_to_scrape=7)
    scraper.run()

    # or with explicit date range:
    scraper = BinanceAnnouncementScraper(
        from_date="20261001", to_date="20261031"
    )
    scraper.run()

CLI usage::

    uv run -m ingestion.ingesters.binance_announcement_ingester --days 7
    uv run -m ingestion.ingesters.binance_announcement_ingester \
        --from-date 20261001 --to-date 20261031
    uv run -m ingestion.ingesters.binance_announcement_ingester \
        --days 7 --catalogs 'Latest Binance News, Maintenance Updates'
    uv run -m ingestion.ingesters.binance_announcement_ingester \
        --days 7 --output /custom/path

"""

from __future__ import annotations

import json
import logging
import re
import time
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curl_cffi import requests as cffi_requests

from ingestion.circular_scraper_base import (
    CircularScraper,
    main_from_parser,
)

logger = logging.getLogger(__name__)


# ── Catalog reference (for user discoverability, not runtime lookup) ───
# Users specify --catalogs by these names; scraper fetches actual data
# from the API at runtime.

_CATALOG_NAMES: dict[str, int] = {
    "New Cryptocurrency Listing": 48,
    "Latest Binance News": 49,
    "API Updates": 51,
    "Latest Activities": 93,
    "Crypto Airdrop": 128,
    "Maintenance Updates": 157,
    "Delisting": 161,
}


class BinanceAnnouncementScraper(CircularScraper):
    """Scrape Binance announcements via the Apex BAPI and save as markdown."""

    storage_name = "binance_announcements"

    def __init__(
        self,
        days_to_scrape: int = 30,
        output_dir: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        catalogs: str | None = None,
    ) -> None:
        self.from_date = from_date
        self.to_date = to_date
        self._catalog_filter_ids = self._parse_catalog_names(catalogs)

        super().__init__(days_to_scrape=days_to_scrape, output_dir=output_dir)

        # Apex BAPI endpoints
        self.list_api_url = (
            "https://www.binance.com/bapi/apex/v1/public/apex/cms/article/list/query"
        )
        self.detail_api_url = (
            "https://www.binance.com/bapi/apex/v1/public/apex/cms/article/detail/query"
        )

        # curl_cffi session with Chrome impersonation for WAF bypass
        self.http_session = cffi_requests.Session(impersonate="chrome120")
        self.http_session.headers.update(
            {
                "clienttype": "web",
                "lang": "en",
                "Accept-Language": "en-US,en;q=0.9",
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
        """Add Binance-specific CLI arguments on top of the shared parser."""
        available = ", ".join(sorted(_CATALOG_NAMES.keys()))
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
        parser.add_argument(
            "--catalogs",
            "-c",
            type=str,
            default=None,
            help=(
                f"Comma-separated catalog names to scrape. "
                f"Available: {available}. Leave empty for all catalogs."
            ),
        )
        return parser

    # ── Catalog filtering ───────────────────────────────────────────────

    @staticmethod
    def _parse_catalog_names(catalog_input: str | None) -> set[int] | None:
        """Parse comma-separated catalog names into a set of IDs.

        Returns None if no filter specified. Raises ValueError on unknown name.
        Matches case-insensitively against the reference list.
        """
        if not catalog_input or not catalog_input.strip():
            return None

        selected_ids: set[int] = set()
        for name in (n.strip() for n in catalog_input.split(",")):
            matched = False
            for known_name, catalog_id in _CATALOG_NAMES.items():
                if known_name.lower() == name.lower():
                    selected_ids.add(catalog_id)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Unknown catalog: '{name}'")
        return selected_ids

    # ── Date range resolution ───────────────────────────────────────────

    def _resolve_date_range(self) -> tuple[datetime | None, datetime | None]:
        """Return (from_datetime, to_datetime) from CLI args or days_to_scrape."""
        if self.from_date and self.to_date:
            try:
                start = datetime.strptime(self.from_date.strip(), "%Y%m%d").replace(
                    tzinfo=UTC
                )
                end = datetime.strptime(self.to_date.strip(), "%Y%m%d").replace(
                    hour=23, minute=59, tzinfo=UTC
                )
                return start, end
            except ValueError as exc:
                logger.warning("Invalid date format: %s. Expected YYYYMMDD.", exc)
                return None, None
        # Fall back to days_to_scrape from base class
        today = datetime.now(UTC)
        start = CircularScraper._days_start_date(today, self.days_to_scrape)
        return start, today

    # ── JSON-to-markdown conversion ─────────────────────────────────────

    @staticmethod
    def _convert_json_tree_to_markdown(node: dict) -> str:
        """Recursively convert Binance JSON content tree to markdown."""
        if not node or "node" not in node:
            return ""

        tag = node.get("tag", "")
        text_parts: list[str] = []

        for child in node.get("child", []):
            if child.get("node") == "text":
                # Text nodes use key 'text', NOT 'content'
                raw_text = child.get("text", "")
                # Normalize whitespace but preserve paragraph breaks
                normalized = " ".join(raw_text.split())
                text_parts.append(normalized)
            else:
                converted = BinanceAnnouncementScraper._convert_json_tree_to_markdown(
                    child
                )
                if converted.strip():
                    text_parts.append(converted)

        result = "".join(text_parts).strip()

        # Tag-based formatting
        if tag == "p":
            return f"\n\n{result}\n" if result else "\n"
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            return f"\n\n{'#' * level} {result}\n" if result else "\n"
        elif tag in ("strong", "b"):
            return f"**{result}**" if result else ""
        elif tag in ("em", "i"):
            return f"*{result}*" if result else ""
        elif tag == "a":
            href = node.get("attr", {}).get("href", "") or node.get("attr", {}).get(
                "appurl", ""
            )
            if isinstance(href, str) and "%suffixOrigin%" in href:
                # Strip placeholder URLs
                href = href.replace("%suffixOrigin%", "").replace("%locale%", "")
            if href and not href.startswith("%") and result:
                return f"[{result}]({href})"
            return result
        elif tag == "ul":
            items: list[str] = []
            for child in node.get("child", []):
                if child.get("tag") == "li":
                    item_text = (
                        BinanceAnnouncementScraper._convert_json_tree_to_markdown(
                            child
                        ).strip()
                    )
                    if item_text:
                        items.append(f"- {item_text}")
            return "\n\n" + "\n".join(items) + "\n" if items else ""

        return result

    # ── PDF download (Binance-specific) ────────────────────────────────

    def _download_pdf(
        self, pdf_document_url: str, referrer_page_url: str
    ) -> str | None:
        """Download a PDF from Binance article attachments."""
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
        """Fetch all articles from all catalogs within the configured date range.

        Paginates each catalog's article list until a record falls outside
        the user's search window (articles sorted newest-first).
        """
        collected: list[dict[str, str]] = []

        from_dt, to_dt = self._resolve_date_range()
        if from_dt is None or to_dt is None:
            logger.info(
                "Both --from-date and --to-date are required when not using --days."
            )
            return collected
        seen_codes: set[str] = set()

        try:
            list_params = {
                "type": "1",
                "pageNo": "1",
                "pageSize": "50",
            }
            resp = self.http_session.get(
                self.list_api_url, params=list_params, timeout=30
            )
            if not resp.ok:
                self.collected_errors.append(
                    f"List API returned HTTP {resp.status_code}"
                )
                return collected

            data = resp.json().get("data", {})
        except cffi_requests.RequestException as exc:
            self.collected_errors.append(f"Failed to fetch list: {exc}")
            return collected

        catalogs = data.get("catalogs", [])
        if not catalogs:
            logger.warning("No catalogs returned from list API.")
            return collected

        # Filter to user-specified catalogs (None = all)
        selected_catalogs = [
            cat
            for cat in catalogs
            if self._catalog_filter_ids is None
            or cat.get("catalogId") in self._catalog_filter_ids
        ]
        logger.info(
            "Scraping %d catalog(s): %s",
            len(selected_catalogs),
            ", ".join(
                cat.get("catalogName", str(cat.get("catalogId")))
                for cat in selected_catalogs
            ),
        )

        for catalog in selected_catalogs:
            catalog_id = catalog.get("catalogId")
            catalog_name = catalog.get("catalogName", "Unknown")
            articles = catalog.get("articles", [])
            if not articles:
                continue

            # Paginate this catalog's articles
            page_no = 1
            max_pages = 20  # Safety limit to avoid infinite loops
            while page_no <= max_pages:
                in_range_count = 0
                hit_cutoff = False
                for article in articles:
                    code = article.get("code")
                    title = article.get("title", "Untitled")
                    release_ts = article.get("releaseDate")

                    if not code or not title:
                        continue

                    # Check date range (millisecond epoch timestamp)
                    if release_ts is not None:
                        article_date = datetime.fromtimestamp(release_ts / 1000, tz=UTC)
                        if article_date < from_dt or article_date > to_dt:
                            logger.info(
                                "Catalog %d (%s): article '%s' (%s) outside range [%s..%s]. Stopping pagination.",
                                catalog_id,
                                catalog_name,
                                title,
                                article_date.strftime("%Y-%m-%d"),
                                from_dt.strftime("%Y-%m-%d"),
                                to_dt.strftime("%Y-%m-%d"),
                            )
                            hit_cutoff = True
                            break  # breaks out of article iteration
                        release_date = article_date
                    else:
                        release_date = datetime.now(UTC)

                    release_str = release_date.strftime("%Y-%m-%d %H:%M")
                    source_url = (
                        f"https://www.binance.com/en/support/announcement/{code}"
                    )

                    if code in seen_codes:
                        continue
                    seen_codes.add(code)

                    collected.append(
                        {
                            "title": title.strip(),
                            "article_code": code,
                            "catalog_id": str(catalog_id),
                            "catalog_name": catalog_name,
                            "release_date": release_str,
                            "source_url": source_url,
                        }
                    )
                    in_range_count += 1

                logger.info(
                    "Catalog %d (%s): fetched %d article(s) on page %d.",
                    catalog_id,
                    catalog_name,
                    in_range_count,
                    page_no,
                )

                # If we hit cutoff, stop paginating this catalog
                if hit_cutoff:
                    break

                # Paginate: fetch next page
                try:
                    list_params_page = {
                        "type": "1",
                        "pageNo": str(page_no + 1),
                        "pageSize": "50",
                    }
                    resp = self.http_session.get(
                        self.list_api_url, params=list_params_page, timeout=30
                    )
                    if not resp.ok:
                        logger.warning(
                            "List API returned HTTP %d on page %d.",
                            resp.status_code,
                            page_no,
                        )
                        break

                    data = resp.json().get("data", {})
                    catalogs = data.get("catalogs", [])
                    if not catalogs:
                        break

                    # Find this catalog in the new page response
                    articles = []
                    for cat in catalogs:
                        if cat.get("catalogId") == catalog_id:
                            articles = cat.get("articles", [])
                            break

                    if not articles:
                        logger.info(
                            "Catalog %d (%s): no more pages.",
                            catalog_id,
                            catalog_name,
                        )
                        break

                except cffi_requests.RequestException as exc:
                    self.collected_errors.append(
                        f"Pagination error on page {page_no}: {exc}"
                    )
                    break

                # 15-second cooldown between list/pagination calls
                time.sleep(15)
                page_no += 1

        return collected

    def _process_record(
        self, record: dict[str, Any]
    ) -> tuple[str | None, str, dict[str, Any]]:
        """Process Binance record — fetches detail API, converts JSON tree to markdown."""
        article_code = record.get("article_code", "")
        downloaded_pdfs: list[str] = []
        markdown_body = ""

        # Fetch detail API with retry logic
        for attempt in range(3):
            try:
                resp = self.http_session.get(
                    self.detail_api_url,
                    params={"articleCode": article_code},
                    timeout=30,
                )
                if not resp.ok:
                    raise RuntimeError(f"HTTP {resp.status_code}")

                detail_data = resp.json().get("data", {})
                body_json_str = detail_data.get("body", "")

                if body_json_str:
                    body_tree = json.loads(body_json_str)
                    markdown_body = self._convert_json_tree_to_markdown(body_tree)

                    # Scan for PDF links in the converted content
                    pdf_links = self._find_pdf_links_in_markdown(markdown_body)
                    if pdf_links:
                        for pdf_url, link_title in pdf_links:
                            logger.info("  -> Found embedded PDF: %s", link_title)
                            saved = self._download_pdf(
                                pdf_document_url=pdf_url,
                                referrer_page_url=record["source_url"],
                            )
                            if saved:
                                downloaded_pdfs.append(saved)

                break  # Success — exit retry loop

            except (RuntimeError, json.JSONDecodeError, OSError) as exc:
                wait_time = 15 * (2**attempt)  # 15 → 30 → 60
                logger.warning(
                    "Detail fetch failed for %s (attempt %d/3): %s. Retrying in %ds...",
                    article_code,
                    attempt + 1,
                    exc,
                    wait_time,
                )
                if attempt < 2:
                    time.sleep(wait_time)
                else:
                    self.collected_errors.append(
                        f"Failed to fetch detail for {article_code}: {exc}"
                    )

        extra_fields = {
            "source_url": record["source_url"],
            "article_code": article_code,
            "catalog_id": str(record.get("catalog_id", 0)),
            "catalog_name": record.get("catalog_name", "Unknown"),
            "release_date": record.get("release_date", "N/A"),
        }

        if downloaded_pdfs:
            extra_fields["Document_Type"] = "Announcement (Contains PDF)"
            extra_fields["Downloaded_PDF_Paths"] = downloaded_pdfs
        else:
            extra_fields["Document_Type"] = "Announcement"

        body_text = self._generate_body(
            pdf_storage_path=downloaded_pdfs[0] if downloaded_pdfs else "",
            pdf_text=markdown_body,
            **extra_fields,
        )
        time.sleep(30)  # Cooldown between detail API calls
        return body_text, downloaded_pdfs[0] if downloaded_pdfs else "", extra_fields

    @staticmethod
    def _find_pdf_links_in_markdown(markdown_text: str) -> list[tuple[str, str]]:
        """Extract PDF links from markdown text (from link syntax)."""

        pattern = r"\[([^\]]+)\]\(([^)]+\.pdf(?:\?[^)]*)?)\)"
        matches = re.findall(pattern, markdown_text)
        return [(url.strip(), title.strip()) for title, url in matches]

    def _extra_frontmatter_fields(self, **extra_fields) -> dict:  # type: ignore[override]
        """Return Binance-specific frontmatter keys."""
        result: dict[str, object] = {
            "Article_Code": extra_fields.get("article_code", "N/A"),
            "Catalog_ID": str(extra_fields.get("catalog_id", 0)),
            "Catalog_Name": extra_fields.get("catalog_name", "Unknown"),
            "Release_Date": extra_fields.get("release_date", "N/A"),
            "Source_URL": extra_fields.get("source_url", ""),
            "Document_Type": extra_fields.get("Document_Type", "Announcement"),
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
        """Generate Binance body content with optional PDF attachments."""
        source_url = extra_fields.get("source_url", "")

        # Build body: article content (pdf_text here = converted markdown from JSON tree)
        body_content = (
            pdf_text.strip() if pdf_text else "*No article body content available.*"
        )

        return f"# Announcement\n\n**Note:** This announcement contains online article content.\n\n{'- [View Local PDF](../' + pdf_storage_path + ')\n' if pdf_storage_path else ''}- [Original Link]({source_url})\n\n---\n## Article Content\n\n{body_content}\n"


if __name__ == "__main__":
    from ingestion.circular_scraper_base import main_from_parser

    main_from_parser(BinanceAnnouncementScraper, "Binance Announcement Scraper")
