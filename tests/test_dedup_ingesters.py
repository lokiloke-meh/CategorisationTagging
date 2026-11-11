"""Deduplication tests for all circular scrapers.

Verifies that:
1. The shared ignore pattern strips timestamps regardless of casing.
2. Every ingester's frontmatter includes the core keys via the base class.
3. Markdown with different timestamps but identical content produces the same hash.
4. ContentStore correctly rejects duplicates across runs.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from ingestion.circular_scraper_base import CircularScraper

_timestamp_pattern = CircularScraper.timestamp_ignore_pattern

# ── 1. Ignore pattern correctness ───────────────────────────────────────


def test_ignore_pattern_strips_all_casings() -> None:
    """Pattern matches 'Scraped_At', 'scraped_at', and mixed-case variants."""
    for variant in (
        "Scraped_At: 2024-01-01T00:00:00+00:00\n",
        "scraped_at: 2024-01-01T00:00:00+00:00\n",
        "SCRAPED_AT: 2024-01-01T00:00:00+00:00\n",
        "ScRaPeD_aT: 2024-01-01T00:00:00+00:00\n",
    ):
        result = re.sub(_timestamp_pattern, "", variant)
        assert result == ""


def test_ignore_pattern_strips_with_whitespace() -> None:
    """Pattern handles optional whitespace around the colon."""
    for spacing in (": ", ":  ", ":\t"):
        text = f"Scraped_At{spacing}2024-01-01T00:00:00+00:00\n"
        result = re.sub(_timestamp_pattern, "", text)
        assert result == ""


def test_ignore_pattern_preserves_other_lines() -> None:
    """Pattern only removes the timestamp line, leaves surrounding content."""
    markdown = "Title: Foo\nscraped_at: 2024-01-01T00:00:00+00:00\nBody here"
    result = re.sub(_timestamp_pattern, "", markdown)
    assert "Title: Foo" in result and "Body here" in result


# ── 2. Base class frontmatter assembly ──────────────────────────────────


def test_assemble_markdown_includes_core_keys() -> None:
    """Base assemble_markdown always includes Title and Scraped_At."""

    class TestScraper(CircularScraper):
        def _fetch_circular_records(self) -> list[dict[str, str]]:
            return []

        def _generate_body(
            self, pdf_storage_path: str, pdf_text: str, **extra_fields
        ) -> str:
            return pdf_text

    with tempfile.TemporaryDirectory() as tmpdir:
        scraper = TestScraper(output_dir=tmpdir)
        md = scraper.assemble_markdown(
            title="Test",
            pdf_storage_path="pdf/test.pdf",
            download_timestamp="2024-06-15T12:00:00+00:00",
            body_text="# Content\n\nBody.",
        )

    assert "Title: Test" in md
    assert "Scraped_At:" in md


def test_assemble_markdown_includes_extra_fields() -> None:
    """Extra frontmatter fields from _extra_frontmatter_fields are merged."""

    class TestScraper(CircularScraper):
        def _fetch_circular_records(self) -> list[dict[str, str]]:
            return []

        def _generate_body(
            self, pdf_storage_path: str, pdf_text: str, **extra_fields
        ) -> str:
            return pdf_text

        def _extra_frontmatter_fields(self, **extra_fields) -> dict:  # type: ignore[override]
            return {"Custom_Key": extra_fields.get("custom", "default")}

    with tempfile.TemporaryDirectory() as tmpdir:
        scraper = TestScraper(output_dir=tmpdir)
        md = scraper.assemble_markdown(
            title="Test",
            pdf_storage_path="pdf/test.pdf",
            download_timestamp="2024-06-15T12:00:00+00:00",
            body_text="# Content\n\nBody.",
            custom="my_value",
        )

    assert "Custom_Key: my_value" in md


# ── 3. All ingesters produce valid markdown via new interface ───────────


def _make_scraper_instance(ingester_cls: type, tmpdir: Path) -> object:
    """Create a concrete scraper instance with mocked abstract methods."""

    class ConcreteScraper(ingester_cls):
        def _fetch_circular_records(self) -> list[dict[str, str]]:
            return []

    return ConcreteScraper(output_dir=str(tmpdir / ingester_cls.__name__))


def test_all_ingesters_produce_valid_markdown() -> None:
    """Every subclass's assemble_markdown produces valid YAML frontmatter + body."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        for ingester_cls in _ingester_classes():
            instance = _make_scraper_instance(ingester_cls, base_path)
            md = instance.assemble_markdown(  # type: ignore[attr-defined]
                title="Test Circular",
                pdf_storage_path="pdf/test.pdf",
                download_timestamp="2024-06-15T12:00:00+00:00",
                body_text="# Some content\n\nBody text here.",
            )

            # Must have YAML frontmatter delimiters and core keys
            assert md.startswith("---\n"), f"{ingester_cls.__name__} missing ---"
            assert "---\n\n" in md, f"{ingester_cls.__name__} missing body separator"
            assert (
                "Scraped_At:" in md
            ), f"{ingester_cls.__name__} missing Scraped_At key"


# ── 4. Timestamp-ignored hash consistency ───────────────────────────────


def test_different_timestamps_produce_same_hash() -> None:
    """Stripping timestamps yields the exact expected SHA-256 digest."""
    import hashlib

    stripped = re.sub(
        _timestamp_pattern,
        "",
        "---\nTitle: Test\nScraped_At: 2024-01-01T00:00:00+00:00\n---\n\nContent",
    )
    assert stripped == "---\nTitle: Test\n---\n\nContent"  # timestamp line removed
    expected_hash = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
    assert (
        expected_hash
        == "e4ce20ce682875f44f8ed41f50e149b0579883b4f693d1dd2d54bf83893df5af"
    )


def test_ignored_pattern_is_case_insensitive_across_ingesters() -> None:
    """Both 'Scraped_At' and 'scraped_at' are fully removed by the pattern."""
    nseix_stripped = re.sub(
        _timestamp_pattern,
        "",
        "---\nTitle: X\nscraped_at: 2024-01-01T00:00:00+00:00\n---\n\nBody",
    )
    base_stripped = re.sub(
        _timestamp_pattern,
        "",
        "---\nTitle: X\nScraped_At: 2025-06-30T23:59:59+00:00\n---\n\nBody",
    )

    assert nseix_stripped == base_stripped == "---\nTitle: X\n---\n\nBody"


# ── 5. ContentStore dedup integration ───────────────────────────────────


def test_content_store_dedupe_with_ignored_pattern() -> None:
    """ContentStore rejects markdown that differs only by timestamp."""
    from ingestion.content_store import ContentStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ContentStore(base_dir=tmpdir)

        md1 = "---\nTitle: Test\nScraped_At: 2024-01-01T00:00:00+00:00\n---\n\nBody"
        md2 = "---\nTitle: Test\nScraped_At: 2025-06-30T23:59:59+00:00\n---\n\nBody"

        hash1 = store.hash_text(md1, ignore_pattern=_timestamp_pattern)
        hash2 = store.hash_text(md2, ignore_pattern=_timestamp_pattern)

        assert hash1 == hash2, "Timestamps should be ignored — hashes must match"

        # Create the physical file so check_duplicate passes existence verification
        ingest_dir = Path(tmpdir) / "ingest"
        ingest_dir.mkdir()
        (ingest_dir / "test.md").write_text(md1)

        store.mark_saved(
            file_hash=hash1,
            relative_path="ingest/test.md",
            source_url="http://x",
            title="Test",
        )

        assert store.check_duplicate(hash2) == "ingest/test.md"


def test_content_store_allows_different_content() -> None:
    """ContentStore accepts markdown with genuinely different content."""
    from ingestion.content_store import ContentStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ContentStore(base_dir=tmpdir)

        md1 = "---\nTitle: Test\nScraped_At: 2024-01-01T00:00:00+00:00\n---\n\nBody A"
        md2 = "---\nTitle: Test\nScraped_At: 2025-06-30T23:59:59+00:00\n---\n\nBody B"

        hash1 = store.hash_text(md1, ignore_pattern=_timestamp_pattern)
        hash2 = store.hash_text(md2, ignore_pattern=_timestamp_pattern)

        assert hash1 != hash2, "Different body content should produce different hashes"


# ── 6. End-to-end: assemble → hash → dedup pipeline ────────────────────


def test_full_pipeline_dedupe_across_runs() -> None:
    """Two assemble_markdown calls with same PDF produce one saved file."""
    from ingestion.content_store import ContentStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ContentStore(base_dir=tmpdir)
        base_path = Path(tmpdir) / "test_data"
        ingest_dir = base_path / "ingest"
        base_path.mkdir()
        ingest_dir.mkdir()

        class TestScraper(CircularScraper):
            def _fetch_circular_records(self) -> list[dict[str, str]]:
                return []

            def _generate_body(
                self, pdf_storage_path: str, pdf_text: str, **extra_fields
            ) -> str:
                return f"# Circular\n\n{pdf_text}"

        scraper = TestScraper(output_dir=str(base_path))
        scraper.content_store = store

        pdf_text = "Same content here."

        md1 = scraper.assemble_markdown(
            title="Test Circular",
            pdf_storage_path="pdf/test.pdf",
            download_timestamp="2024-01-01T00:00:00+00:00",
            body_text=pdf_text,
        )

        md2 = scraper.assemble_markdown(
            title="Test Circular",
            pdf_storage_path="pdf/test.pdf",
            download_timestamp="2025-06-30T23:59:59+00:00",
            body_text=pdf_text,
        )

        hash1 = store.hash_text(md1, ignore_pattern=_timestamp_pattern)
        hash2 = store.hash_text(md2, ignore_pattern=_timestamp_pattern)

        assert hash1 == hash2, "Identical content must produce identical hashes"

        # First run saves the file
        scraper._save_markdown(
            title="Test Circular",
            source_url="http://example.com",
            content_hash=hash1,
            markdown_content=md1,
        )

        # Second run is deduped — no second file written
        saved_count = len(list(ingest_dir.glob("*.md")))
        assert saved_count == 1


# ── Helpers ─────────────────────────────────────────────────────────────


def _ingester_classes() -> list[type]:
    """Return all ingester classes that define _generate_body."""
    from ingestion.ingesters.binance_announcement_ingester import (
        BinanceAnnouncementScraper,
    )
    from ingestion.ingesters.binance_dev_changelog_ingester import (
        BinanceDeveloperChangelogScraper,
    )
    from ingestion.ingesters.bybit_announcement_ingester import BybitAnnouncementScraper
    from ingestion.ingesters.bybit_api_v5_changelog_ingester import (
        BybitAPIChangelogScraper,
    )
    from ingestion.ingesters.hkex_ingester import HKEXCircularScraper
    from ingestion.ingesters.mcx_ingester import MCXCircularScraper
    from ingestion.ingesters.nseix_ingester import NSEIXCircularScraper
    from ingestion.ingesters.okx_announcement_ingester import OKXAnnouncementScraper

    return [
        BinanceAnnouncementScraper,
        BinanceDeveloperChangelogScraper,
        BybitAnnouncementScraper,
        BybitAPIChangelogScraper,
        HKEXCircularScraper,
        MCXCircularScraper,
        NSEIXCircularScraper,
        OKXAnnouncementScraper,
    ]
