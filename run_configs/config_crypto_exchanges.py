from pathlib import Path
from typing import ClassVar

# -- Tag examples (source of truth for available tags) ----------------------

EXAMPLES_DIR = Path(__file__).parent.parent / "tagging" / "tagging_examples_yaml"

TAG_EXAMPLES: dict[str, str | Path] = {
    "OrderInfrastructureChanges": EXAMPLES_DIR / "OrderInfrastructureChanges.yaml",
    "AllCorporateActions": EXAMPLES_DIR / "AllCorporateActions.yaml",
    "AccountMaintenance": EXAMPLES_DIR / "AccountMaintenance.yaml",
}

# -- Derived settings -------------------------------------------------------

target_tags: list[str] = list(TAG_EXAMPLES.keys())

# -- Runtime configuration ---------------------------------------------------

model_chat: str = "gemma-4-12b-it-qat-q4_0"
model_decision: str = "Cloudflare_clef-flash-Q4_K_M"
text_chunk_size: int = 800
min_chars_for_summarization: int = 4000

eval_max_retries: int = 5
invoke_max_retries: int = 1

default_target_levels: ClassVar[list[str]] = ["High"]

# -- Category → tag subset mapping -------------------------------------------

category_tag_config: dict[str, list[str]] | None = {
    "MarketWideImpact": [
        "OrderInfrastructureChanges",
        "AllCorporateActions",
        "AccountMaintenance",
    ],
    "SegmentAssetLevelImpact": [
        "OrderInfrastructureChanges",
        "AllCorporateActions",
        "AccountMaintenance",
    ],
    "SpecificSecurityImpact": ["AllCorporateActions"],
}
