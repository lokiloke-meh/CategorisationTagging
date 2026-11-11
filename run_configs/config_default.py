"""Default profile — all settings visible in one file."""

from pathlib import Path
from typing import ClassVar

# -- Tag examples (source of truth for available tags) ----------------------

EXAMPLES_DIR = Path(__file__).parent.parent / "tagging" / "tagging_examples_yaml"

TAG_EXAMPLES: dict[str, str | Path] = {
    "MarketWideBoardLotSizeFrameworkChanges": EXAMPLES_DIR
    / "MarketWideBoardLotSizeFrameworkChanges.yaml",
    "SegmentWideSettlementProcedureChanges": EXAMPLES_DIR
    / "SegmentWideSettlementProcedureChanges.yaml",
    "MockDRDisasterRecoverySession": EXAMPLES_DIR
    / "MockDRDisasterRecoverySession.yaml",
    "TradingHoursAmendment": EXAMPLES_DIR / "TradingHoursAmendment.yaml",
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
        "MarketWideBoardLotSizeFrameworkChanges",
        "MockDRDisasterRecoverySession",
        "SegmentWideSettlementProcedureChanges",
        "TradingHoursAmendment",
    ],
    "SegmentAssetLevelImpact": [
        "MarketWideBoardLotSizeFrameworkChanges",
        "MockDRDisasterRecoverySession",
        "SegmentWideSettlementProcedureChanges",
        "TradingHoursAmendment",
    ],
    "SpecificSecurityImpact": [],
}
