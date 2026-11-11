from __future__ import annotations

import pytest

from utils.category_mode import resolve_category_mode


def test_resolve_mode_normal_and_ties() -> None:
    """Mode resolution with no-tie, two-way ties, and three-way tie."""
    # No tie — majority wins
    assert resolve_category_mode(["High", "High", "Medium"]) == "High"

    # Two-way ties → higher preference wins
    assert resolve_category_mode(["High", "Medium"]) == "High"
    assert resolve_category_mode(["Medium", "Low"]) == "Medium"

    # Three-way tie → highest preference wins
    assert resolve_category_mode(["High", "Medium", "Low"]) == "High"


def test_resolve_mode_empty_raises() -> None:
    """Empty input raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        resolve_category_mode([])
