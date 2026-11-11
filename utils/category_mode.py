"""Deterministic category mode resolution.

Replaces statistics.mode() which crashes on ties (e.g., [High, Medium]).
Tie-breaking preference: High > Medium > Low.
"""

from __future__ import annotations

from collections import Counter

_CATEGORY_ORDER = {"High": 3, "Medium": 2, "Low": 1}


def resolve_category_mode(values: list[str]) -> str:
    """Return the most frequent category, breaking ties by preference (High > Medium > Low).

    Args:
        values: Category strings from LLM responses (e.g., ["High", "Medium", "High"]).

    Returns:
        The winning category string. Raises ValueError on empty input.
    """
    if not values:
        raise ValueError("Cannot resolve mode of empty list")

    counts = Counter(values)
    max_count = max(counts.values())
    candidates = [cat for cat, count in counts.items() if count == max_count]

    # Tie-break by preference order; fall back to first encountered
    return max(candidates, key=lambda c: _CATEGORY_ORDER.get(c, 0))
