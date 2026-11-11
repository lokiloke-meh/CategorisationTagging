from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvaluationContext:
    """Typed evaluation context passed from chains to evaluators.

    Used by all chain types. Fields are populated selectively depending on the adapter:
      - ChatAdapter:   categories + model_thought + vote_results (answers is empty)
      - DecisionModel: categories + answers     (model_thought and vote_results are empty)
    """

    # Populated by both adapters. Maps tag names to their final category choice.
    # Example: {"IPAddressChangesbytheExchange": "High", "MarketWideImpact": "Low"}
    categories: dict[str, str] = field(default_factory=dict)
    # ChatAdapter only. The model's reasoning text from its chain-of-thought output.
    # DecisionModel does not produce this field. Example: "The exchange is updating its IP addresses to improve redundancy..."
    model_thought: str | None = None
    # ChatAdapter only. Maps tag names to their repeated vote values (e.g., Relevance, Relevance1-4).
    # DecisionModel does not produce this field. Example: {"IPAddressChangesbytheExchange": ["High", "High", "Medium", "High", "High"]}
    vote_results: dict[str, list[str]] = field(default_factory=dict)
    # DecisionModel only. Maps question names to their full answer objects containing choice, probabilities, and confidence.
    # ChatAdapter does not populate this field. Example: {"relevance": {"type": "choice", "choice": "High", "probabilities": {"Low": 0.13, "Medium": 0.22, "High": 0.65}, "confidence": 0.48}}
    answers: dict[str, dict] = field(default_factory=dict)


@dataclass
class TagResult:
    """
    Example:

    tags={"MarketWideImpact": "High", "IPAddressChangesbytheExchange": "Low"}
    summary="Exchange to update IP addresses by 2026-03-15"
    effectivedate="2026-03-15"
    """

    tags: dict[str, str]
    summary: str | None = None
    effectivedate: str | None = field(default=None)

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def keys(self):
        return self.__dict__.keys()
