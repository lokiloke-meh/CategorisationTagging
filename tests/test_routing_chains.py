from __future__ import annotations

from prompt_chains.base_chains import BaseChain, ChainResult, Document
from prompt_chains.long_text_summariser import WordTextSplitter


def test_text_splitter_preserves_word_count() -> None:
    """Splitting a document preserves all words (no words dropped)."""
    original = "The quick brown fox jumps over the lazy dog and runs away."
    chunks = WordTextSplitter(chunk_size=10, chunk_overlap=2).split_text(original)
    assert set(" ".join(chunks).split()) == set(original.split())


def _named_chain(name: str, payload: dict) -> BaseChain:
    """Helper that returns a chain producing a distinctive output."""

    class _Named(BaseChain):
        def _invoke(self, docs: list[Document]) -> ChainResult:
            return ChainResult(
                chain_name=name,
                input_data="",
                output_data={**payload},
                success=True,
            )

    return _Named()


def test_bool_router_routes_to_matched_destination() -> None:
    """BoolRouterChain invokes the matched route's chain and returns its output_data."""
    from prompt_chains.routing_chains_bool import (
        BoolRoute,
        BoolRouterChain,
        TextLengthCondition,
    )

    router = BoolRouterChain(
        routes=[
            BoolRoute(
                name="long",
                condition=TextLengthCondition(min_chars=1),
                destination_chain=_named_chain("long_route", {"route": "long"}),
            ),
            BoolRoute(
                name="short",
                condition=TextLengthCondition(max_chars=0),
                destination_chain=_named_chain("short_route", {"route": "short"}),
            ),
        ]
    )
    result = router.invoke([Document(page_content="hello world")])
    assert result.success is True
    assert result.output_data == {"route": "long"}


def test_bool_router_falls_back_to_default_when_no_match() -> None:
    """BoolRouterChain invokes the default chain and returns its output_data when no route matches."""
    from prompt_chains.routing_chains_bool import (
        BoolRoute,
        BoolRouterChain,
        TextLengthCondition,
    )

    router = BoolRouterChain(
        routes=[
            BoolRoute(
                name="long",
                condition=TextLengthCondition(min_chars=100),
                destination_chain=_named_chain("long_route", {"route": "long"}),
            )
        ],
        default_chain=_named_chain("default_route", {"route": "fallback"}),
    )
    result = router.invoke([Document(page_content="test")])
    assert result.success is True
    assert result.output_data == {"route": "fallback"}

    # No match + no default → returns success=False (not exception)
    router_fail = BoolRouterChain(
        routes=[
            BoolRoute(
                name="short",
                condition=TextLengthCondition(min_chars=100),
                destination_chain=_named_chain("long_route", {"route": "long"}),
            )
        ],
    )
    assert router_fail.invoke([Document(page_content="test")]).success is False


def test_categorical_router_routes_to_matched_category() -> None:
    """CategoricalRouterChain invokes the chain mapped to the matched keyword."""
    from prompt_chains.routing_chains_categorical import (
        CategoricalRouterChain,
        KeywordCategoricalCondition,
    )

    router = CategoricalRouterChain(
        condition=KeywordCategoricalCondition({"refund": "billing", "login": "auth"}),
        destination_map={
            "billing": _named_chain("billing_route", {"category": "billing"}),
            "auth": _named_chain("auth_route", {"category": "auth"}),
        },
    )
    result = router.invoke([Document(page_content="I need a refund for my order")])
    assert result.success is True
    assert result.output_data == {"category": "billing"}


def test_categorical_router_falls_back_to_default_when_no_keyword_match() -> None:
    """CategoricalRouterChain invokes the default chain when no keyword matches."""
    from prompt_chains.routing_chains_categorical import (
        CategoricalRouterChain,
        KeywordCategoricalCondition,
    )

    router = CategoricalRouterChain(
        condition=KeywordCategoricalCondition({"xyz": "matched"}),
        destination_map={"matched": _named_chain("matched_route", {"category": "matched"})},
        default_chain=_named_chain("default_route", {"category": "unknown"}),
    )
    result = router.invoke([Document(page_content="test")])
    assert result.success is True
    assert result.output_data == {"category": "unknown"}


def test_categorical_router_returns_failure_without_match_or_default() -> None:
    """CategoricalRouterChain returns success=False when no keyword matches and no default exists."""
    from prompt_chains.routing_chains_categorical import (
        CategoricalRouterChain,
        KeywordCategoricalCondition,
    )

    router = CategoricalRouterChain(
        condition=KeywordCategoricalCondition({"xyz": "matched"}),
        destination_map={"matched": _named_chain("matched_route", {"category": "matched"})},
    )
    assert router.invoke([Document(page_content="test")]).success is False
