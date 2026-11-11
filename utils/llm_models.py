from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMResult:
    """Standardized output from any LLM adapter."""

    model_name: str
    user_input: str
    model_result: dict | str
    model_options: dict[str, Any] | None = None
    system_prompt: str | None = None
    # Using metadata to flag fields that should be excluded from standard serialization
    model_thought: str | None = None
    model_load_duration_secs: float | None = None
    prompt_token_count: int | None = None
    completion_token_count: int | None = None
    total_token_count: int | None = None
    token_gen_duration_secs: float | None = None
    prompt_eval_duration_secs: float | None = None
    response_tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None
    success: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Converts the dataclass to a dictionary suitable for JSON serialization."""
        excluded_fields = {
            f.name for f in fields(self) if f.metadata.get("exclude_from_dict")
        }
        full_data = asdict(self)
        return {k: v for k, v in full_data.items() if k not in excluded_fields}

    def to_otel_attributes(self) -> dict[str, str | bool | float | int]:
        """Converts LLMResult to OpenTelemetry attributes.

        Large fields are saved to a separate file; the path is stored in the span.
        """
        from opentelemetry.trace import get_current_span

        from utils.otel_utils import record_span_details

        span = get_current_span()
        entry = {**self.to_dict(), "data_type": "LLMResult"}
        large_fields = {
            "model_result",
            "model_thought",
            "user_input",
            "system_prompt",
            "options",
        }
        return record_span_details(span, "llm", entry, large_fields, "llm.")


# ── Payload Adapter (normalizes caller payload → backend wire format) ───


class ChatPayloadAdapter(ABC):
    """Translates a unified caller payload into a backend-specific wire format.

    Callers always pass ``{"format": schema}`` for JSON-mode requests.
    Each concrete adapter translates this to its native wire format:
      - Ollama:  passes through unchanged
      - llama.cpp: wraps as ``{"response_format": {"type": "json_object", "schema": ...}}``
    """

    @abstractmethod
    def normalize(self, payload: dict) -> dict:
        """Return a backend-specific payload ready for the HTTP call."""


class OllamaPayloadAdapter(ChatPayloadAdapter):
    """Ollama wire format — ``format`` key with JSON schema directly."""

    def normalize(self, payload: dict) -> dict:
        return payload


class LlamaCppPayloadAdapter(ChatPayloadAdapter):
    """llama.cpp (OpenAI-compatible) wire format.

    Translates the common ``{"format": schema}`` into
    ``{"response_format": {"type": "json_object", "schema": ...}}``.
    """

    def normalize(self, payload: dict) -> dict:
        if "format" in payload and "response_format" not in payload:
            payload = dict(payload)  # shallow copy — don't mutate caller's dict
            schema = payload.pop("format")
            payload["response_format"] = {"type": "json_object", "schema": schema}
        return payload


# ── Different LLM Model types ───


@runtime_checkable
class ChatAdapter(Protocol):
    """Structural type for any chat-capable LLM adapter."""

    def chat_with_expected_type_result(
        self, chat_payload: dict, result_type: str
    ) -> LLMResult: ...


@runtime_checkable
class DecisionModel(Protocol):
    """Structural type for a structured decision-making LLM adapter.

    Accepts state + questions and returns structured answers with
    probabilities and confidence scores.
    """

    def make_decision(self, state: str, questions: dict) -> LLMResult: ...


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Structural type for any embedding-capable LLM adapter."""

    def get_embeddings(self, text: str) -> LLMResult: ...
