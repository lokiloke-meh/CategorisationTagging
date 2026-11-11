from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from prompt_chains.base_chains import ChainResult, Document


tracer = trace.get_tracer(__name__)


# --- DATACLASSES ---


@dataclass
class EvalResult:
    """The output of an evaluation step."""

    eval_name: str
    eval_success: bool
    passed: bool | None = None
    score: float | None = None
    feedback: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    eval_children: list[EvalResult] = field(default_factory=list)
    eval_intermediate_results: Any = None
    error_message: str | None = None

    def __post_init__(self):
        """Automatically captures the current OpenTelemetry trace context if available."""
        span = trace.get_current_span()
        span_context = span.get_span_context()
        if span_context.is_valid:
            # Capture trace/span IDs in metadata for long-term traceability in logs/cache
            if "trace_id" not in self.metadata:
                self.metadata["trace_id"] = format(span_context.trace_id, "032x")
            if "span_id" not in self.metadata:
                self.metadata["span_id"] = format(span_context.span_id, "016x")

            # Capture session.id from span attributes if available
            if hasattr(span, "attributes"):
                session_id = span.attributes.get("session.id")
                if session_id and "session_id" not in self.metadata:
                    self.metadata["session_id"] = session_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_otel_attributes(self) -> dict[str, str | bool | float | int]:
        """Converts EvalResult to OpenTelemetry attributes.

        Large fields are saved to a separate file; the path is stored in the span.
        """
        from utils.otel_utils import record_span_details

        span = trace.get_current_span()
        data = self.to_dict()
        entry = {**data, "data_type": "EvalResult"}
        large_fields = {"eval_intermediate_results", "eval_children"}
        return record_span_details(span, "eval", entry, large_fields, "eval.")


# --- THE EVALUATOR INTERFACE ---


class BaseEvaluator(ABC):
    """Interface for all Evaluators (both deterministic and LLM-based)."""

    # Control flags for chain-level behavior
    trigger_retry: bool = False
    hard_fail: bool = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def evaluate(self, inputs: list[Document], output: ChainResult) -> EvalResult:
        """Run an evaluation inside an OpenTelemetry span every time."""
        with tracer.start_as_current_span(f"eval.{self.name}") as span:

            if not output.success:
                return EvalResult(
                    eval_name=self.name,
                    eval_success=False,
                    error_message="ChainResult is not successful, nothing to evaluate.",
                )

            try:
                result = self._evaluate(inputs=inputs, output=output)
            except Exception as e:  # noqa: BLE001
                span.record_exception(e)
                result = EvalResult(
                    eval_name=self.name, eval_success=False, error_message=repr(e)
                )

            span.set_attributes(result.to_otel_attributes())

            if not result.eval_success:
                span.set_status(Status(StatusCode.ERROR, str(result.error_message)))
            else:
                span.set_status(Status(StatusCode.OK))

            return result

    @abstractmethod
    def _evaluate(self, inputs: list[Document], output: ChainResult) -> EvalResult:
        pass
