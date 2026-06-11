from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, TYPE_CHECKING
import json
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from utils.otel_setup import span_writer

if TYPE_CHECKING:
    from prompt_chains.base_chains import ChainResult, Document


tracer = trace.get_tracer(__name__)


# --- DATACLASSES ---

@dataclass
class EvalResult:
    """The output of an evaluation step."""
    eval_name: str
    eval_success: bool
    passed: Optional[bool] = None
    score: Optional[float] = None
    feedback: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    eval_children: list[EvalResult] = field(default_factory=list)
    eval_intermediate_results: Any = None
    error_message: Optional[str] = None 

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
        attributes = {}

        # Determine unique file path using trace context for naming
        span = trace.get_current_span()
        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, "032x") if span_context.is_valid else "untraced"
        span_id = format(span_context.span_id, "016x") if span_context.is_valid else "untraced"

        # Retrieve the session directory from span attributes if available
        session_dir = "logs"
        if hasattr(span, "attributes"):
            session_dir = span.attributes.get("session.dir_path", "logs")

        output_dir = os.path.join(session_dir, "llm_details")
        os.makedirs(output_dir, exist_ok=True)
        base_detail_path = os.path.join(output_dir, trace_id, span_id)
        
        data = self.to_dict()
        entry = {
            "trace_id": trace_id,
            "span_id": span_id,
            "data_type": "EvalResult",
            **data
        }

        span_writer.write_span(base_detail_path, entry)

        attributes["eval.details_path"] = f"{base_detail_path}.{span_writer.file_format}"

        # Filter out large fields from the trace log attributes
        large_fields = {"eval_intermediate_results", "eval_children"}
        for field_name, value in data.items():
            if value is None:
                continue
            
            if field_name in large_fields:
                continue

            key = f"eval.{field_name}"
            if isinstance(value, (str, bool, int, float)):
                attributes[key] = value
            else:
                attributes[key] = json.dumps(value, default=str, ensure_ascii=False)
        return attributes



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
                    error_message="ChainResult is not successful, nothing to evaluate."
                    )

            try:
                result = self._evaluate(inputs=inputs, output=output)
            except Exception as e:
                span.record_exception(e)
                result = EvalResult(
                    eval_name=self.name,
                    eval_success=False,
                    error_message=repr(e)
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
        
