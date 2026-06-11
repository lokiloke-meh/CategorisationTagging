
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field, asdict, fields
import json
import yaml
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from utils.otel_setup import span_writer

from evals.base_evals import BaseEvaluator


tracer = trace.get_tracer(__name__)


# ==========================================
# Core Data Structures
# ==========================================
class Document:
    """
    A simple container for text content and associated metadata.
    Used as the standard input unit for all chains.
    """
    def __init__(self, page_content: str, 
                 title: str = None,
                 metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}
        self.title = title

@dataclass
class ChainResult:
    """
    The standardized output format for all Chain operations.
    
    Encapsulates the result of a chain's execution, including telemetry 
    data, execution history, and evaluation outcomes.
    
    Attributes:
        chain_name: Name of the chain that produced this result.
        input_data: A string representation of the input provided to the chain.
        output_data: The main payload produced by the chain (e.g., dict, str).
        metadata: Arbitrary key-value pairs (e.g., trace IDs, model parameters).
        history: A list of ChainResult objects from sub-chains.
        evaluations: A list of EvalResult objects from attached evaluators.
        success: Boolean flag indicating if the execution completed without errors.
        error_message: Detailed error description if success is False.
    """
    chain_name: str                      
    input_data: str                      
    output_data: Any                     
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[ChainResult] = field(default_factory=list)
    evaluations: list[Any] = field(default_factory=list)
    success: bool = True 
    error_message: Optional[str] = None
    eval_context: dict[str, Any] = field(default_factory=dict, metadata={"exclude_from_dict": True}, repr=False)

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

    def to_dict(self) -> dict:
        """Converts the dataclass (and all nested dataclasses) to a dictionary."""
        # Consistent metadata-based filtering
        excluded_fields = {f.name for f in fields(self) if f.metadata.get("exclude_from_dict")}
        full_data = asdict(self)
        data = {k: v for k, v in full_data.items() if k not in excluded_fields}

        # Reduce logging bloat for nested results by only keeping pointers to trace/span IDs.
        # This prevents recursive repetition in logs when chains are nested (e.g. SequentialChain).
        for field_name in ["history"]:
            if data.get(field_name) and isinstance(data[field_name], list):
                # Check if the list contains serialized ChainResult objects (dicts with chain_name)
                if isinstance(data[field_name][0], dict) and "chain_name" in data[field_name][0]:
                    data[field_name] = [
                        {
                            "chain_name": item.get("chain_name"),
                            "trace_id": item.get("metadata", {}).get("trace_id") if isinstance(item.get("metadata"), dict) else None,
                            "span_id": item.get("metadata", {}).get("span_id") if isinstance(item.get("metadata"), dict) else None,
                            "success": item.get("success")
                        }
                        for item in data[field_name] if isinstance(item, dict)
                    ]

        # Ensure error_messages (which might be Exception objects) are stringified
        if data.get("error_message") is not None:
            data["error_message"] = str(data["error_message"])
        return data

    def to_otel_attributes(self) -> dict[str, str | bool | float | int]:
        """
        Converts the ChainResult to OpenTelemetry attributes.
        Large fields (input, output, history) are saved to a separate file per trace_id
        with file locking for safe concurrent access, and the path is stored in the span.
        """
        # Determine unique file path using trace context for naming
        span = trace.get_current_span()
        if not span.is_recording():
            return {}

        attributes = {}
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
            **data
        }

        span_writer.write_span(base_detail_path, entry)

        attributes["chain.details_path"] = f"{base_detail_path}.{span_writer.file_format}"

        # Filter out large fields from the trace log attributes
        large_fields = {"input_data", "output_data", "history"}
        for field_name, value in data.items():
            if value is None or field_name in large_fields:
                continue
            
            key = f"chain.{field_name}"
            if isinstance(value, (str, bool, int, float)):
                attributes[key] = value
            else:
                attributes[key] = json.dumps(value, default=str, ensure_ascii=False)
        return attributes

# ==========================================
# Base Class
# ==========================================
class BaseChain(ABC):
    """
    The abstract base class for all prompt chains.
    
    Implements a template method pattern where the public `invoke` method handles 
    orchestration (tracing, evaluation, retries) while subclasses implement 
    the specific logic in _invoke.
    """
    def __init__(self, evaluators: list[BaseEvaluator] = None, 
                 include_parent_eval: bool = False,
                 eval_max_retries: int = 0,
                 invoke_max_retries: int = 0):
        """
        Initializes the chain with optional evaluators and retry settings.
        
        Args:
            evaluators: A list of BaseEvaluator objects to run against the result.
            include_parent_eval: Whether to store the parent evaluator's result 
                when it produces child results.
            eval_max_retries: Number of additional attempts to run _invoke if 
                evaluations fail.
            invoke_max_retries: Number of additional attempts to run _invoke if 
                the initial execution returns success=False.
        """
        self.evaluators = evaluators or []
        self.include_parent_eval = include_parent_eval
        self.eval_max_retries = eval_max_retries
        self.invoke_max_retries = invoke_max_retries

    @property
    def name(self): 
        return self.__class__.__name__

    def invoke(self, docs: list[Document]) -> ChainResult:
        """
        The primary entry point for executing a chain.
        
        Manages the execution lifecycle including tracing, evaluator execution, 
        and retry logic based on evaluator feedback.

        Args:
            docs: A list of Document objects to process.

        Returns:
            ChainResult: The finalized output of the chain execution.
        """
        with tracer.start_as_current_span(f"chain.{self.name}") as span:
            result = None
            # Outer loop handles retries for non-successful execution results
            for invoke_attempt in range(self.invoke_max_retries + 1):
                
                # Inner loop handles retries triggered by evaluations
                for eval_attempt in range(self.eval_max_retries + 1):
                    # Run the actual business logic of the chain
                    result = self._invoke(docs)

                    # If the core execution failed, stop the eval loop to check for invoke retries
                    if not result.success:
                        break

                    # Determine if we should attempt a retry based on evaluations
                    should_retry_eval = self._run_evaluations_and_check_retry(docs, result)
                    
                    if not should_retry_eval or eval_attempt >= self.eval_max_retries:
                        break
                    
                    print(f"   [Evaluation retry triggered for {self.name} (attempt {eval_attempt + 1}/{self.eval_max_retries})]")
                
                # Exit outer loop if successful or we've exhausted invoke retries
                if (result and result.success) or invoke_attempt >= self.invoke_max_retries:
                    break

                print(f"   [Execution retry triggered for {self.name} (attempt {invoke_attempt + 1}/{self.invoke_max_retries}) due to success=False.]")

            # Apply the standardized attributes to the span
            span.set_attributes(result.to_otel_attributes())
            if not result.success:
                span.set_status(Status(StatusCode.ERROR, str(result.error_message)))
            else:
                span.set_status(Status(StatusCode.OK))

            return result

    def _run_evaluations_and_check_retry(self, docs: list[Document], result: ChainResult) -> bool:
        """
        Orchestrates evaluator execution and determines if a retry is warranted.
        Returns True if a retry is triggered, False otherwise.
        """
        if not (result.success and self.evaluators):
            return False

        print(f"   [Evaluating {self.name} with {len(self.evaluators)} evaluators...]")
        retry_triggered = False

        for evaluator in self.evaluators:
            eval_res = evaluator.evaluate(inputs=docs, output=result)
            
            # Flatten parent/child evaluations into a processing list
            has_children = bool(eval_res.eval_children)
            evals = ([eval_res] if (not has_children or self.include_parent_eval) else []) + (eval_res.eval_children or [])

            for res in evals:
                result.evaluations.append(res)
                eval_label = res.eval_name if has_children else evaluator.name

                if not res.eval_success:
                    print(f"Warning: Evaluation Error [{eval_label}] - {res.error_message}")

                if not res.passed:
                    print(f"Warning: Evaluation Failed [{eval_label}] - {res.feedback}")
                    
                    if getattr(evaluator, 'trigger_retry', True):
                        retry_triggered = True

                    # Hard Fail: Stop all further evaluations and force an immediate retry
                    if getattr(evaluator, 'hard_fail', False):
                        result.success = False
                        result.error_message = result.error_message or f"Critical Evaluation Failure: {eval_label}"
                        # Returning True immediately satisfies the "do not wait" requirement
                        return True 
        
        return retry_triggered

    @abstractmethod
    def _invoke(self, docs: list[Document]) -> ChainResult:
        """
        Implements the core logic of the chain.
        
        Subclasses must override this method. It is called by `invoke`.
        
        Args:
            docs: Input documents.
        Returns:
            ChainResult: The result of the internal chain logic.
        """
        pass

# ==========================================
# Sequential Chains
# ==========================================
class SequentialChain(BaseChain):
    """
    A pipeline that executes a series of chains in sequence.
    
    The output of each chain becomes the input for the next. If any chain 
    in the sequence fails, the entire SequentialChain aborts (short-circuiting).
    
    Args:
        chains: An ordered list of BaseChain instances to execute.
    """
    def __init__(self, chains: list[BaseChain]):
        super().__init__()
        self.chains = chains

    def _invoke(self, docs: list[Document]) -> ChainResult:
        chain_name = self.__class__.__name__
        current_docs = docs
        step_history = []
        final_text = ""
        initial_input_preview = f"{len(docs)} initial documents."
        
        for i, chain in enumerate(self.chains):
            inner_chain_name = getattr(chain, 'name', chain.__class__.__name__)
            print(f"[{i+1}/{len(self.chains)}] Executing {inner_chain_name}...")
            
            # Execute the chain
            result = chain.invoke(current_docs)
            step_history.append(result)
            
            # THE SHORT-CIRCUIT:
            # If the chain failed, abort the pipeline immediately!
            if not result.success:
                print(f"Pipeline Aborted at {inner_chain_name}: {result.error_message}")
                return ChainResult(
                    chain_name=chain_name,
                    input_data=initial_input_preview,
                    output_data=None,
                    success=False,
                    error_message=f"Aborted at {inner_chain_name}: {result.error_message}",
                    history=step_history
                )
            
            # If successful, prepare the output for the next chain
            final_text = result.output_data
            current_docs = [Document(page_content=final_text)]
            
        return ChainResult(
            chain_name=chain_name,
            input_data=initial_input_preview,
            output_data=final_text,
            history=step_history
        )
    
