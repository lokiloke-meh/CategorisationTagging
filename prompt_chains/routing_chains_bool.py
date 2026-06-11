from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
from opentelemetry import trace
from prompt_chains.base_chains import Document, BaseChain, ChainResult


# ==========================================
# THE ROUTING CONDITION INTERFACE
# ==========================================
class BoolRoutingCondition(ABC):
    """
    The interface for all routing rules. 
    Returns True if the Document matches the criteria, False otherwise.
    """
    @abstractmethod
    def evaluate(self, docs: list[Document]) -> bool:
        """
        Evaluates the input documents against specific criteria.

        Args:
            docs: A list of Documents to check.

        Returns:
            bool: True if criteria are met, False otherwise.
        """
        pass

# ==========================================
# ROUTING STRATEGIES
# ==========================================
class TextLengthCondition(BoolRoutingCondition):
    """
    Routes based on the total character length of the input documents.
    Often used to decide if a document is long enough to require summarization
    before further processing.
    """
    def __init__(self, min_chars: int = 0, max_chars: float = float('inf')):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def evaluate(self, docs: list[Document]) -> bool:
        total_length = sum(len(doc.page_content) for doc in docs)
        return self.min_chars <= total_length < self.max_chars

class KeywordCondition(BoolRoutingCondition):
    """
    Routes based on whether a specific keyword exists in the text.
    Case-insensitive search across all document content.
    """
    def __init__(self, keyword: str):
        self.keyword = keyword.lower()

    def evaluate(self, docs: list[Document]) -> bool:
        combined_text = " ".join(doc.page_content.lower() for doc in docs)
        return self.keyword in combined_text


# ==========================================
# ROUTE CONFIGURATION
# ==========================================
@dataclass
class BoolRoute:
    """
    A container mapping a condition to a specific chain.
    
    Attributes:
        name: A human-readable identifier for the route (used in logs).
        condition: The BoolRoutingCondition to evaluate.
        destination_chain: The BaseChain to execute if the condition is met.
    """
    name: str
    condition: BoolRoutingCondition
    destination_chain: BaseChain


# ==========================================
# ROUTER CHAIN
# ==========================================
class BoolRouterChain(BaseChain):
    """
    A branching chain that selects one execution path from multiple options.
    
    It evaluates routes top-to-bottom. The first route whose condition returns True 
    is executed. If any condition fails to evaluate correctly, the entire chain 
    fails immediately to prevent unpredictable behavior in sequential pipelines.
    """
    def __init__(self, routes: list[BoolRoute], default_chain: Optional[BaseChain] = None, custom_name: str = "BoolRouterChain"):
        super().__init__()
        self.routes = routes
        self.default_chain = default_chain
        self._custom_name = custom_name

    @property
    def name(self): 
        return self._custom_name

    def _invoke(self, docs: list[Document]) -> ChainResult:
        """
        Determines the execution path by evaluating conditions.
        
        If a condition is met, the destination chain is invoked. 
        If no condition is met, the default_chain is used (if provided).
        If errors occurred in all routes and no default_chain is provided, 
        returns a critical failure.
        """
        selected_route_name = None
        selected_chain = None
        evaluation_errors = []

        # Evaluate routes in the order they were registered
        for route in self.routes:
            try:
                condition_result = route.condition.evaluate(docs)
                
                if not isinstance(condition_result, bool):
                    error_msg = f"BoolRoutingCondition for '{route.name}' returned {type(condition_result)} instead of bool."
                    evaluation_errors.append(error_msg)
                    continue

                if condition_result:
                    selected_chain = route.destination_chain
                    selected_route_name = route.name
                    break
            
            except Exception as e:
                trace.get_current_span().record_exception(e)
                error_msg = f"Error evaluating BoolRoutingCondition for '{route.name}': {e}"
                evaluation_errors.append(error_msg)
                continue

        # Determine final chain: either the matched route or the default fallback
        if not selected_chain:
            # Check if all routes failed evaluation (critical error)
            if len(evaluation_errors) == len(self.routes) and self.routes:
                error_msg = f"Critical Routing Error: All routes failed evaluation. Errors: {'; '.join(evaluation_errors)}"
                return ChainResult(
                    chain_name=self.name,
                    input_data=f"{len(docs)} documents",
                    output_data=None,
                    success=False,
                    error_message=error_msg
                )

            if self.default_chain:
                selected_chain = self.default_chain
                selected_route_name = "default_route"
            else:
                error_msg = "No routing conditions were met, and no default_chain was provided."
                return ChainResult(
                    chain_name=self.name,
                    input_data=f"{len(docs)} documents",
                    output_data=None,
                    success=False,
                    error_message=error_msg
                )

        print(f" -> Router determined path: [{selected_route_name}] -> executing {selected_chain.name}...")

        # Execute the selected chain
        result = selected_chain.invoke(docs)

        # Wrap the result, injecting routing metadata for the logs
        result.metadata["routed_via_bool"] = selected_route_name
        
        # We wrap this in a new ChainResult so the overarching history graph 
        # clearly shows that a Router was used, and which path it took.
        return ChainResult(
            chain_name=self.name,
            input_data=f"{len(docs)} documents",
            output_data=result.output_data,
            success=result.success,
            error_message=result.error_message,
            metadata={"routed_via_bool": selected_route_name},
            history=[result]  # The executed chain's result becomes the history of the router!
        )