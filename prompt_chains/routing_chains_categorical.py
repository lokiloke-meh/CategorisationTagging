from abc import ABC, abstractmethod

from opentelemetry import trace

from prompt_chains.base_chains import BaseChain, ChainResult, Document


# ==========================================
# THE CATEGORICAL CONDITION INTERFACE
# ==========================================
class CategoricalCondition(ABC):
    """
    Evaluates the input documents and returns a string
    representing the category/intent of the text.
    """

    @abstractmethod
    def evaluate(self, docs: list[Document]) -> str:
        """
        Evaluates the input documents and returns a classification string.

        Args:
            docs: A list of Documents to categorize.

        Returns:
            str: The category label used for routing.
        """


# ==========================================
# CONCRETE CATEGORICAL CLASSIFIERS
# ==========================================
class KeywordCategoricalCondition(CategoricalCondition):
    """
    A fast, deterministic router based on simple keyword mapping.
    """

    def __init__(self, keyword_map: dict[str, str]):
        # e.g., {"refund": "billing", "password": "tech_support"}
        self.keyword_map = keyword_map

    def evaluate(self, docs: list[Document]) -> str:
        """
        Performs a case-insensitive keyword search across document content.

        Args:
            docs: A list of Documents to evaluate.

        Returns:
            str: The mapped category or 'UNKNOWN_ALIEN' if no keywords match.
        """
        combined_text = " ".join([d.page_content.lower() for d in docs])

        for keyword, category in self.keyword_map.items():
            if keyword.lower() in combined_text:
                return category

        return "UNKNOWN_ALIEN"


# ==========================================
# THE CATEGORICAL ROUTER CHAIN
# ==========================================
class CategoricalRouterChain(BaseChain):
    """
    Evaluates the text to get a category string, then uses a hash map
    to instantly route to the corresponding Chain.
    """

    def __init__(
        self,
        condition: CategoricalCondition,
        destination_map: dict[str, BaseChain],
        default_chain: BaseChain | None = None,
        custom_name: str = "CategoricalRouter",
        invoke_max_retries: int = 0,
    ):
        super().__init__(invoke_max_retries=invoke_max_retries)
        self.condition = condition
        self.destination_map = destination_map
        self.default_chain = default_chain
        self._custom_name = custom_name

    @property
    def name(self):
        return self._custom_name

    def _invoke(self, docs: list[Document]) -> ChainResult:
        """
        Determines the routing path by evaluating the input text against
        categorical conditions and executing the corresponding destination chain.
        """
        # Determine the category string
        try:
            category = self.condition.evaluate(docs)

            # Ensure the classification result is a string for reliable dictionary lookup
            if not isinstance(category, str):
                error_msg = f"CategoricalCondition returned {type(category)} instead of expected str."
                return ChainResult(
                    chain_name=self.name,
                    input_data=f"{len(docs)} documents",
                    output_data=None,
                    success=False,
                    error_message=error_msg,
                )

        except Exception as e:
            # Record the exception in the current OpenTelemetry span for observability
            trace.get_current_span().record_exception(e)
            return ChainResult(
                chain_name=self.name,
                input_data=f"{len(docs)} documents",
                output_data=None,
                success=False,
                error_message=f"Critical failure during categorical evaluation: {e}",
            )

        # Map category to destination chain using dict.get()
        selected_chain = self.destination_map.get(category)

        if not selected_chain:
            if self.default_chain:
                selected_chain = self.default_chain
                category = "default_category"
            else:
                return ChainResult(
                    chain_name=self.name,
                    input_data=f"{len(docs)} documents",
                    output_data=None,
                    success=False,
                    error_message=f"Category '{category}' had no mapped chain and no default_chain was provided.",
                )

        print(
            f" -> Classifier determined category: [{category}] -> executing {selected_chain.name}..."
        )

        # Execute the selected chain
        # (Notice we call .invoke() so the child chain's own evaluators still run!)
        result = selected_chain.invoke(docs)

        # Wrap and return the result with routing metadata
        result.metadata["routed_via_category"] = category

        return ChainResult(
            chain_name=self.name,
            input_data=f"{len(docs)} documents",
            output_data=result.output_data,
            success=result.success,
            error_message=result.error_message,
            metadata={"routed_via_category": category},
            history=[result],
        )
