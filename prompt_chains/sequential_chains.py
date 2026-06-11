from prompt_chains.base_chains import Document, BaseChain, ChainResult

# ==========================================
# Sequential Chains
# ==========================================
class SequentialChain(BaseChain):
    """
    Takes a list of chains. Passes the initial Documents to the first chain.
    Takes the text output of the first chain, packages it into a new Document,
    and passes it to the next chain, and so on.
    """
    def __init__(self, chains: list[BaseChain],
                 invoke_max_retries: int = 0):
        super().__init__(invoke_max_retries=invoke_max_retries)
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
            history=step_history,
            error_message=result.error_message
        )
    
