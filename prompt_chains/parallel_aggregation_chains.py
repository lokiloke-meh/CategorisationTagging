from prompt_chains.base_chains import Document, BaseChain, ChainResult
from evals.base_evals import BaseEvaluator
import json


class ParallelDictAggregationChain(BaseChain):
    """
    Executes multiple chains in parallel. 
    Verifies that every child chain outputs a Python dictionary.
    Combines all outputs into a single master dictionary.
    """
    def __init__(self, 
                 chains: list[BaseChain], 
                 evaluators: list[BaseEvaluator] = None,
                 custom_name: str = "ParallelDictAggregator",
                 eval_max_retries: int = 0,
                 invoke_max_retries: int = 0):
        
        super().__init__(evaluators=evaluators,
                         eval_max_retries=eval_max_retries,
                         invoke_max_retries=invoke_max_retries)
        self.chains = chains
        self._custom_name = custom_name
    

    def _invoke(self, docs: list[Document]) -> ChainResult:

        if not self.chains:
            raise ValueError(f"{self.name} requires at least one chain.")
        
        step_history = []
        master_dict = {}
        all_metadata = {}
        
        print(f"\n[⚡ Starting {self.name} with {len(self.chains)} dict branches]")

        for i, chain in enumerate(self.chains):
            print(f" ├── Executing branch {i+1}: {chain.name}...")
            
            # Execute the chain
            result = chain.invoke(docs)
            step_history.append(result)
            
            if not result.success:
                print(f" ❌ Aborted. '{chain.name}' failed: {result.error_message}")
                return ChainResult(
                    chain_name=self.name, 
                    input_data=f"{len(docs)} docs", 
                    output_data=None,
                    success=False, 
                    error_message=f"Branch '{chain.name}' failed.", 
                    history=step_history
                )
            
            # EXTRACT OR PARSE THE DICTIONARY
            chain_dict = None
            
            # Expect a dictionary
            if isinstance(result.output_data, dict):
                chain_dict = result.output_data
                
            
            # Verify it is a dictionary
            if not isinstance(chain_dict, dict):
                return ChainResult(
                    chain_name=self.name, 
                    input_data=f"{len(docs)} docs", 
                    output_data=None,
                    success=False, 
                    error_message=f"'{chain.name}' returned a {type(chain_dict).__name__}, not a dict.", 
                    history=step_history
                )

            # Combine the dictionaries but overwrite if keys are the same
            master_dict.update(chain_dict)

            all_metadata[chain.name] = result.metadata


        print(f"[✅ {self.name} completed successfully]\n")
        
        return ChainResult(
            chain_name=self.name,
            input_data=f"{len(docs)} documents",
            output_data=master_dict,
            success=True,
            metadata=all_metadata,
            history=step_history
        )
