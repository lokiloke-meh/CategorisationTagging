import statistics
from evals.base_evals import BaseEvaluator, EvalResult
from prompt_chains.base_chains import ChainResult, Document


class RepeatedCategoriesConsistencyEvals(BaseEvaluator):

    trigger_retry: bool = False
    hard_fail: bool = False

    def _evaluate(self, inputs: list[Document], output: ChainResult) -> EvalResult:
        
        if not output.success:
            return EvalResult(
                eval_name=self.name,
                eval_success=False,
                error_message="ChainResult is not successful, nothing to evaluate."
                )

        
        try:
            # expect a dictionary with repeated keys (keys are meaningless) and values (representing repeated results)
            all_result_categories_the_same =  len(set(output.eval_context["repeated_categories"].values())) == 1

            if all_result_categories_the_same:
                return EvalResult(
                    eval_name=self.name,
                    eval_success=True,
                    passed=True,
                    score=1
                    )
            
            else:

                highest_occurence_category = statistics.mode(output.eval_context["repeated_categories"].values())
                highest_occurence_category_rate = len([x for x in output.eval_context["repeated_categories"].values() if x == highest_occurence_category]) / len(output.eval_context["repeated_categories"].values())

                return EvalResult(
                    eval_name=self.name,
                    eval_success=True,
                    passed=False,
                    score=highest_occurence_category_rate,
                    feedback=f"Not all categories returned are the same. Occurence rate = {highest_occurence_category_rate}"
                    )

        except Exception as e:
            return EvalResult(
                eval_name=self.name,
                eval_success=False,
                error_message=repr(e)
                )



