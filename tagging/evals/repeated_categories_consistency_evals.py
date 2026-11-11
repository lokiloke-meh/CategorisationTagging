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
                error_message="ChainResult is not successful, nothing to evaluate.",
            )

        try:
            vote_results = output.eval_context.vote_results
            if not vote_results:
                return EvalResult(
                    eval_name=self.name,
                    eval_success=False,
                    error_message="No vote results found in evaluation context.",
                )

            if len(vote_results) != 1:
                return EvalResult(
                    eval_name=self.name,
                    eval_success=True,
                    passed=False,
                    score=0.0,
                    feedback=f"Expected 1 tag in vote_results, got {len(vote_results)}",
                )

            from utils.category_mode import resolve_category_mode

            votes = next(iter(vote_results.values()))
            all_result_categories_the_same = len(set(votes)) == 1

            if all_result_categories_the_same:
                return EvalResult(
                    eval_name=self.name, eval_success=True, passed=True, score=1
                )

            else:
                highest_occurrence_category = resolve_category_mode(votes)
                highest_occurence_category_rate = len(
                    [x for x in votes if x == highest_occurrence_category]
                ) / len(votes)

                return EvalResult(
                    eval_name=self.name,
                    eval_success=True,
                    passed=False,
                    score=highest_occurence_category_rate,
                    feedback=f"Categories not consistent. Occurrence rate = {highest_occurence_category_rate:.2%}",
                )

        except Exception as e:  # noqa: BLE001
            return EvalResult(
                eval_name=self.name, eval_success=False, error_message=repr(e)
            )
