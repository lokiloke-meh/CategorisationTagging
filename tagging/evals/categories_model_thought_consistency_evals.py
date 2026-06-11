import re
from collections import Counter
from evals.base_evals import BaseEvaluator, EvalResult
from prompt_chains.base_chains import ChainResult, Document


class CategoriesModelThoughtConsistencyEvals(BaseEvaluator):
    def __init__(self, categories: list[str]):
        super().__init__()
        self.categories = categories

    trigger_retry: bool = True
    hard_fail: bool = True

    def _evaluate(self, inputs: list[Document], output: ChainResult) -> EvalResult:
        
        if not output.success:
            return EvalResult(
                eval_name=self.name,
                eval_success=False,
                error_message="ChainResult is not successful, nothing to evaluate."
                )

        
        try:
            
            model_thought = output.eval_context.get("model_thought")
            single_result_category = output.eval_context.get("single_result_category")


            if not model_thought:
                return EvalResult(
                    eval_name=self.name,
                    eval_success=True,
                    passed=True,
                    score=1,
                    feedback="No model_thought found in eval_context."
                    )
            
            if not single_result_category:
                return EvalResult(
                    eval_name=self.name,
                    eval_success=True,
                    passed=False,
                    score=0,
                    feedback="No single_result_category found in eval_context."
                    )


            if model_thought:
                # Split text into sentences using basic punctuation rules (. ! ?)
                # Extract only the last few sentences
                # Join them back together into one lowercased block for searching
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', model_thought) if s.strip()]
                search_block = " ".join(sentences[-10:]).lower()

                # Map lowercase keys to their original preserved casing
                # Example: {"low": "Low", "medium": "Medium", "high": "HIGH"}
                keyword_map = {k.lower(): k for k in self.categories}

                # If any conclusion markers appear, we prioritize the text following the latest one that precedes a keyword
                conclusion_markers = ["conclusion", "final", "therefore"]
                marker_positions = []
                for marker in conclusion_markers:
                    start = 0
                    while True:
                        idx = search_block.find(marker, start)
                        if idx == -1:
                            break
                        marker_positions.append((idx, len(marker)))
                        start = idx + 1

                # Check markers starting from the end of the text
                marker_positions.sort(key=lambda x: x[0], reverse=True)
                for idx, m_len in marker_positions:
                    sub_block = search_block[idx + m_len:]
                    if any(re.search(r'\b' + re.escape(k) + r'\b', sub_block) for k in keyword_map.keys()):
                        search_block = sub_block
                        break

                found_words = []
                for keyword_lowercase in keyword_map.keys():
                    matches = re.findall(r'\b' + re.escape(keyword_lowercase) + r'\b', search_block)
                    found_words.extend(matches)

                if not found_words:
                    return EvalResult(
                        eval_name=self.name,
                        eval_success=True,
                        passed=False,
                        score=0,
                        feedback="Category was not found in the last few sentences of model_thought"
                        )
                
                # Count frequencies
                # Handle ties by checking which keyword appears closest to the end of the text
                counts = Counter(found_words)
                most_common = counts.most_common()
                highest_freq = most_common[0][1]
                top_candidates = [word for word, freq in most_common if freq == highest_freq]

                if len(top_candidates) > 1:
                    winner_lowercase = max(top_candidates, key=lambda w: search_block.rfind(w))
                    most_likely_category = keyword_map[winner_lowercase]
                else:
                    most_likely_category = keyword_map[top_candidates[0]]
                

                if most_likely_category == single_result_category:
                    return EvalResult(
                        eval_name=self.name,
                        eval_success=True,
                        passed=True,
                        score=1
                        )
                else:
                    return EvalResult(
                        eval_name=self.name,
                        eval_success=True,
                        passed=False,
                        score=0,
                        feedback="model_thought not consistent with result category."
                        )

        except Exception as e:
            return EvalResult(
                eval_name=self.name,
                eval_success=False,
                error_message=repr(e)
                )
