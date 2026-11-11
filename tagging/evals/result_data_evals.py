from datetime import datetime, timedelta

from evals.base_evals import BaseEvaluator, EvalResult
from prompt_chains.base_chains import ChainResult, Document


class ResultDataValidationEvals(BaseEvaluator):

    @staticmethod
    def is_valid_date(date_string):
        try:
            datetime.strptime(date_string, "%Y-%m-%d")  # noqa: DTZ007
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def is_value_categorical(value):
        try:
            return value in ["Low", "Medium", "High"]
        except Exception:  # noqa: BLE001
            return False

    def _evaluate(self, inputs: list[Document], output: ChainResult) -> EvalResult:

        if not output.success:
            return EvalResult(
                eval_name=self.name,
                eval_success=False,
                error_message="ChainResult is not successful, nothing to evaluate.",
            )

        try:

            check_date = True
            fields_not_in_predefined_choices = []
            feedback = ""
            score = 1

            for k, v in output.output_data.tags.items():
                # check all tags that the values are between 0 and 1
                if not self.__class__.is_value_categorical(v):
                    fields_not_in_predefined_choices.append(k)

            if len(fields_not_in_predefined_choices) > 0:
                score = 0
                feedback += f"Fields {fields_not_in_predefined_choices} are not in [Low, Medium, High]. "

            effectivedate = getattr(output.output_data, "effectivedate", None)
            if effectivedate:
                # check if effectivedate is of date YYYY-MM-DD if exists
                check_date = self.__class__.is_valid_date(effectivedate)
                if check_date:
                    date_parsed = datetime.strptime(  # noqa: DTZ007
                        effectivedate, "%Y-%m-%d"
                    )
                    # check if date is too back or too far in the future
                    historical_threshold = datetime.now() - timedelta(  # noqa: DTZ005
                        days=180
                    )
                    historical_check = date_parsed >= historical_threshold

                    future_threshold = datetime.now() + timedelta(  # noqa: DTZ005
                        days=360
                    )
                    future_check = date_parsed <= future_threshold

                    if not historical_check:
                        score = min(score - 0.5, 0)
                        feedback += "effectivedate is too far back in the past, that is unlikely. "

                    if not future_check:
                        score = min(score - 0.5, 0)
                        feedback += (
                            "effectivedate is too far in the future, that is unlikely. "
                        )

                else:
                    score = min(score - 0.2, 0)
                    feedback += "effectivedate is not in YYYY-MM-DD format. "

            if score == 1:
                return EvalResult(
                    eval_name=self.name, eval_success=True, passed=True, score=1
                )
            else:
                return EvalResult(
                    eval_name=self.name,
                    eval_success=True,
                    passed=score > 0.75,
                    score=score,
                    feedback=feedback,
                )

        except Exception as e:  # noqa: BLE001
            return EvalResult(
                eval_name=self.name, eval_success=False, error_message=repr(e)
            )
