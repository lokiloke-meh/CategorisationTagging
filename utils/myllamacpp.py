import json

import requests
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from utils.llm_models import ChatPayloadAdapter, LlamaCppPayloadAdapter, LLMResult

tracer = trace.get_tracer(__name__)


def _extract_llamacpp_chat_metrics(model_response: dict) -> dict:
    """Extracts usage and timing metrics from OpenAI-compatible chat completions."""
    usage = model_response.get("usage", {})
    timings = model_response.get("timings", {})

    metrics: dict[str, float | int] = {}

    if (prompt_token_value := usage.get("prompt_tokens")) is not None:
        metrics["prompt_token_count"] = prompt_token_value

    if (completion_token_value := usage.get("completion_tokens")) is not None:
        metrics["completion_token_count"] = completion_token_value

    if (total_token_value := usage.get("total_tokens")) is not None:
        metrics["total_token_count"] = total_token_value

    if (predicted_per_second_value := timings.get("predicted_per_second")) is not None:
        metrics["response_tokens_per_second"] = float(predicted_per_second_value)

    if (prompt_ms := timings.get("prompt_ms")) is not None:
        metrics["prompt_eval_duration_secs"] = prompt_ms / 1000

    if (predicted_ms := timings.get("predicted_ms")) is not None:
        metrics["token_gen_duration_secs"] = predicted_ms / 1000

    if (prompt_tokens_per_second_value := timings.get("prompt_per_second")) is not None:
        metrics["prompt_tokens_per_second"] = float(prompt_tokens_per_second_value)

    return metrics


class llamaCppChat:
    """OpenAI-compatible chat adapter for llama.cpp servers.

    Drop-in replacement for ollamaChat — same constructor signature and method
    name. Swap one class for the other without touching calling code.

    Example:
        model = llamaCppChat(model="gemma-4-12b-it-qat-q4_0")
        result = model.chat_with_expected_type_result(
            chat_payload={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Explain quantum computing."},
                ],
            },
            result_type="string",
        )
    """

    def __init__(self, model: str = "gemma-4-12b-it-qat-q4_0"):
        self.model = model
        self.url = "http://localhost:8080/v1/chat/completions"
        self._payload_adapter: ChatPayloadAdapter = LlamaCppPayloadAdapter()

    def chat_with_expected_type_result(
        self, chat_payload: dict, result_type: str
    ) -> LLMResult:
        # Normalize payload via adapter (caller passes common format)
        chat_payload = self._payload_adapter.normalize(chat_payload)
        chat_payload["model"] = self.model

        if result_type == "string":
            pass
        elif result_type == "dict":
            if not chat_payload.get("response_format"):
                raise OSError(
                    "Expected to pass response_format if we want JSON results"
                )
        else:
            raise ValueError(f"Not supported result_type {result_type}")

        # Extract context for logging
        system_prompt = ""
        user_input = ""
        for msg in chat_payload.get("messages", []):
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break

        for msg in reversed(chat_payload.get("messages", [])):
            if msg.get("role") == "user":
                user_input = msg.get("content", "")
                break

        model_options = {k: v for k, v in chat_payload.items() if k != "messages"}

        base_result_data = {
            "model_name": self.model,
            "user_input": user_input,
            "system_prompt": system_prompt,
            "model_options": model_options,
        }

        with tracer.start_as_current_span(f"llamacpp.chat.{result_type}") as span:
            span.set_attribute("llm.request.model", self.model)

            try:
                response = requests.post(self.url, json=chat_payload)
                if response.status_code == 200:
                    model_response = json.loads(response._content.decode("utf-8"))

                    return_result = base_result_data.copy()
                    return_result.update(_extract_llamacpp_chat_metrics(model_response))

                    choice = model_response.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    content = message.get("content", "")

                    # Capture reasoning/thinking content if available
                    reasoning_content = message.get("reasoning_content")
                    if reasoning_content:
                        return_result["model_thought"] = reasoning_content

                    if result_type == "dict":
                        try:
                            return_result["model_result"] = json.loads(content)
                        except json.JSONDecodeError:
                            error_message = (
                                f"Failed to decode JSON. Full model reply: {message}"
                            )
                            print(error_message)
                            return LLMResult(
                                **return_result,
                                success=False,
                                error_message=error_message,
                                model_result="",
                            )
                    else:
                        return_result["model_result"] = content

                    result = LLMResult(**return_result)
                else:
                    result = LLMResult(
                        **base_result_data,
                        model_result={},
                        success=False,
                        error_message=f"HTTPError {response.status_code}",
                    )

            except Exception as e:  # noqa: BLE001
                span.record_exception(e)
                result = LLMResult(
                    **base_result_data,
                    model_result={},
                    success=False,
                    error_message=repr(e),
                )

            span.set_attributes(result.to_otel_attributes())

            if not result.success:
                span.set_status(Status(StatusCode.ERROR, str(result.error_message)))
            else:
                span.set_status(Status(StatusCode.OK))

            return result


def _extract_llamacpp_decision_metrics(model_response: dict) -> dict:
    """Extracts usage metrics from llama.cpp decision endpoint responses."""
    usage = model_response.get("usage", {})
    metrics = {}

    if usage.get("input_tokens"):
        metrics["prompt_token_count"] = usage["input_tokens"]

    return metrics


class llamacppDecisionModel:
    """Structured decision-making LLM adapter for llama.cpp endpoints.

    Accepts multiple questions in a single call, each returning structured output
    with confidence scores and probability distributions. For single-question
    generative output or thinking-mode reasoning, use llamaCppChat instead.

    Question types:
        choice — pick one option from criteria keys. Returns {"choice": str, "probabilities": {...}, "confidence": float}.
        noul   — raw numeric value between 0 and 1 (e.g. rate urgency on a scale). Returns {"noul": float} only.
        score  — ordinal/ranked scoring with mapped legend. Returns {"score": float, "legend": {int: str}, "probabilities": {...}, "confidence": float}.

    Example:
        model = llamacppDecisionModel()
        result = model.make_decision(
            state="HKEX announced trading hours extension by 30 minutes starting April 2025. Settlement for FX trades changes to T+1.",
            questions={
                "trading_hours": {
                    "type": "choice",
                    "instructions": "Categorize relevance as Low, Medium, or High for trading hours changes",
                    "criteria": {"Low": "Not relevant at all", "Medium": "Some minor impact", "High": "Significant system change required"}
                },
                "urgency_noul": {
                    "type": "noul",
                    "instructions": "Rate how urgent this change is on a scale of 0 (not urgent) to 1 (critical)"
                },
                "urgency_score": {
                    "type": "score",
                    "instructions": "How urgent is this change?",
                    "criteria": ["can wait", "this week", "today", "right now"]
                }
            },
        )
        result.model_result == {
            "trading_hours": {"type": "choice", "choice": "Medium", "probabilities": {"High": 0.13, "Low": 0.22, "Medium": 0.65}, "confidence": 0.48},
            "urgency_noul": {"type": "noul", "noul": 0.72},
            "urgency_score": {"type": "score", "score": 0.19, "legend": {"0": "can wait", "1": "this week", "2": "today", "3": "right now"}, "probabilities": {"0": 0.86, ...}}
        }
    """

    def __init__(
        self,
        url: str = "http://localhost:8080/v1/systemone",
        model: str = "Cloudflare_clef-flash-Q4_K_M",
    ):
        self.url = url
        self.model = model

    def make_decision(self, state: str, questions: dict) -> LLMResult:
        """Make a structured decision using the llama.cpp decision endpoint."""

        request_payload = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }

        result_base_data = {
            "model_name": self.model,
            "user_input": state,
            "system_prompt": questions,
            "model_options": None,
        }

        with tracer.start_as_current_span("llamacpp.decision") as span:
            span.set_attribute("llm.request.model", self.model)

            try:
                response = requests.post(self.url, json=request_payload)
                if response.status_code == 200:
                    api_response = json.loads(response._content.decode("utf-8"))

                    extracted_metrics = result_base_data.copy()
                    extracted_metrics.update(
                        _extract_llamacpp_decision_metrics(api_response)
                    )
                    extracted_metrics["model_result"] = api_response.get("answers", {})

                    result = LLMResult(**extracted_metrics)
                else:
                    result = LLMResult(
                        **result_base_data,
                        model_result={},
                        success=False,
                        error_message=f"HTTPError {response.status_code}",
                    )

            except Exception as e:  # noqa: BLE001
                span.record_exception(e)
                result = LLMResult(
                    **result_base_data,
                    model_result={},
                    success=False,
                    error_message=repr(e),
                )

            span.set_attributes(result.to_otel_attributes())

            if not result.success:
                span.set_status(Status(StatusCode.ERROR, str(result.error_message)))
            else:
                span.set_status(Status(StatusCode.OK))

            return result


class llamaCppEmbeddings:
    """Embedding adapter for llama.cpp embedding endpoints.

    Example:
        model = llamaCppEmbeddings(model="embeddinggemma-2-BF16")
        result = model.get_embeddings(text="search query")
    """

    def __init__(self, url: str = "http://127.0.0.1:8082/v1/embeddings", model: str = "embeddinggemma-2-BF16"):
        self.url = url
        self.model = model

    def get_embeddings(self, text: str) -> LLMResult:
        request_payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float",
        }

        with tracer.start_as_current_span("llamacpp.embed") as span:
            span.set_attribute("llm.request.model", self.model)

            try:
                response = requests.post(self.url, json=request_payload)
                if response.status_code == 200:
                    api_response = json.loads(response._content.decode("utf-8"))

                    result = LLMResult(
                        model_name=self.model,
                        user_input=text,
                        model_result=api_response["data"][0]["embedding"],
                    )
                else:
                    result = LLMResult(
                        model_name=self.model,
                        user_input=text,
                        model_result=[],
                        success=False,
                        error_message=f"HTTPError {response.status_code}",
                    )

            except Exception as e:  # noqa: BLE001
                span.record_exception(e)
                result = LLMResult(
                    model_name=self.model,
                    user_input=text,
                    model_result=[],
                    success=False,
                    error_message=repr(e),
                )

            span.set_attributes(result.to_otel_attributes())

            if not result.success:
                span.set_status(Status(StatusCode.ERROR, str(result.error_message)))
            else:
                span.set_status(Status(StatusCode.OK))

            return result
