import json

import requests
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from utils.llm_models import ChatPayloadAdapter, LLMResult, OllamaPayloadAdapter

tracer = trace.get_tracer(__name__)


def _extract_ollama_metrics(model_response: dict, thinking_mode: bool) -> dict:
    """Extracts common performance metrics from Ollama API responses."""
    metrics = {}

    if model_response.get("prompt_eval_count"):
        metrics["prompt_token_count"] = model_response.get("prompt_eval_count")

    if model_response.get("prompt_eval_duration"):
        metrics["prompt_eval_duration_secs"] = (
            model_response.get("prompt_eval_duration") / 10**9
        )

    if model_response.get("load_duration"):
        metrics["model_load_duration_secs"] = (
            model_response.get("load_duration") / 10**9
        )

    if model_response.get("eval_count") and model_response.get("eval_duration"):
        metrics["response_tokens_per_second"] = (
            model_response.get("eval_count")
            * 10**9
            / model_response.get("eval_duration")
        )
        print(f"response_tokens_per_second = {metrics['response_tokens_per_second']}")

    if thinking_mode:
        return_thinking = model_response["message"].get("thinking")

        # return the model thought process if available
        if return_thinking:
            metrics["model_thought"] = return_thinking
        else:
            print("Model thinking not found")

    return metrics


class ollamaChat:
    def __init__(self, model: str = "gemma4:e4b"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"  # https://docs.ollama.com/api/chat
        self._payload_adapter: ChatPayloadAdapter = OllamaPayloadAdapter()

    def chat_with_expected_type_result(
        self, chat_payload: dict, result_type: str
    ) -> LLMResult:

        # Normalize payload via adapter (caller passes common format)
        chat_payload = self._payload_adapter.normalize(chat_payload)
        chat_payload["model"] = self.model

        if result_type == "string":
            chat_payload["format"] = None
        elif result_type == "dict":
            if not chat_payload.get("format"):
                raise OSError(
                    "Expected to pass json schema if we want to get json results"
                )
        else:
            raise ValueError(f"Not supported result_type {result_type}")

        # check if thinking mode is turned on
        thinking_mode = bool(chat_payload.get("think"))

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

        model_options = chat_payload.get("options", {})

        # Group shared context into a single dictionary to avoid repeating parameters
        base_result_data = {
            "model_name": self.model,
            "user_input": user_input,
            "system_prompt": system_prompt,
            "model_options": model_options,
        }

        with tracer.start_as_current_span(f"ollama.chat.{result_type}") as span:
            span.set_attribute("llm.request.model", self.model)

            try:
                response = requests.post(self.url, json=chat_payload)
                if response.status_code == 200:
                    model_response = json.loads(response._content.decode("utf-8"))

                    if result_type == "string":
                        result = self.parse_ollama_api_str_response(
                            model_response, thinking_mode, base_result_data
                        )
                    else:
                        result = self.parse_ollama_api_json_response(
                            model_response, thinking_mode, base_result_data
                        )
                else:
                    result = LLMResult(
                        **base_result_data,
                        model_result={},
                        success=False,
                        error_message=f"Requests returned HTTPError {response.status_code}",
                    )

            except Exception as e:  # noqa: BLE001
                span.record_exception(e)
                result = LLMResult(
                    **base_result_data,
                    model_result={},
                    success=False,
                    error_message=repr(e),
                )

            # Apply the standardized attributes to the span
            span.set_attributes(result.to_otel_attributes())

            if not result.success:
                span.set_status(Status(StatusCode.ERROR, str(result.error_message)))
            else:
                span.set_status(Status(StatusCode.OK))

            return result

    def parse_ollama_api_json_response(
        self, model_response: dict, thinking_mode: bool, base_result_data: dict
    ) -> LLMResult:

        return_result = base_result_data.copy()

        try:
            return_result.update(_extract_ollama_metrics(model_response, thinking_mode))

            return_result["model_result"] = json.loads(
                model_response["message"]["content"]
            )

        except json.JSONDecodeError:
            error_message = (
                f"Failed to decode JSON. Full model reply: {model_response['message']}"
            )
            print(error_message)

            return LLMResult(
                **return_result,
                success=False,
                error_message=error_message,
                model_result="",
            )

        except Exception as e:  # noqa: BLE001
            return LLMResult(
                **return_result, success=False, error_message=e, model_result=""
            )

        return LLMResult(**return_result)

    def parse_ollama_api_str_response(
        self, model_response: dict, thinking_mode: bool, base_result_data: dict
    ) -> LLMResult:

        return_result = base_result_data.copy()

        try:
            return_result.update(_extract_ollama_metrics(model_response, thinking_mode))

            return_result["model_result"] = model_response["message"]["content"]

        except Exception as e:  # noqa: BLE001
            return LLMResult(
                **return_result, success=False, error_message=e, model_result=""
            )

        return LLMResult(**return_result)


class ollamaEmbeddings:
    """Embedding adapter for Ollama servers.

    Example:
        model = ollamaEmbeddings(model="qwen3-embedding:8b")
        result = model.get_embeddings(text="search query")
    """

    def __init__(self, model: str = "qwen3-embedding:8b"):
        self.model = model
        self.url = "http://localhost:11434/api/embed"

    def get_embeddings(self, text: str) -> LLMResult:
        request_payload = {
            "model": self.model,
            "input": text,
        }

        with tracer.start_as_current_span("ollama.embed") as span:
            span.set_attribute("llm.request.model", self.model)
            try:
                response = requests.post(self.url, json=request_payload)
                if response.status_code == 200:
                    api_response = json.loads(response._content.decode("utf-8"))

                    embeddings = api_response.get("embeddings")

                    # Validation: Ensure embeddings is a list of lists of floats/ints
                    if not isinstance(embeddings, list):
                        raise TypeError("Ollama response embeddings must be a list.")

                    for i, vector in enumerate(embeddings):
                        if not isinstance(vector, list):
                            raise TypeError(
                                f"Embedding at index {i} must be a list, got {type(vector).__name__}."
                            )
                        if not all(isinstance(x, (float, int)) for x in vector):
                            raise ValueError(
                                f"Embedding at index {i} contains non-numeric values."
                            )

                    result = LLMResult(
                        **_extract_ollama_metrics(api_response, thinking_mode=False),
                        model_name=self.model,
                        user_input=text,
                        model_result=embeddings[0],
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
