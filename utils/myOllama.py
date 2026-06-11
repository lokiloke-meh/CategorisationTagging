import os
import requests
import json
from abc import ABC, abstractmethod
from typing import Optional, Any
from dataclasses import dataclass, asdict, field, fields
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from utils.otel_setup import span_writer

tracer = trace.get_tracer(__name__)

def _extract_ollama_metrics(model_response: dict, thinking_mode:bool) -> dict:
    """Extracts common performance metrics from Ollama API responses."""
    metrics = {}

    if model_response.get("prompt_eval_count"):
        metrics["prompt_token_count"] = model_response.get("prompt_eval_count")
    
    if model_response.get("prompt_eval_duration"):
        metrics["prompt_eval_duration_secs"] = model_response.get("prompt_eval_duration") / 10**9

    if model_response.get("load_duration"):
        metrics["model_load_duration_secs"] = model_response.get("load_duration") / 10**9

    if model_response.get("eval_count") and model_response.get("eval_duration"):
        metrics["response_tokens_per_second"] = model_response.get("eval_count") * 10**9 / model_response.get("eval_duration")
        print(f"response_tokens_per_second = {metrics['response_tokens_per_second']}")

    if thinking_mode:
        return_thinking = model_response["message"].get("thinking")
                    
        # return the model thought process if available
        if return_thinking:
            metrics["model_thought"] = return_thinking
        else:
            print("Model thinking not found")
    
    return metrics


@dataclass
class LLMResult:
    """Standardized output directly from an LLM."""
    model_name: str
    user_input: str
    model_result: dict | str
    model_options: Optional[dict[str, Any]] = None
    system_prompt: Optional[str] = None
    # Using metadata to flag fields that should be excluded from standard serialization
    model_thought: Optional[str] = field(default=None, metadata={"exclude_from_dict": True})
    model_load_duration_secs: Optional[float] = None
    prompt_token_count: Optional[float] = None
    prompt_eval_duration_secs: Optional[float] = None
    response_tokens_per_second: Optional[float] = None
    success: bool = True 
    error_message: Optional[str] = None 

    def to_dict(self) -> dict:
        """Converts the dataclass to a dictionary suitable for JSON serialization."""
        # Filter fields based on metadata to avoid "hardcoding" field names in logic
        excluded_fields = {f.name for f in fields(self) if f.metadata.get("exclude_from_dict")}
        full_data = asdict(self)
        data = {k: v for k, v in full_data.items() if k not in excluded_fields}
        
        # Handle potential non-string error_messages (like Exception objects) 
        # that might have been passed during instantiation.
        if data.get("error_message") is not None:
            data["error_message"] = str(data["error_message"])
        return data

    def to_otel_attributes(self) -> dict[str, str | bool | float | int]:
        """
        Converts the LLMResult to OpenTelemetry attributes.
        To keep trace logs manageable, large text fields (result and thought)
        are saved to a separate file, and the path is stored in the span.
        """
        # Determine unique file path using trace context for naming
        span = trace.get_current_span()
        if not span.is_recording():
            return {}

        attributes = {}
        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, "032x") if span_context.is_valid else "untraced"
        span_id = format(span_context.span_id, "016x") if span_context.is_valid else "untraced"

        # Retrieve the session directory from span attributes if available
        session_dir = "logs"
        if hasattr(span, "attributes"):
            session_dir = span.attributes.get("session.dir_path", "logs")

        # The base path for the span detail file, WITHOUT extension
        output_dir = os.path.join(session_dir, "llm_details")
        os.makedirs(output_dir, exist_ok=True)
        base_detail_path = os.path.join(output_dir, trace_id, span_id)
        
        entry = {
            "trace_id": trace_id,
            "span_id": span_id,
            "data_type": "LLMResult",
            "system_prompt": self.system_prompt,
            "user_input": self.user_input,
            "model_options": self.model_options,
            "model_name": self.model_name,
            "model_result": self.model_result,
            "model_thought": self.model_thought,
            "model_load_duration_secs": self.model_load_duration_secs,
            "prompt_token_count": self.prompt_token_count,
            "prompt_eval_duration_secs": self.prompt_eval_duration_secs,
            "response_tokens_per_second": self.response_tokens_per_second,
            "success": self.success,
            "error_message": self.error_message
        }

        span_writer.write_span(base_detail_path, entry)

        attributes["llm.details_path"] = f"{base_detail_path}.{span_writer.file_format}"

        # Filter out large fields from the trace log attributes
        large_fields = {"model_result", "model_thought", "user_input", "system_prompt", "options"}
        for field_name, value in self.to_dict().items():
            if value is None or field_name in large_fields:
                continue
            
            key = f"llm.{field_name}"
            if isinstance(value, (str, bool, int, float)):
                attributes[key] = value
            else:
                # Convert dicts (like model_result) to JSON strings for span storage
                attributes[key] = json.dumps(value)
        return attributes


class AIChat(ABC):
    @abstractmethod
    def chat_with_expected_type_result(self, chat_payload, result_type):
        pass


class ollamaChat(AIChat):
    def __init__(self, model:str="gemma4:e4b"):
        self.model = model
        self.url = "http://localhost:11434/api/chat" # https://docs.ollama.com/api/chat

    
    def chat_with_expected_type_result(self, chat_payload:dict, result_type:str) -> LLMResult:
        
        # overwrite models
        chat_payload["model"] = self.model

        # https://docs.ollama.com/capabilities/structured-outputs

        if result_type == 'string':
            # force to not generate json if result_as_json == False
            chat_payload["format"] = None
        elif result_type == 'dict':
            # we expect to pass json schema if we want to get json results
            if not chat_payload.get("format"):
                raise IOError("Expected to pass json schema if we want to get json results")
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
            "model_options": model_options
        }

        with tracer.start_as_current_span(f"ollama.chat.{result_type}") as span:
            span.set_attribute("llm.request.model", self.model)
            
            try:
                response = requests.post(self.url, json=chat_payload)
                if response.status_code == 200:
                    model_response = json.loads(response._content.decode('utf-8'))

                    if result_type == 'string':
                        result = self.parse_ollama_api_str_response(model_response, thinking_mode, base_result_data)
                    else:
                        result = self.parse_ollama_api_json_response(model_response, thinking_mode, base_result_data)
                else:
                    result = LLMResult(
                        **base_result_data,
                        model_result={},
                        success=False,
                        error_message=f"Requests returned HTTPError {response.status_code}"
                    )

            except Exception as e:
                span.record_exception(e)
                result = LLMResult(
                    **base_result_data,
                    model_result={},
                    success=False,
                    error_message=repr(e)
                )
            
            # Apply the standardized attributes to the span
            span.set_attributes(result.to_otel_attributes())
            
            if not result.success:
                span.set_status(Status(StatusCode.ERROR, str(result.error_message)))
            else:
                span.set_status(Status(StatusCode.OK))
            
            return result



    def parse_ollama_api_json_response(self, model_response:dict, thinking_mode:bool, base_result_data:dict) -> LLMResult:

        return_result = base_result_data.copy()

        try:
            return_result.update(_extract_ollama_metrics(model_response, thinking_mode))

            return_result["model_result"] = json.loads(model_response["message"]["content"])
            
        except json.JSONDecodeError as json_error:
            error_message = f"Failed to decode JSON. Full model reply: {model_response['message']}"
            print(error_message)
            
            return LLMResult(**return_result, **{"success": False, "error_message": error_message, "model_result":""})

        except Exception as e:
            return LLMResult(**return_result, **{"success": False, "error_message": e, "model_result":""})
           
        
        return LLMResult(**return_result)
        


    def parse_ollama_api_str_response(self, model_response:dict, thinking_mode:bool, base_result_data:dict) -> LLMResult:
        
        return_result = base_result_data.copy()

        try:
            return_result.update(_extract_ollama_metrics(model_response, thinking_mode))

            return_result["model_result"] = model_response["message"]["content"]
            
        except Exception as e:
            return LLMResult(**return_result, **{"success": False, "error_message": e, "model_result":""})
        
        return LLMResult(**return_result)


class AIEmbeddings(ABC):
    @abstractmethod
    def get_embeddings(self, body_payload):
        pass


class ollamaEmbeddings(AIEmbeddings):
    def __init__(self, model:str="qwen3-embedding:8b"):
        self.model = model
        self.url = "http://localhost:11434/api/embed" # https://docs.ollama.com/api/embed

    def get_embeddings(self, body_payload: dict) -> LLMResult:
        # overwrite models
        body_payload["model"] = self.model

        model_options = body_payload.get("options", {})
        user_input = body_payload.get("input", "")
        
        return_result = {
            "model_name": self.model,
            "user_input": user_input,
            "model_options": model_options
        }

        with tracer.start_as_current_span("ollama.embed") as span:
            span.set_attribute("llm.request.model", self.model)
            try:
                response = requests.post(self.url, json=body_payload)
                if response.status_code == 200:
                    model_response = json.loads(response._content.decode('utf-8'))

                    return_result.update(_extract_ollama_metrics(model_response))

                    embeddings = model_response.get("embeddings")
                    
                    # Validation: Ensure embeddings is a list of lists of floats/ints
                    if not isinstance(embeddings, list):
                        raise ValueError("Ollama response 'embeddings' is not a list.")
                    
                    for i, vector in enumerate(embeddings):
                        if not isinstance(vector, list):
                            raise ValueError(f"Embedding at index {i} is not a list.")
                        if not all(isinstance(x, (float, int)) for x in vector):
                            raise ValueError(f"Embedding at index {i} contains non-numeric values.")

                    return_result["model_result"] = embeddings
                    result = LLMResult(**return_result)

                else:
                    result = LLMResult(
                        **return_result,
                        model_result={},
                        success=False,
                        error_message=f"Requests returned HTTPError {response.status_code}"
                    )

            except Exception as e:
                span.record_exception(e)
                result = LLMResult(
                    **return_result,
                    model_result={},
                    success=False,
                    error_message=repr(e)
                )

            # Apply the standardized attributes to the span
            span.set_attributes(result.to_otel_attributes())
            
            if not result.success:
                span.set_status(Status(StatusCode.ERROR, str(result.error_message)))
            else:
                span.set_status(Status(StatusCode.OK))
            
            return result