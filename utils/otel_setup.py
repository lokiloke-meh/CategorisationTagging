import atexit
import glob
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class SpanParquetWriter:
    """
    Handles asynchronous writing of individual span details to Parquet files.
    """

    def __init__(self, max_workers: int = 4, file_format: str = "parquet"):
        # Since files are unique per span_id, we can use multiple workers
        # to perform disk I/O and Parquet serialization in parallel.
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self.file_format = file_format

    @property
    def file_format(self) -> str:
        return self._file_format

    @file_format.setter
    def file_format(self, value: str):
        if value.lower() not in ["json", "parquet"]:
            raise ValueError("file_format must be either 'json' or 'parquet'")
        self._file_format = value.lower()

    def write_span(self, base_file_path: str, data: dict):
        """Schedules a background write for a specific span."""
        self._executor.submit(self._do_write, base_file_path, data)

    def _do_write(self, base_path: str, data: dict):
        try:
            full_path = f"{base_path}.{self.file_format}"
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if self.file_format == "parquet":
                # PyArrow/Pandas can struggle with complex nested structures, multi-dimensional
                # arrays, or empty dictionaries when inferring Parquet schemas.
                # To ensure reliability for logging, we serialize complex nested collections
                # into JSON strings at the top level.
                processed_data = {}
                for k, v in data.items():
                    # If it's a collection (dict or list), serialize it to a JSON string.
                    # This avoids "Can only convert 1-dimensional array values" and
                    # "Conversion failed for column ... with type object" errors.
                    if isinstance(v, (dict, list)):
                        processed_data[k] = json.dumps(
                            v, default=str, ensure_ascii=False
                        )
                    else:
                        processed_data[k] = v

                df = pd.DataFrame([processed_data])
                df.to_parquet(
                    full_path, index=False, engine="pyarrow", compression="snappy"
                )
            else:
                with open(full_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            print(f"Error writing {self.file_format} span to {full_path}: {e}")

    def shutdown(self):
        self._executor.shutdown(wait=True)


# Singleton instance
span_writer = SpanParquetWriter(file_format="json")

# Register shutdown to ensure all background threads finish writing before program exit
atexit.register(span_writer.shutdown)


class SessionContextSpanProcessor(SpanProcessor):
    """
    A custom SpanProcessor that injects session context attributes into every span on start.
    """

    def __init__(self, session_id: str, session_dir_path: str):
        self.session_id = session_id
        self.session_dir_path = session_dir_path

    def on_start(self, span, parent_context=None):
        span.set_attribute("session.id", self.session_id)
        span.set_attribute("session.dir_path", self.session_dir_path)


class TraceIdFileSpanExporter(SpanExporter):
    """
    Exports OpenTelemetry spans to files named {trace_id}.json within a session folder.
    Each file contains a valid JSON array of spans for that specific trace.
    """

    def __init__(self, log_dir: str, session_id: str):
        self.log_dir = os.path.abspath(log_dir)
        self.session_id = session_id
        os.makedirs(self.log_dir, exist_ok=True)

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            span_context = span.get_span_context()
            if not span_context.is_valid:
                continue

            trace_id = format(span_context.trace_id, "032x")
            file_path = os.path.join(self.log_dir, f"{trace_id}.json")

            # Read existing spans, append the new one, and write back as a valid JSON array
            trace_data = []
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        trace_data = json.load(f)
                except json.JSONDecodeError:
                    trace_data = []

            span_dict = json.loads(span.to_json())
            # Explicitly inject the session_id into the JSON for easier lookup
            span_dict["session_id"] = self.session_id
            trace_data.append(span_dict)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=4)

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def get_trace_logs(trace_id: str, log_dir: str = "logs/traces") -> list[dict]:
    """
    Retrieves all spans for a given trace_id by searching recursively for the JSON file.
    """
    # Strip '0x' prefix if present to ensure the glob pattern matches filenames on disk
    trace_id = (
        trace_id[2:]
        if isinstance(trace_id, str) and trace_id.startswith("0x")
        else trace_id
    )

    pattern = os.path.join(os.path.abspath(log_dir), "**", f"{trace_id}.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return []

    try:
        with open(files[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except OSError, json.JSONDecodeError:
        return []


def get_span_details(
    trace_id: str, span_id: str, log_dir: str = "logs/traces"
) -> dict | None:
    """
    Reads the detail file for a specific span (supports Parquet or JSON).
    Returns the data as a dictionary or None if not found.
    """
    # Strip '0x' prefixes to match the hex-formatted directory and filenames on disk
    span_id = (
        span_id[2:]
        if isinstance(span_id, str) and span_id.startswith("0x")
        else span_id
    )
    trace_id = (
        trace_id[2:]
        if isinstance(trace_id, str) and trace_id.startswith("0x")
        else trace_id
    )

    # Matches: session_dir/*_details/trace_id/span_id.* (llm/chain/eval)
    pattern = os.path.join(
        os.path.abspath(log_dir), "*", "*_details", trace_id, f"{span_id}.*"
    )
    files = glob.glob(pattern)
    if not files:
        return None

    try:
        if files[0].endswith(".parquet"):
            df = pd.read_parquet(files[0])
            # Parquet files for single spans contain exactly one row
            data = df.to_dict(orient="records")[0]

            # Attempt to restore JSON strings back to dictionaries/lists for the UI
            for k, v in data.items():
                if isinstance(v, str):  # noqa: SIM102
                    # Quick heuristic to see if it might be a serialized JSON collection
                    if (v.startswith("{") and v.endswith("}")) or (
                        v.startswith("[") and v.endswith("]")
                    ):
                        try:
                            data[k] = json.loads(v)
                        except json.JSONDecodeError, TypeError:
                            pass
            return data
        else:
            with open(files[0], "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"Error reading span details for {span_id}: {e}")
        return None


def cleanup_trace_logs(max_age_days: int = 7, log_dir: str = "logs/traces") -> int:
    """
    Deletes session directories older than max_age_days based on the directory name.
    """
    log_dir_path = os.path.abspath(log_dir)
    if not os.path.exists(log_dir_path):
        return 0

    cutoff_date = datetime.now() - timedelta(days=max_age_days)  # noqa: DTZ005
    deleted_count = 0

    for session_folder in os.listdir(log_dir_path):
        session_path = os.path.join(log_dir_path, session_folder)
        if os.path.isdir(session_path):
            try:
                # Extract date from session folder name (format: YYYYMMDD_HHMMSS)
                folder_date_str = session_folder.split("_")[0]
                folder_date = datetime.strptime(  # noqa: DTZ007
                    folder_date_str, "%Y%m%d"
                )
                if folder_date < cutoff_date:
                    shutil.rmtree(session_path)
                    deleted_count += 1
            except ValueError, IndexError:
                continue
    return deleted_count


def setup_otel_tracing(
    log_dir: str = "logs/traces", enabled: bool = True, log_format: str = "parquet"
):
    """
    Initializes OpenTelemetry. Can be toggled via the 'enabled' parameter
    or the 'OTEL_ENABLED' environment variable.
    """
    env_enabled = os.environ.get("OTEL_ENABLED", "true").lower() == "true"

    if not enabled or not env_enabled:
        return

    span_writer.file_format = log_format.lower()

    # Create a unique session ID based on the current timestamp
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    session_dir = os.path.join(log_dir, session_id)

    # Define the service resource (metadata about your application)
    resource = Resource(attributes={SERVICE_NAME: "llama-application"})

    # Initialize the TracerProvider
    provider = TracerProvider(resource=resource)

    # Use the custom exporter that segments by trace_id within the session folder
    exporter = TraceIdFileSpanExporter(session_dir, session_id)

    # BatchSpanProcessor sends spans in batches to reduce overhead
    span_processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(span_processor)

    # Add the session processor to inject session metadata into every span
    session_processor = SessionContextSpanProcessor(session_id, session_dir)
    provider.add_span_processor(session_processor)

    # Register the provider globally so that any call to
    # trace.get_tracer() uses this configuration.
    trace.set_tracer_provider(provider)
