"""Shared telemetry helpers to eliminate duplicate path construction."""

from __future__ import annotations

import json
import os

from opentelemetry import trace


def get_trace_and_span_hex() -> tuple[str, str]:
    """Return (trace_id_hex, span_id_hex) for the current span, or ('untraced', '0') if none."""
    current_span = trace.get_current_span()
    span_context = current_span.get_span_context()
    return (
        format(span_context.trace_id, "032x") if span_context.is_valid else "untraced",
        format(span_context.span_id, "016x") if span_context.is_valid else "0",
    )


def build_detail_path(data_type: str) -> tuple[str, dict]:
    """Build the full path for span detail writing and return (detail_path, attributes_dict).

    Writes the data entry to disk via span_writer and returns a dict with 'details_path' attribute.
    Returns empty attrs if no active recording span exists.
    """
    from utils.otel_setup import span_writer

    current_span = trace.get_current_span()
    if not current_span.is_recording():
        return "", {}

    trace_id_hex, span_id_hex = get_trace_and_span_hex()
    session_directory = "logs"
    if hasattr(current_span, "attributes"):
        session_directory = current_span.attributes.get("session.dir_path", "logs")

    details_output_dir = os.path.join(session_directory, f"{data_type}_details")
    os.makedirs(details_output_dir, exist_ok=True)
    span_detail_file_path = os.path.join(details_output_dir, trace_id_hex, span_id_hex)

    return span_detail_file_path, {
        "details_path": f"{span_detail_file_path}.{span_writer.file_format}"
    }


def record_span_details(
    current_span,
    data_type: str,
    entry: dict,
    large_field_names: set[str],
    attribute_prefix: str,
) -> dict:
    """Record span details to disk and return attributes dict.

    Args:
        current_span: Current OpenTelemetry span context
        data_type: Type identifier for directory naming ("llm", "chain", "eval")
        entry: Dict containing trace_id, span_id, data_type, and all result fields
        large_field_names: Set of field names to exclude from span attributes (kept in detail file only)
        attribute_prefix: Attribute key prefix (e.g., "llm.", "chain.", "eval.")

    Returns:
        Dict mapping attribute keys to values for span.set_attributes()
    """
    from utils.otel_setup import span_writer

    if not current_span.is_recording():
        return {}

    trace_id_hex, span_id_hex = get_trace_and_span_hex()
    session_directory = "logs"
    if hasattr(current_span, "attributes"):
        session_directory = current_span.attributes.get("session.dir_path", "logs")

    details_output_dir = os.path.join(session_directory, f"{data_type}_details")
    os.makedirs(details_output_dir, exist_ok=True)
    span_detail_file_path = os.path.join(details_output_dir, trace_id_hex, span_id_hex)

    entry["trace_id"] = trace_id_hex
    entry["span_id"] = span_id_hex
    entry["data_type"] = data_type

    span_writer.write_span(span_detail_file_path, entry)

    otel_attributes: dict[str, str | bool | float | int] = {}

    for field_name, value in entry.items():
        if value is None or field_name in large_field_names:
            continue

        attribute_key = f"{attribute_prefix}{field_name}"
        if isinstance(value, (str, bool, int, float)):
            otel_attributes[attribute_key] = value
        else:
            otel_attributes[attribute_key] = json.dumps(
                value, default=str, ensure_ascii=False
            )

    details_path_attribute_key = f"{attribute_prefix}details_path"
    otel_attributes[details_path_attribute_key] = (
        f"{span_detail_file_path}.{span_writer.file_format}"
    )

    return otel_attributes
