from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

# Ensure utils directory is on sys.path for otel_setup import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from otel_setup import get_span_details, get_trace_logs  # noqa: E402


def json_serializable(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default JSON encoder."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def render_trace_html(trace_id: str, log_dir: str) -> str:
    """Generate interactive HTML Gantt chart for a trace.

    Returns the HTML string — callers decide whether to write to disk or serve via Flask.
    """
    if not trace_id:
        return _empty_html("No Trace ID Available", "This article does not have an OpenTelemetry trace associated with it.")

    spans = get_trace_logs(trace_id, log_dir)
    if not spans:
        return _empty_html(
            "No Trace Data Found",
            f"Trace ID: <code>{trace_id}</code>\nSearched in: <code>{log_dir}</code>",
        )

    chart_data, span_details_map = [], {}
    for span in spans:
        name = span.get("name", "Unknown Span")
        start = span.get("start_time")
        end = span.get("end_time")
        span_id = span.get("context", {}).get("span_id", "Unknown")

        if not (start and end):
            continue

        chart_data.append({
            "Task": name, "Start": start, "Finish": end,
            "Span ID": span_id, "Parent ID": span.get("parent_id", "None"),
        })
        details = get_span_details(trace_id, span_id, log_dir)
        if details:
            span_details_map[span_id] = details

    if not chart_data:
        return _empty_html("No Timing Data", f"No valid timing spans found in trace: <code>{trace_id}</code>")

    df = pd.DataFrame(chart_data)
    df["Start"] = pd.to_datetime(df["Start"])
    df["Finish"] = pd.to_datetime(df["Finish"])
    df["Duration"] = (df["Finish"] - df["Start"]).dt.total_seconds()
    df = df.sort_values(by=["Start", "Duration"], ascending=[True, False])
    df["Display Name"] = df.apply(lambda x: f"{x['Task']} ({x['Span ID'][-4:]})", axis=1)

    fig = px.timeline(
        df, x_start="Start", x_end="Finish", y="Display Name", color="Task",
        hover_data=["Span ID", "Parent ID"], title=f"Detailed Trace Timeline: {trace_id}",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=df["Display Name"].tolist(), autorange="reversed", title="")
    fig.update_layout(clickmode="event+select", margin={"l": 20, "r": 20, "t": 60, "b": 20}, title_x=0.03)

    details_json = json.dumps(span_details_map, indent=2, ensure_ascii=False, default=json_serializable)
    return _html_template(trace_id, fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="plotly-gantt"), details_json)




def generate_trace_gantt(trace_id: str, log_dir: str = "logs/traces") -> None:
    """Read trace logs and save interactive HTML Gantt chart to disk."""
    if not trace_id:
        print("Usage: python visualize_trace_html.py <trace_id>")
        return

    html_content = render_trace_html(trace_id, log_dir)
    output_dir = Path(log_dir) / "gantt_charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"trace_gantt_{trace_id}.html"
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Chart saved: {output_path.resolve()}")


def _empty_html(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html><body style="font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #f9fafb;">
<div style="text-align: center;"><h3>{title}</h3><p>{body}</p></div>
</body></html>"""


def _html_template(trace_id: str, plotly_html: str, details_json: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Trace Visualization: {trace_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #f8fafc; min-height: 100vh; }}
        #chart-container {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 20px; overflow-x: auto; padding: 16px; }}
        #details-pane {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); padding: 16px 20px; }}
        #details-output {{ white-space: pre-wrap; word-break: break-all; background: #f9fafb; padding: 14px; border: 1px solid #e5e7eb; border-radius: 6px; margin: 0; font-family: ui-monospace, monospace; font-size: 13px; color: #1e293b; }}
        h4 {{ margin: 0 0 8px; font-size: 15px; font-weight: 600; color: #0f172a; }}
        #instruction-text {{ margin: 0 0 8px; color: #6b7280; font-size: 13px; }}
    </style>
</head>
<body>
    <div id="chart-container">{plotly_html}</div>
    <div id="details-pane">
        <h4>Span Details</h4>
        <p id="instruction-text">Click a bar in the timeline to view its input/output details.</p>
        <pre id="details-output">// Select a span bar in the timeline above</pre>
    </div>
    <script type="text/javascript">
        const spanDetails = {details_json};
        function attachListener() {{
            const plotEl = document.getElementById('plotly-gantt');
            if (plotEl && plotEl.on) {{
                plotEl.on('plotly_click', function(data) {{
                    const spanId = data.points[0].customdata[0];
                    const details = spanDetails[spanId];
                    const output = document.getElementById('details-output');
                    const instruction = document.getElementById('instruction-text');
                    if (details) {{
                        output.textContent = JSON.stringify(details, null, 2);
                        instruction.textContent = "Viewing details for Span: " + spanId;
                    }} else {{
                        instruction.textContent = "Span ID: " + spanId + " (No file found)";
                        output.textContent = "// No detailed span log file found on disk for " + spanId;
                    }}
                }});
            }} else {{ setTimeout(attachListener, 100); }}
        }}
        attachListener();
    </script>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_id")
    parser.add_argument("--log_dir", default="logs/traces")
    args = parser.parse_args()
    generate_trace_gantt(args.trace_id, args.log_dir)
