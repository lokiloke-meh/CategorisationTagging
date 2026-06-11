import sys
import os
import json
import pandas as pd
import argparse
import plotly.express as px
from otel_setup import get_trace_logs, get_span_details

def generate_trace_gantt(trace_id: str, log_dir: str = "logs/traces"):
    """
    Reads trace logs for a given ID and generates an interactive HTML Gantt chart.
    """
    # 1. Fetch the spans using the existing utility
    spans = get_trace_logs(trace_id, log_dir)
    
    if not spans:
        print(f"Error: No trace data found for ID: {trace_id}")
        print(f"Ensure the trace ID is correct and the logs exist in {log_dir}")
        return

    # 2. Prepare data for the timeline
    chart_data = []
    span_details_map = {}
    for span in spans:
        # Extract core timing and identification data
        # OTel span.to_json() typically provides ISO8601 timestamps
        name = span.get("name", "Unknown Span")
        start = span.get("start_time")
        end = span.get("end_time")
        
        span_id = span.get("context", {}).get("span_id", "Unknown")
        parent_id = span.get("parent_id", "None")

        if start and end:
            chart_data.append({
                "Task": name,
                "Start": start,
                "Finish": end,
                "Span ID": span_id,
                "Parent ID": parent_id
            })
            
            # Pre-fetch details for every span to embed in the offline HTML
            details = get_span_details(trace_id, span_id, log_dir)
            if details:
                span_details_map[span_id] = details
            else:
                # Useful for debugging during generation
                print(f"  [DEBUG] No details found for Span ID: {span_id}")

    if not chart_data:
        print("No valid timing data found in the trace spans.")
        return

    # 3. Create DataFrame and convert timestamps
    df = pd.DataFrame(chart_data)
    df["Start"] = pd.to_datetime(df["Start"])
    df["Finish"] = pd.to_datetime(df["Finish"])
    # Calculate duration for tie-breaking: longer spans (parents) should come first
    df["Duration"] = (df["Finish"] - df["Start"]).dt.total_seconds()
    
    # Sort by start time (earliest first) and then duration (longest first).
    # This ensures parents appear above children and the timeline flows chronologically.
    df = df.sort_values(by=["Start", "Duration"], ascending=[True, False])
    df["Display Name"] = df.apply(lambda x: f"{x['Task']} ({x['Span ID'][-4:]})", axis=1)

    # 4. Create the timeline using Plotly
    fig = px.timeline(
        df, 
        x_start="Start", 
        x_end="Finish", 
        y="Display Name", 
        color="Task",
        hover_data=["Span ID", "Parent ID"],
        title=f"Detailed Trace Timeline: {trace_id}"
    )

    # Ensure the Y-axis follows the DataFrame order (earliest at top)
    # 'trace' order ensures Plotly respects the sorted DataFrame sequence
    fig.update_yaxes(categoryorder='array', categoryarray=df["Display Name"].tolist() ,autorange="reversed", title="")
    fig.update_layout(clickmode='event+select', margin=dict(l=20, r=20, t=40, b=20))

    def json_serializable(obj):
        """JSON serializer for objects not serializable by default json code"""
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        return str(obj)

    details_json = json.dumps(span_details_map, indent=2, ensure_ascii=False, default=json_serializable)

    # 5. Create a standalone HTML template mirroring the Dash layout and interactivity
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Trace Visualization: {trace_id}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column; }}
            #chart-container {{ height: 50vh; border-bottom: 2px solid #ccc; }}
            #details-pane {{ height: 45vh; padding: 20px; box-sizing: border-box; display: flex; flex-direction: column; }}
            #details-output {{ 
                white-space: pre-wrap; word-break: break-all; background-color: #f8f9fa; 
                padding: 15px; border: 1px solid #ddd; border-radius: 5px; 
                overflow-y: auto; flex-grow: 1; margin: 0; font-family: monospace;
            }}
            h4 {{ margin: 0 0 10px 0; font-family: sans-serif; }}
            #instruction-text {{ margin: 0 0 10px 0; color: #666; font-family: sans-serif; }}
        </style>
    </head>
    <body>
        <div id="chart-container">
            {fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='plotly-gantt')}
        </div>
        <div id="details-pane">
            <h4>Span Details</h4>
            <p id="instruction-text">Click a bar in the timeline to view its input/output details.</p>
            <pre id="details-output"></pre>
        </div>

        <script type="text/javascript">
            const spanDetails = {details_json};
            
            function attachListener() {{
                const plotEl = document.getElementById('plotly-gantt');
                
                // Check if Plotly and the element are ready
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
                        }}
                    }});
                }} else {{
                    setTimeout(attachListener, 100);
                }}
            }}
            attachListener();
        </script>
    </body>
    </html>
    """

    output_dir = os.path.join("logs", "gantt_charts")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"trace_gantt_{trace_id}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Chart successfully generated: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_id")
    parser.add_argument("--log_dir", help="optional") 
    args = parser.parse_args()

    if not args.trace_id:
        print("Usage: python visualize_trace.py <trace_id>")
    else:
        generate_trace_gantt(args.trace_id, args.log_dir if args.log_dir else "logs/traces")
        
