import os
import json
import argparse
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from otel_setup import get_trace_logs, get_span_details

def create_dash_app(trace_id: str, log_dir: str = "logs/traces"):
    app = Dash(__name__)

    # 1. Fetch the initial spans
    spans = get_trace_logs(trace_id, log_dir)
    
    if not spans:
        print(f"Error: No trace data found for ID: {trace_id}")
        return None

    # 2. Prepare Gantt Data
    chart_data = []
    for span in spans:
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

    df = pd.DataFrame(chart_data)
    df["Start"] = pd.to_datetime(df["Start"])
    df["Finish"] = pd.to_datetime(df["Finish"])
    df["Duration"] = (df["Finish"] - df["Start"]).dt.total_seconds()
    df = df.sort_values(by=["Start", "Duration"], ascending=[True, False])
    df["Display Name"] = df.apply(lambda x: f"{x['Task']} ({x['Span ID'][-4:]})", axis=1)

    # Create the figure
    fig = px.timeline(
        df, 
        x_start="Start", 
        x_end="Finish", 
        y="Display Name", 
        color="Task",
        hover_data=["Span ID", "Parent ID"],
        title=f"Detailed Trace Timeline: {trace_id}"
    )
    fig.update_yaxes(categoryorder='array', categoryarray=df["Display Name"].tolist() ,autorange="reversed", title="")
    fig.update_layout(clickmode='event+select', margin=dict(l=20, r=20, t=40, b=20))

    # 3. App Layout
    app.layout = html.Div([
        # Top Half: Gantt Chart
        html.Div([
            dcc.Graph(
                id='gantt-chart',
                figure=fig,
                style={'height': '100%'}
            )
        ], style={'height': '50vh', 'borderBottom': '2px solid #ccc'}),

        # Bottom Half: Details Pane
        html.Div([
            html.H4("Span Details", style={'margin': '10px 0'}),
            html.P("Click a bar in the timeline to view its input/output details.", id='instruction-text'),
            html.Pre(
                id='details-output',
                style={
                    'whiteSpace': 'pre-wrap',
                    'wordBreak': 'break-all',
                    'backgroundColor': '#f8f9fa',
                    'padding': '15px',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'maxHeight': '35vh',
                    'overflowY': 'auto'
                }
            )
        ], style={'height': '45vh', 'padding': '20px', 'fontFamily': 'monospace'})
    ], style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column'})

    # 4. Interaction Callback
    @app.callback(
        [Output('details-output', 'children'),
         Output('instruction-text', 'children')],
        Input('gantt-chart', 'clickData')
    )
    def update_span_details(clickData):
        if not clickData:
            return "", "Click a bar in the timeline to view its input/output details."
        
        # Extract Span ID from clickData custom_data
        # customdata[0] corresponds to the first element in hover_data
        span_id = clickData['points'][0]['customdata'][0]
        
        details = get_span_details(trace_id, span_id, log_dir)
        
        def json_serializable(obj):
            """JSON serializer for objects not serializable by default json code"""
            if hasattr(obj, 'tolist'): # Handles numpy arrays
                return obj.tolist()
            if hasattr(obj, 'item'): # Handles numpy scalars (float64, int64, etc)
                return obj.item()
            return str(obj)

        if details:
            return json.dumps(details, indent=2, ensure_ascii=False, default=json_serializable), f"Viewing details for Span: {span_id}"
        return "No extended details found for this span.", f"Span ID: {span_id} (No file found)"

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_id")
    parser.add_argument("--log_dir", default="logs/traces") 
    args = parser.parse_args()

    dash_app = create_dash_app(args.trace_id, args.log_dir)
    if dash_app:
        dash_app.run(debug=True, port=8050)