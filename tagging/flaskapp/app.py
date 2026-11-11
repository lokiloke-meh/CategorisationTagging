from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, url_for

from tagging.flaskapp import services
from tagging.flaskapp.renderers.pipeline_renderer import render_pipeline_html
from tagging.flaskapp.renderers.trace_renderer import render_trace_html


def create_app(data_store_dir: str | None = None) -> Flask:
    """Application factory for the Tagging Notice Visualizer."""
    app = Flask(__name__)

    # Configure data store directory
    resolved_dir = str(services.resolve_data_store_dir(data_store_dir))
    app.config["DATA_STORE_DIR"] = resolved_dir

    @app.context_processor
    def inject_global_template_vars():
        current_data_dir = Path(app.config["DATA_STORE_DIR"])
        return {
            "data_dir_name": current_data_dir.name,
            "data_dir_path": str(current_data_dir),
        }

    @app.route("/", methods=["GET"])
    def index():
        data_dir = app.config["DATA_STORE_DIR"]
        available_topics = services.list_available_topics(custom_path=data_dir)

        # Retrieve filter parameters
        selected_topics = request.args.getlist("topics")
        selected_tags = request.args.getlist("tags")
        since_date = request.args.get("since", "").strip() or None

        has_queried = "topics" in request.args

        articles = []
        available_tags = []

        if selected_topics:
            available_tags = services.get_tags_for_topics(
                selected_topics, custom_path=data_dir
            )
            if has_queried:
                # If no specific tags were selected, filter_tags is None (meaning include all)
                filter_tags = selected_tags if selected_tags else None
                articles = services.run_report(
                    topics=selected_topics,
                    filter_tags=filter_tags,
                    since_date=since_date,
                    target_levels=["High"],
                    custom_path=data_dir,
                )
        elif has_queried:
            # User submitted form without selecting any topic
            articles = []

        from urllib.parse import urlencode
        back_params = request.args.to_dict(flat=False)
        query_string = urlencode(back_params, doseq=True) if any(v for v in back_params.values()) else ""

        return render_template(
            "index.html",
            available_topics=available_topics,
            selected_topics=selected_topics,
            available_tags=available_tags,
            selected_tags=selected_tags,
            since_date=since_date,
            articles=articles,
            has_queried=has_queried,
            query_string=query_string,
        )

    @app.route("/api/tags", methods=["GET"])
    def api_tags():
        """Returns JSON list of tags for the provided topics list."""
        data_dir = app.config["DATA_STORE_DIR"]
        topics = request.args.getlist("topics")
        if not topics:
            return jsonify({"tags": []})

        tags = services.get_tags_for_topics(topics, custom_path=data_dir)
        return jsonify({"tags": tags})

    @app.route("/article/<topic>/<file_hash>", methods=["GET"])
    def article_detail(topic: str, file_hash: str):
        """Displays article overview with 3 tabs (Trace Gantt, Pipeline DAG, Raw Content)."""
        data_dir = app.config["DATA_STORE_DIR"]
        article = services.get_article_details(topic, file_hash, custom_path=data_dir)
        if not article:
            abort(404, description=f"Article '{file_hash}' not found in topic '{topic}'.")

        from urllib.parse import urlencode
        back_params = request.args.to_dict(flat=False)
        filtered = {k: v for k, v in back_params.items() if v}
        query_string = urlencode(filtered, doseq=True) if filtered else ""
        back_url = f"{url_for('index')}?{query_string}" if query_string else url_for('index')

        article["raw_content_html"] = article.get("raw_content", "").replace("\n", "<br>")
        return render_template("article.html", article=article, back_url=back_url)

    @app.route("/article/<topic>/<file_hash>/trace", methods=["GET"])
    def article_trace(topic: str, file_hash: str):
        """Renders standalone Plotly Gantt HTML for iframe embedding or direct viewing."""
        data_dir = app.config["DATA_STORE_DIR"]
        article = services.get_article_details(topic, file_hash, custom_path=data_dir)
        if not article:
            abort(404, description=f"Article '{file_hash}' not found.")

        trace_id = article.get("trace_id", "")
        log_dir = Path(article["topic_dir"]) / "traces"
        trace_html = render_trace_html(trace_id, str(log_dir))
        return Response(trace_html, mimetype="text/html")

    @app.route("/article/<topic>/<file_hash>/pipeline", methods=["GET"])
    def article_pipeline(topic: str, file_hash: str):
        """Renders standalone Mermaid Pipeline HTML for iframe embedding or direct viewing."""
        data_dir = app.config["DATA_STORE_DIR"]
        article = services.get_article_details(topic, file_hash, custom_path=data_dir)
        if not article:
            abort(404, description=f"Article '{file_hash}' not found.")

        pipeline_id = article.get("pipeline_id", "")
        registry_path = Path(article["topic_dir"]) / "pipeline_registry.json"
        pipeline_html = render_pipeline_html(pipeline_id, registry_path)
        return Response(pipeline_html, mimetype="text/html")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
